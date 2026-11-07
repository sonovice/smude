use std::{sync::mpsc, time::Duration};
use wgpu::util::DeviceExt;

pub(super) struct Pipeline {
    layout: wgpu::BindGroupLayout,
    rows: wgpu::ComputePipeline,
    columns: wgpu::ComputePipeline,
    distances: wgpu::ComputePipeline,
}
impl Pipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, String> {
        let memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exact seed distance fields"),
            source: wgpu::ShaderSource::Wgsl(include_str!("distance.wgsl").into()),
        });
        let entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 3 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 0,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("distance buffers"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("distance pipeline"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let result = Self {
            layout: layout.clone(),
            rows: pipeline("rows"),
            columns: pipeline("columns"),
            distances: pipeline("distances"),
        };
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(error.to_string());
        }
        if let Some(error) = pollster::block_on(memory_scope.pop()) {
            return Err(error.to_string());
        }
        Ok(result)
    }
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        seeds: &[u32],
        dimensions: (u32, u32),
        cap: u16,
    ) -> Result<Vec<u32>, String> {
        let (width, height) = dimensions;
        let bytes = seeds.len() as u64 * 4;
        if width == 0 || height == 0 || seeds.len() != width as usize * height as usize {
            return Err("invalid distance field dimensions".into());
        }
        if bytes * 2 > device.limits().max_storage_buffer_binding_size {
            return Err("distance field exceeds GPU binding limit".into());
        }
        let memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("two binary seed masks"),
            contents: bytemuck::cast_slice(seeds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let make = |label: &str, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let prefix = make(
            "integer seed integral",
            bytes * 2,
            wgpu::BufferUsages::STORAGE,
        );
        let result = make(
            "packed exact distances",
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = make(
            "distance readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("distance dimensions"),
            contents: bytemuck::cast_slice(&[width, height, u32::from(cap), 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [&input, &prefix, &result, &params];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("distance page"),
            layout: &self.layout,
            entries: &entries,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("exact distance transform"),
        });
        for (pipeline, dispatch) in [
            (&self.rows, (height.div_ceil(64), 1)),
            (&self.columns, (width.div_ceil(64), 1)),
            (&self.distances, (width.div_ceil(8), height.div_ceil(8))),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("integer distance pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(dispatch.0, dispatch.1, 1);
        }
        encoder.copy_buffer_to_buffer(&result, 0, &readback, 0, bytes);
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = sender.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(20)),
            })
            .map_err(|e| e.to_string())?;
        receiver
            .recv_timeout(Duration::from_secs(2))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(error.to_string());
        }
        if let Some(error) = pollster::block_on(memory_scope.pop()) {
            return Err(error.to_string());
        }
        let mapped = readback.get_mapped_range(..).map_err(|e| e.to_string())?;
        let output = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        Ok(output)
    }
}
