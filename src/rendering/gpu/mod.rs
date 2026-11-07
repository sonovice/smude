//! Shared offscreen GPU implementation: Metal on Apple, Vulkan on Android/Linux,
//! and Direct3D on Windows. It consumes our own fitted geometry and formulas.
mod distance;
mod map;
#[cfg(test)]
mod tests;
use crate::geometry::Geometry;

use crate::rendering::RenderProfile;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use std::{
    sync::{Mutex, Once, OnceLock, mpsc},
    time::{Duration, Instant},
};
use wgpu::util::DeviceExt;

static CONTEXT: OnceLock<Result<Mutex<Context>, String>> = OnceLock::new();

pub(crate) fn prewarm() {
    static START: Once = Once::new();
    if CONTEXT.get().is_none() {
        START.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("smude-gpu-init".into())
                .spawn(|| {
                    let _ = CONTEXT.get_or_init(|| Context::new().map(Mutex::new));
                });
        });
    }
}

pub(crate) struct Output {
    pub color: Option<RgbImage>,
    pub tones: Option<(GrayImage, GrayImage, GrayImage)>,
    pub profile: RenderProfile,
}
struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    render: wgpu::ComputePipeline,
    horizontal: wgpu::ComputePipeline,
    paper: wgpu::ComputePipeline,
    adapter: String,
    timestamps: bool,
    distance: distance::Pipeline,
}

impl Context {
    fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err("Only a software GPU adapter is available".into());
        }
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter
                .limits()
                .max_storage_buffer_binding_size
                .min(256 * 1024 * 1024),
            max_buffer_size: adapter.limits().max_buffer_size.min(512 * 1024 * 1024),
            ..wgpu::Limits::default()
        };
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Smude image processing"),
            required_limits: limits,
            required_features: if timestamps {
                wgpu::Features::TIMESTAMP_QUERY
            } else {
                wgpu::Features::empty()
            },
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Smude bicubic and paper normalization"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Smude buffers"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Smude image pipeline"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let render = pipeline("render");
        let horizontal = pipeline("horizontal");
        let paper = pipeline("paper");
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(error.to_string());
        }
        let distance = distance::Pipeline::new(&device)?;
        if let Some(error) = pollster::block_on(memory_scope.pop()) {
            return Err(error.to_string());
        }
        Ok(Self {
            device,
            queue,
            layout,
            render,
            horizontal,
            paper,
            adapter: format!("{} ({:?})", info.name, info.backend),
            timestamps,
            distance,
        })
    }

    fn run(
        &self,
        image: &RgbImage,
        map: &map::Map,
        dimensions: (u32, u32),
        tone: bool,
        color: bool,
        mut profile: RenderProfile,
    ) -> Result<Output, String> {
        let (width, height) = dimensions;
        let pixels = u64::from(width) * u64::from(height);
        let source_pixels = u64::from(image.width()) * u64::from(image.height());
        let sizes = [
            source_pixels * 4,
            pixels * 4,
            (map.nodes.len() * 48) as u64,
            (map.dense.len() * 8) as u64,
            if tone { pixels * 8 } else { 8 },
        ];
        if sizes
            .into_iter()
            .any(|n| n > self.device.limits().max_storage_buffer_binding_size)
        {
            return Err("GPU buffer binding limit exceeded; use CPU for this image size".into());
        }
        let upload_start = Instant::now();
        let rgba: Vec<u32> = image
            .as_raw()
            .par_chunks_exact(3)
            .map(|p| {
                u32::from(p[0]) | (u32::from(p[1]) << 8) | (u32::from(p[2]) << 16) | 0xff000000
            })
            .collect();
        let memory_scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let storage = |label: &str, contents: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let empty = |label: &str, size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let source = storage("source", bytemuck::cast_slice(&rgba));
        let nodes = storage("inverse nodes", bytemuck::cast_slice(&map.nodes));
        let roots = storage("inverse roots", bytemuck::cast_slice(&map.roots));
        let dense = storage(
            "exact inverse coordinates",
            bytemuck::cast_slice(&map.dense),
        );
        let rendered = empty("rendered RGB and luminance", pixels * 4);
        let moments = empty(
            "horizontal exact integer statistics",
            if tone { pixels * 8 } else { 8 },
        );
        let tones = empty(
            "gray binary and normalized paper",
            if tone { pixels * 4 } else { 4 },
        );
        let values = [
            image.width(),
            image.height(),
            width,
            height,
            map.columns,
            map::TILE,
            0,
            0,
        ];
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("image dimensions"),
                contents: bytemuck::cast_slice(&values),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let buffers = [
            &source, &nodes, &roots, &dense, &rendered, &moments, &tones, &params,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("page buffers"),
            layout: &self.layout,
            entries: &entries,
        });
        let planes = u64::from(color) + u64::from(tone);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("processed image readback"),
            size: pixels * 4 * planes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let queries = self.timestamps.then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("image stage timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 6,
            })
        });
        let query_resolve = queries.as_ref().map(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stage query resolution"),
                size: 48,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let query_readback = queries.as_ref().map(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stage query readback"),
                size: 48,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        profile.upload_seconds = upload_start.elapsed().as_secs_f64();
        let wait_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Smude page"),
            });
        let mut pass = |pipeline: &wgpu::ComputePipeline, label: &str, index: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: queries.as_ref().map(|q| wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(index * 2),
                    end_of_pass_write_index: Some(index * 2 + 1),
                }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        };
        pass(&self.render, "bicubic reconstruction", 0);
        if tone {
            pass(&self.horizontal, "horizontal statistics", 1);
            pass(&self.paper, "paper normalization", 2);
        }
        let mut offset = 0;
        if color {
            encoder.copy_buffer_to_buffer(&rendered, 0, &readback, offset, pixels * 4);
            offset += pixels * 4;
        }
        if tone {
            encoder.copy_buffer_to_buffer(&tones, 0, &readback, offset, pixels * 4);
        }
        if let (Some(queries), Some(resolve), Some(read)) =
            (&queries, &query_resolve, &query_readback)
        {
            encoder.resolve_query_set(queries, 0..if tone { 6 } else { 2 }, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, read, 0, 48);
        }
        let submission = self.queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = sender.send(r);
        });
        let (q_sender, q_receiver) = mpsc::sync_channel(1);
        if let Some(read) = &query_readback {
            read.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = q_sender.send(r);
            });
        }
        self.device
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
        profile.gpu_wait_readback_seconds = wait_start.elapsed().as_secs_f64();
        if let Some(read) = &query_readback {
            q_receiver
                .recv_timeout(Duration::from_secs(2))
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            let bytes = read.get_mapped_range(..).map_err(|e| e.to_string())?;
            let ticks: &[u64] = bytemuck::cast_slice(&bytes);
            let duration = |i: usize| {
                ticks[i + 1].saturating_sub(ticks[i]) as f64
                    * f64::from(self.queue.get_timestamp_period())
                    / 1e9
            };
            profile.gpu_render_seconds = Some(duration(0));
            if tone {
                profile.gpu_horizontal_seconds = Some(duration(2));
                profile.gpu_paper_seconds = Some(duration(4));
            }
            drop(bytes);
            read.unmap();
        }
        let unpack_start = Instant::now();
        let mapped = readback.get_mapped_range(..).map_err(|e| e.to_string())?;
        // Vulkan readback allocations on Android may be uncached. Copy once
        // with the platform memcpy before strided RGB/gray channel accesses.
        #[cfg(target_os = "android")]
        let cached = mapped.to_vec();
        #[cfg(target_os = "android")]
        let bytes = cached.as_slice();
        #[cfg(not(target_os = "android"))]
        let bytes = &mapped[..];
        let color_image = if color {
            let mut rgb = vec![0; pixels as usize * 3];
            rgb.par_chunks_mut(3 * 4096)
                .enumerate()
                .for_each(|(block, out)| {
                    for (j, pixel) in out.chunks_exact_mut(3).enumerate() {
                        let i = (block * 4096 + j) * 4;
                        pixel.copy_from_slice(&bytes[i..i + 3]);
                    }
                });
            Some(RgbImage::from_raw(width, height, rgb).unwrap())
        } else {
            None
        };
        let tone_images = if tone {
            let bytes = &bytes[if color { pixels as usize * 4 } else { 0 }..];
            let (mut gray, mut binary, mut clean) = (
                vec![0; pixels as usize],
                vec![0; pixels as usize],
                vec![0; pixels as usize],
            );
            gray.par_chunks_mut(4096)
                .zip(binary.par_chunks_mut(4096))
                .zip(clean.par_chunks_mut(4096))
                .enumerate()
                .for_each(|(block, ((g, b), c))| {
                    for j in 0..g.len() {
                        let i = (block * 4096 + j) * 4;
                        g[j] = bytes[i];
                        b[j] = bytes[i + 1];
                        c[j] = bytes[i + 2];
                    }
                });
            Some((
                GrayImage::from_raw(width, height, gray).unwrap(),
                GrayImage::from_raw(width, height, binary).unwrap(),
                GrayImage::from_raw(width, height, clean).unwrap(),
            ))
        } else {
            None
        };
        drop(mapped);
        readback.unmap();
        profile.unpack_seconds = unpack_start.elapsed().as_secs_f64();
        profile.adapter = Some(self.adapter.clone());
        Ok(Output {
            color: color_image,
            tones: tone_images,
            profile,
        })
    }
}

fn distance_packed(
    seeds: &[u32],
    width: usize,
    height: usize,
    cap: u16,
) -> Result<Vec<u32>, String> {
    let context = CONTEXT
        .get_or_init(|| Context::new().map(Mutex::new))
        .as_ref()
        .map_err(Clone::clone)?;
    let context = context
        .lock()
        .map_err(|_| "GPU context lock poisoned".to_string())?;
    context.distance.run(
        &context.device,
        &context.queue,
        seeds,
        (width as u32, height as u32),
        cap,
    )
}

pub(crate) fn distance_pair(
    a: &[u8],
    b: &[u8],
    width: usize,
    height: usize,
    cap: u8,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let seeds: Vec<_> = a
        .par_iter()
        .zip(b)
        .map(|(a, b)| u32::from(*a == 0) | (u32::from(*b == 0) << 1))
        .collect();
    let packed = distance_packed(&seeds, width, height, u16::from(cap))?;
    Ok((
        packed.par_iter().map(|v| (v & 65535) as u8).collect(),
        packed.par_iter().map(|v| (v >> 16) as u8).collect(),
    ))
}

pub(crate) fn distance_single(
    seeds: &[u16],
    width: usize,
    height: usize,
    cap: u16,
) -> Result<Vec<u16>, String> {
    let mask: Vec<_> = seeds.par_iter().map(|v| u32::from(*v == 0)).collect();
    Ok(distance_packed(&mask, width, height, cap)?
        .par_iter()
        .map(|v| (v & 65535) as u16)
        .collect())
}

pub(crate) fn render(
    geometry: &Geometry,
    image: &RgbImage,
    width: u32,
    height: u32,
    tone: bool,
    color: bool,
) -> Result<Output, String> {
    let start = Instant::now();
    let initialized = CONTEXT.get().is_some();
    let context = CONTEXT
        .get_or_init(|| Context::new().map(Mutex::new))
        .as_ref()
        .map_err(Clone::clone)?;
    let mut profile = RenderProfile {
        backend: "gpu".into(),
        initialization_seconds: if initialized {
            0.
        } else {
            start.elapsed().as_secs_f64()
        },
        ..RenderProfile::default()
    };
    let limit = context
        .lock()
        .map_err(|_| "GPU context lock poisoned".to_string())?
        .device
        .limits()
        .max_storage_buffer_binding_size;
    if u64::from(width) * u64::from(height) * if tone { 8 } else { 4 } > limit
        || u64::from(image.width()) * u64::from(image.height()) * 4 > limit
    {
        return Err("Image exceeds GPU buffer binding limit".into());
    }
    let start = Instant::now();
    let map = map::Map::build(geometry, width, height)?;
    profile.map_seconds = start.elapsed().as_secs_f64();
    profile.map_evaluations = map.evaluations;
    profile.map_nodes = map.nodes.len();
    profile.exact_map_pixels = map.dense.len();
    profile.max_sampled_map_error_pixels = map.max_probe_error;
    context
        .lock()
        .map_err(|_| "GPU context lock poisoned".to_string())?
        .run(image, &map, (width, height), tone, color, profile)
}
