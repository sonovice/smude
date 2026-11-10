//! Shared offscreen GPU implementation: Metal on Apple, Vulkan on Android/Linux,
//! and Direct3D on Windows. It consumes our own fitted geometry and formulas.
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
/// Neighbours whose window sums one shader invocation derives from each other.
const RUN: u32 = 32;
/// Parts of a page the GPU reconstructs while the map of the next is built.
const BANDS: u32 = 4;

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
    pub tones: Option<(GrayImage, GrayImage)>,
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
            label: Some("Smude bicubic and diagnostic binarization"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 8 {
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
        })
    }

    /// A storage buffer of `size` bytes. `fill` writes the bytes that follow
    /// a given number of them; it runs in parallel on blocks of the staging
    /// memory of the queue, which the contents thus reach without a copy.
    fn upload(
        &self,
        label: &str,
        size: usize,
        fill: impl Fn(usize, wgpu::WriteOnly<[u8]>) + Sync,
    ) -> Result<wgpu::Buffer, String> {
        const BLOCK: usize = 1 << 18;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            // An empty buffer cannot be bound.
            size: size.max(8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(size) = wgpu::BufferSize::new(size as u64) {
            let mut view = self
                .queue
                .write_buffer_with(&buffer, 0, size)
                .ok_or_else(|| format!("GPU upload of {label} failed"))?;
            // Only a view of known size can move to another thread.
            let (blocks, rest) = view.slice(..).into_chunks::<BLOCK>();
            fill(blocks.len() * BLOCK, rest);
            let blocks: Vec<_> = blocks.into_iter().collect();
            blocks
                .into_par_iter()
                .enumerate()
                .for_each(|(block, out)| fill(block * BLOCK, out.into()));
        }
        Ok(buffer)
    }

    /// Blocks until a submission has finished and a buffer it fills is mapped.
    fn wait(
        &self,
        submission: wgpu::SubmissionIndex,
        mapped: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(20)),
            })
            .map_err(|e| e.to_string())?;
        mapped
            .recv_timeout(Duration::from_secs(2))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())
    }

    fn run(
        &self,
        geometry: &Geometry,
        image: &RgbImage,
        dimensions: (u32, u32),
        tone: bool,
        color: bool,
        mut profile: RenderProfile,
    ) -> Result<Output, String> {
        let (width, height) = dimensions;
        let pixels = u64::from(width) * u64::from(height);
        let limit = self.device.limits().max_storage_buffer_binding_size;
        if pixels * 4 > limit || u64::from(image.width()) * u64::from(image.height()) * 4 > limit {
            return Err("Image exceeds GPU buffer binding limit".into());
        }
        let upload_start = Instant::now();
        let memory_scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let source = self.upload("source", image.as_raw().len() / 3 * 4, |offset, out| {
            let rgb = &image.as_raw()[offset / 4 * 3..][..out.len() / 4 * 3];
            let (mut out, _) = out.into_chunks::<4>();
            let mut rgba = [[0; 4]; 1024];
            for rgb in rgb.chunks(3 * rgba.len()) {
                let rgba = &mut rgba[..rgb.len() / 3];
                for (rgba, rgb) in rgba.iter_mut().zip(rgb.chunks_exact(3)) {
                    *rgba = [rgb[0], rgb[1], rgb[2], 0xff];
                }
                let (mut head, tail) = out.split_at(rgba.len());
                head.copy_from_slice(rgba);
                out = tail;
            }
        })?;
        let empty = |label: &str, size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let rendered = empty("rendered RGB and luminance", pixels * 4);
        let plane = if tone { pixels * 4 } else { 4 };
        let sums = empty("horizontal luminance sums", plane);
        let squares = empty("horizontal squared luminance sums", plane);
        let tone_bytes = u64::from(width.div_ceil(2)) * u64::from(height) * 4;
        let tones = empty(
            "gray and diagnostic binary",
            if tone { tone_bytes } else { 4 },
        );
        // wgpu zeroes a storage buffer before the first pass that may read it.
        // Asked for here, that and the source upload run while the map is
        // being built.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Smude page buffers"),
            });
        for buffer in [&rendered, &sums, &squares, &tones] {
            encoder.clear_buffer(buffer, 0, None);
        }
        self.queue.submit([encoder.finish()]);
        profile.upload_seconds = upload_start.elapsed().as_secs_f64();
        let map_start = Instant::now();
        let map = map::Map::new(geometry, width, height);
        profile.map_seconds = map_start.elapsed().as_secs_f64();
        profile.map_evaluations = map.evaluations;
        let upload_start = Instant::now();
        let roots = empty("inverse roots", u64::from(map.columns * map.rows) * 8);
        let color_bytes = if color { pixels * 4 } else { 0 };
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("processed image readback"),
            size: color_bytes + if tone { tone_bytes } else { 0 },
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let band_rows = map.rows.div_ceil(BANDS);
        let bands = map.rows.div_ceil(band_rows);
        // The timed passes: those of every band, then the one over the page.
        let band_passes = 1 + u32::from(tone);
        let passes = bands * band_passes + u32::from(tone);
        let queries = self.timestamps.then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("image stage timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * passes,
            })
        });
        let query_bytes = 16 * u64::from(passes);
        let query_resolve = queries.as_ref().map(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stage query resolution"),
                size: query_bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let query_readback = queries.as_ref().map(|_| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stage query readback"),
                size: query_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        profile.upload_seconds += upload_start.elapsed().as_secs_f64();
        // A workgroup is eight by eight invocations of `pixels` each. `index`
        // counts the timed passes.
        let pass = |encoder: &mut wgpu::CommandEncoder,
                    group: &wgpu::BindGroup,
                    (pipeline, label, pixels): (&wgpu::ComputePipeline, &str, [u32; 2]),
                    index: u32,
                    rows: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: queries.as_ref().map(|q| wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(index * 2),
                    end_of_pass_write_index: Some(index * 2 + 1),
                }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            let [x, y] = pixels.map(|pixels| 8 * pixels);
            pass.dispatch_workgroups(width.div_ceil(x), rows.div_ceil(y), 1);
        };
        let strips = |label: &str, strips: Vec<&[u8]>| {
            let size = strips.iter().map(|strip| strip.len()).sum();
            self.upload(label, size, |mut offset, mut out| {
                for mut strip in strips.iter().copied() {
                    let skipped = offset.min(strip.len());
                    offset -= skipped;
                    strip = &strip[skipped..];
                    let length = strip.len().min(out.len());
                    let (head, tail) = out.split_at(length);
                    head.into_slice(..).copy_from_slice(&strip[..length]);
                    out = tail;
                }
            })
        };
        let mut group = None;
        for (index, first) in (0..).zip((0..map.rows).step_by(band_rows as usize)) {
            let map_start = Instant::now();
            let band = map.band(first..(first + band_rows).min(map.rows))?;
            profile.map_seconds += map_start.elapsed().as_secs_f64();
            profile.map_evaluations += band.evaluations;
            profile.map_nodes += band.nodes();
            profile.exact_map_pixels += band.dense();
            profile.max_sampled_map_error_pixels = profile
                .max_sampled_map_error_pixels
                .max(band.max_probe_error);
            if (band.nodes() * 48).max(band.dense() * 8) as u64 > limit {
                return Err(
                    "GPU buffer binding limit exceeded; use CPU for this image size".into(),
                );
            }
            let upload_start = Instant::now();
            let nodes = band.strips.iter().map(|s| bytemuck::cast_slice(&s.nodes));
            let nodes = strips("inverse nodes", nodes.collect())?;
            let dense = band.strips.iter().map(|s| bytemuck::cast_slice(&s.dense));
            let dense = strips("exact inverse coordinates", dense.collect())?;
            self.queue.write_buffer(
                &roots,
                u64::from(first * map.columns) * 8,
                bytemuck::cast_slice(&band.roots().collect::<Vec<_>>()),
            );
            let first = first * map::TILE;
            let values = [
                image.width(),
                image.height(),
                width,
                height,
                map.columns,
                map::TILE,
                RUN,
                first,
            ];
            let params = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("image dimensions"),
                    contents: bytemuck::cast_slice(&values),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let buffers = [
                &source, &nodes, &roots, &dense, &rendered, &sums, &squares, &tones, &params,
            ];
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            let group = group.insert(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("band buffers"),
                layout: &self.layout,
                entries: &entries,
            }));
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Smude band"),
                });
            let rows = (first + band_rows * map::TILE).min(height) - first;
            let index = index * band_passes;
            let stage = (&self.render, "bicubic reconstruction", [1, 1]);
            pass(&mut encoder, group, stage, index, rows);
            if tone {
                let stage = (&self.horizontal, "horizontal statistics", [RUN, 1]);
                pass(&mut encoder, group, stage, index + 1, rows);
            }
            self.queue.submit([encoder.finish()]);
            profile.upload_seconds += upload_start.elapsed().as_secs_f64();
        }
        let wait_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Smude page"),
            });
        if let (true, Some(group)) = (tone, &group) {
            let stage = (&self.paper, "diagnostic binarization", [2, RUN]);
            pass(&mut encoder, group, stage, passes - 1, height);
        }
        if color {
            encoder.copy_buffer_to_buffer(&rendered, 0, &readback, 0, color_bytes);
        }
        if tone {
            encoder.copy_buffer_to_buffer(&tones, 0, &readback, color_bytes, tone_bytes);
        }
        let submission = self.queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = sender.send(r);
        });
        self.wait(submission, receiver)?;
        // Metal only reports the timestamps of passes that have finished by
        // the time they are resolved. Their readback overlaps the unpacking.
        let timestamps = if let (Some(queries), Some(resolve), Some(read)) =
            (&queries, &query_resolve, &query_readback)
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Smude stage timestamps"),
                });
            encoder.resolve_query_set(queries, 0..2 * passes, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, read, 0, query_bytes);
            let submission = self.queue.submit([encoder.finish()]);
            let (sender, receiver) = mpsc::sync_channel(1);
            read.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = sender.send(r);
            });
            Some((read, submission, receiver))
        } else {
            None
        };
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(error.to_string());
        }
        if let Some(error) = pollster::block_on(memory_scope.pop()) {
            return Err(error.to_string());
        }
        profile.gpu_wait_readback_seconds = wait_start.elapsed().as_secs_f64();
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
            let (mut gray, mut binary) = (vec![0; pixels as usize], vec![0; pixels as usize]);
            gray.par_chunks_mut(width as usize)
                .zip(binary.par_chunks_mut(width as usize))
                .zip(bytes[color_bytes as usize..].par_chunks(width.div_ceil(2) as usize * 4))
                .for_each(|((gray, binary), row)| {
                    for ((gray, binary), tones) in
                        gray.iter_mut().zip(binary).zip(row.chunks_exact(2))
                    {
                        (*gray, *binary) = (tones[0], tones[1]);
                    }
                });
            Some((
                GrayImage::from_raw(width, height, gray).unwrap(),
                GrayImage::from_raw(width, height, binary).unwrap(),
            ))
        } else {
            None
        };
        drop(mapped);
        readback.unmap();
        profile.unpack_seconds = unpack_start.elapsed().as_secs_f64();
        if let Some((read, submission, receiver)) = timestamps {
            let wait_start = Instant::now();
            self.wait(submission, receiver)?;
            profile.gpu_wait_readback_seconds += wait_start.elapsed().as_secs_f64();
            let bytes = read.get_mapped_range(..).map_err(|e| e.to_string())?;
            let ticks: &[u64] = bytemuck::cast_slice(&bytes);
            let duration = |pass: u32| {
                let ticks = &ticks[2 * pass as usize..];
                ticks[1].saturating_sub(ticks[0]) as f64
                    * f64::from(self.queue.get_timestamp_period())
                    / 1e9
            };
            let stage = |pass: u32| -> f64 {
                (0..bands)
                    .map(|band| duration(band * band_passes + pass))
                    .sum()
            };
            profile.gpu_render_seconds = Some(stage(0));
            if tone {
                profile.gpu_horizontal_seconds = Some(stage(1));
                profile.gpu_paper_seconds = Some(duration(passes - 1));
            }
            drop(bytes);
            read.unmap();
        }
        profile.adapter = Some(self.adapter.clone());
        Ok(Output {
            color: color_image,
            tones: tone_images,
            profile,
        })
    }
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
    let profile = RenderProfile {
        backend: "gpu".into(),
        initialization_seconds: if initialized {
            0.
        } else {
            start.elapsed().as_secs_f64()
        },
        ..RenderProfile::default()
    };
    context
        .lock()
        .map_err(|_| "GPU context lock poisoned".to_string())?
        .run(geometry, image, (width, height), tone, color, profile)
}
