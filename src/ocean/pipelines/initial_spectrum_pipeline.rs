use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferInitDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor,
    ComputePipeline, Extent3d, Origin3d, PipelineLayoutDescriptor, RawComputePipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};

use crate::ocean::ocean_parameters::OceanSpectrumParameters;
use crate::ocean::utils::clamp;

const WG_COUNT: u32 = 16;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
    size: u32,
    length_scale: f32,
    cut_off_low: f32,
    cut_off_high: f32,
    gravity_acceleration: f32,
    depth: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpectrumParamers {
    scale: f32,
    angle: f32,
    spread_blend: f32,
    swell: f32,
    alpha: f32,
    peak_omega: f32,
    gamma: f32,
    short_waves_fade: f32,
}

impl SpectrumParamers {
    fn from_ocean_parameters(o: OceanSpectrumParameters) -> Self {
        Self {
            scale: o.scale,
            angle: o.wind_direction / 180.0 * std::f32::consts::PI,
            spread_blend: o.spread_blend,
            swell: clamp(o.swell, 0.01, 1.0),
            alpha: Self::jonswap_alpha(9.81, o.fetch, o.wind_speed),
            peak_omega: Self::jonswap_peak_frequency(9.81, o.fetch, o.wind_speed),
            gamma: o.peak_enhancement,
            short_waves_fade: o.short_waves_fade,
        }
    }

    fn jonswap_alpha(g: f32, fetch: f32, wind_speed: f32) -> f32 {
        0.076 * f32::powf(g * fetch / wind_speed / wind_speed, -0.22)
    }

    fn jonswap_peak_frequency(g: f32, fetch: f32, wind_speed: f32) -> f32 {
        22.0 * f32::powf(wind_speed * fetch / g / g, -0.33)
    }
}

pub struct InitialSpectrumPipeline {
    size: u32,
    textures_bind_group: BindGroup,
    calculate_initial_spectrum_pipeline: ComputePipeline,
    calculate_conjugated_spectrum_pipeline: ComputePipeline,

    texture_size: Extent3d,
    noise_texture: Texture,
    parameters_buffer: Buffer,
    parameters_bind_group: BindGroup,

    noise_data: Vec<f32>,
}

impl InitialSpectrumPipeline {
    pub fn init(
        wave_params: OceanSpectrumParameters,
        device: &RenderDevice,
        h0k_texture: &Texture,
        waves_data_texture: &Texture,
        h0_texture: &Texture,
    ) -> Self {
        let texture_size = Extent3d {
            width: wave_params.size,
            height: wave_params.size,
            depth_or_array_layers: 1,
        };

        let parameters = Parameters {
            size: wave_params.size,
            length_scale: wave_params.length_scale,
            cut_off_low: wave_params.cut_off_low,
            cut_off_high: wave_params.cut_off_high,
            gravity_acceleration: wave_params.gravity_acceleration,
            depth: wave_params.depth,
        };

        let spectrum_parameters = SpectrumParamers::from_ocean_parameters(wave_params);

        let parameters_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Parameters Buffer"),
            contents: bytemuck::cast_slice(&[parameters]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let spectrum_parameters_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Spectrum parameters buffer"),
            contents: bytemuck::cast_slice(&[spectrum_parameters]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let parameters_bind_group_layout = device.create_bind_group_layout(
            "IS - Parameters bind group layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let parameters_bind_group = device.create_bind_group(
            "Parameters bind group",
            &parameters_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: parameters_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: spectrum_parameters_buffer.as_entire_binding(),
                },
            ],
        );

        let noise_texture = device.create_texture(&TextureDescriptor {
            label: Some("Noise texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let texture_bind_group_layout = device.create_bind_group_layout(
            "IS - Texture bind group layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: false },
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadWrite,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadWrite,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadWrite,
                    },
                    count: None,
                },
            ],
        );

        let noise_data = generate_noise_data(wave_params.size as usize);

        let shader = unsafe {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Initial spectrum shader"),
                source: ShaderSource::Wgsl(
                    include_str!("../../shaders/initial_spectrum.wgsl").into(),
                ),
            })
        };

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Initial spectrum pipeline layout"),
            bind_group_layouts: &[&texture_bind_group_layout, &parameters_bind_group_layout],
            push_constant_ranges: &[],
        });

        let calculate_initial_spectrum_pipeline =
            device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("Initial spectrum pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("calculate_initial_spectrum"),
                compilation_options:
                    bevy::render::render_resource::PipelineCompilationOptions::default(),
                cache: None,
            });

        let calculate_conjugated_spectrum_pipeline =
            device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("Calculate conjugated spectrum pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("calculate_conjugated_spectrum"),
                compilation_options:
                    bevy::render::render_resource::PipelineCompilationOptions::default(),
                cache: None,
            });

        let textures_bind_group = device.create_bind_group(
            "IS - Texture bind group",
            &texture_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&noise_texture.create_view(
                        &TextureViewDescriptor {
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&h0_texture.create_view(
                        &TextureViewDescriptor {
                            format: Some(TextureFormat::Rgba32Float),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&waves_data_texture.create_view(
                        &TextureViewDescriptor {
                            format: Some(TextureFormat::Rgba32Float),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&h0k_texture.create_view(
                        &TextureViewDescriptor {
                            format: Some(TextureFormat::Rgba32Float),
                            ..Default::default()
                        },
                    )),
                },
            ],
        );

        Self {
            size: wave_params.size,
            noise_data,
            texture_size,
            noise_texture,
            textures_bind_group,
            calculate_initial_spectrum_pipeline,
            calculate_conjugated_spectrum_pipeline,
            parameters_buffer,
            parameters_bind_group,
        }
    }

    pub fn dispatch(&self, encoder: &mut CommandEncoder, queue: &RenderQueue) {
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &self.noise_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &bytemuck::cast_slice(&self.noise_data),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16 * self.size),
                rows_per_image: Some(self.size),
            },
            self.texture_size,
        );

        {
            let (dispatch_width, dispatch_height) = compute_work_group_count(
                (self.texture_size.width, self.texture_size.height),
                (WG_COUNT, WG_COUNT),
            );
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Calculate Initial Spectrum"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.calculate_initial_spectrum_pipeline);
            compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.parameters_bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        {
            let (dispatch_width, dispatch_height) = compute_work_group_count(
                (self.texture_size.width, self.texture_size.height),
                (WG_COUNT, WG_COUNT),
            );

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Calculate Conjugated Spectrum"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.calculate_conjugated_spectrum_pipeline);
            compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.parameters_bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }
    }
}

fn generate_noise_data(size: usize) -> Vec<f32> {
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = rand::rng();
    let normal: StandardNormal = StandardNormal;
    let mut buf: Vec<f32> = vec![0.0; 4 * size * size];
    for i in 0..4 * size * size {
        buf[i] = normal.sample(&mut rng);
    }

    buf
}

fn compute_work_group_count(
    (width, height): (u32, u32),
    (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;

    return (x, y);
}
