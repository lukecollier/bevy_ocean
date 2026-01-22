use bevy::render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
        BufferBindingType, BufferInitDescriptor, BufferUsages, CommandEncoder,
        ComputePassDescriptor, ComputePipeline, Extent3d, PipelineCompilationOptions,
        PipelineLayoutDescriptor, PushConstantRange, RawComputePipelineDescriptor,
        ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, Texture,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
        TextureViewDimension,
    },
    renderer::RenderDevice,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
    ping_pong: u32,
    step: u32,
    size: u32,
}

pub struct FFT {
    size: u32,
    buffer: Texture,

    precompute_pipeline: ComputePipeline,
    horizontal_step_pipeline: ComputePipeline,
    vertical_step_pipeline: ComputePipeline,
    scale_pipeline: ComputePipeline,
    permute_pipeline: ComputePipeline,
    swap_pipeline: ComputePipeline,

    parameters_buffer: Buffer,
    precompute_data_texture: Texture,
    texture_bind_group: BindGroup,
    parameters_bind_group: BindGroup,
}

impl FFT {
    pub fn init(size: u32, device: &RenderDevice, input: &Texture, input_b: &Texture) -> Self {
        let shader = unsafe {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("FFT shader"),
                source: ShaderSource::Wgsl(include_str!("../../shaders/fft.wgsl").into()),
            })
        };

        let texture_bind_group_layout = device.create_bind_group_layout(
            "FFT texture bind group layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadWrite,
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
                BindGroupLayoutEntry {
                    binding: 4,
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

        let parameters_bind_group_layout = device.create_bind_group_layout(
            "FFT Parameters bind group layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        );

        let parameters = Parameters {
            ping_pong: 0,
            step: 0,
            size,
        };

        let parameters_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("FFT parameters buffer"),
            contents: bytemuck::cast_slice(&[parameters]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let parameters_bind_group = device.create_bind_group(
            "FFT parameters",
            &parameters_bind_group_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: parameters_buffer.as_entire_binding(),
            }],
        );

        let precompute_data_texture = device.create_texture(&TextureDescriptor {
            label: Some("FFT precompute buffer"),
            size: Extent3d {
                width: (size as f64).log(2.0) as u32,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let buffer = device.create_texture(&TextureDescriptor {
            label: Some("FFT Buffer"),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let buffer_b = device.create_texture(&TextureDescriptor {
            label: Some("FFT Buffer B"),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let texture_bind_group = device.create_bind_group(
            "FFT texture bind group",
            &texture_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&precompute_data_texture.create_view(
                        &TextureViewDescriptor {
                            format: Some(TextureFormat::Rgba32Float),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&input.create_view(
                        &TextureViewDescriptor {
                            format: Some(TextureFormat::Rgba32Float),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&buffer.create_view(
                        &TextureViewDescriptor {
                            format: Some(TextureFormat::Rgba32Float),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&input_b.create_view(
                        &TextureViewDescriptor {
                            format: Some(TextureFormat::Rgba32Float),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&buffer_b.create_view(
                        &TextureViewDescriptor {
                            format: Some(TextureFormat::Rgba32Float),
                            ..Default::default()
                        },
                    )),
                },
            ],
        );

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("FFT pipeline layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..12,
            }],
        });

        let precompute_pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("FFT - Calculate twiddle factors and input indices"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("calculat_twiddle_factors_and_input_indices"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let horizontal_step_pipeline =
            device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("FFT - Horizontal step"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("horizontal_step_inverse_fft"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let vertical_step_pipeline =
            device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("FFT - Vertical step"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("vertical_step_inverse_fft"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let scale_pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("FFT - Scale"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scale"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let permute_pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("FFT - Permute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("permute"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let swap_pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("FFT - Swap"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("swap"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        return Self {
            size,
            buffer,
            precompute_pipeline,
            parameters_buffer,
            precompute_data_texture,
            texture_bind_group,
            parameters_bind_group,
            horizontal_step_pipeline,
            vertical_step_pipeline,
            scale_pipeline,
            permute_pipeline,
            swap_pipeline,
        };
    }

    pub fn precompute(&self, encoder: &mut CommandEncoder) {
        let log_size = (self.size as f64).log(2.0) as u32;

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("FFT precompute"),
            timestamp_writes: None,
        });

        let parameters = Parameters {
            ping_pong: 0,
            size: self.size,
            step: 0,
        };

        compute_pass.set_pipeline(&self.precompute_pipeline);
        compute_pass.set_bind_group(0, &self.texture_bind_group, &[]);

        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));

        compute_pass.dispatch_workgroups(log_size, self.size / 2 / 8, 1);
    }

    pub fn dispatch(&self, encoder: &mut CommandEncoder) {
        let log_size = (self.size as f64).log(2.0) as u32;
        let mut ping_pong = 0u32;

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("FFT"),
            timestamp_writes: None,
        });

        compute_pass.set_bind_group(0, &self.texture_bind_group, &[]);

        compute_pass.set_pipeline(&self.horizontal_step_pipeline);
        for i in 0..log_size {
            ping_pong = if ping_pong == 0 { 1 } else { 0 };
            let parameters = Parameters {
                ping_pong,
                size: self.size,
                step: i,
            };

            compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));
            compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, 1);
        }

        compute_pass.set_pipeline(&self.vertical_step_pipeline);
        for i in 0..log_size {
            ping_pong = if ping_pong == 0 { 1 } else { 0 };
            let parameters = Parameters {
                ping_pong,
                size: self.size,
                step: i,
            };

            compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));
            compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, 1);
        }

        if ping_pong == 1 {
            compute_pass.set_pipeline(&self.swap_pipeline);
            compute_pass.set_push_constants(
                0,
                bytemuck::cast_slice(&[Parameters {
                    ping_pong: 0,
                    size: self.size,
                    step: 0,
                }]),
            );
            compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, 1);
        }

        compute_pass.set_pipeline(&self.permute_pipeline);
        compute_pass.set_push_constants(
            0,
            bytemuck::cast_slice(&[Parameters {
                ping_pong: 0,
                size: self.size,
                step: 0,
            }]),
        );
        compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, 1);
    }
}
