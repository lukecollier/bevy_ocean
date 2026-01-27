use bevy::render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType,
        CommandEncoder, ComputePassDescriptor, ComputePipeline, PipelineCompilationOptions,
        PipelineLayoutDescriptor, PushConstantRange, RawComputePipelineDescriptor,
        ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, Texture,
        TextureFormat, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::RenderDevice,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    time: f32,
}

pub struct TimeDependentSpectrumPipeline {
    size: u32,
    layers: u32,
    textures_bind_group: BindGroup,
    pipeline: ComputePipeline,
}

impl TimeDependentSpectrumPipeline {
    pub fn init<'a>(
        size: u32,
        layers: u32,
        device: &RenderDevice,

        h0_texture: &'a Texture,
        waves_data_texture: &'a Texture,
        amp_dx_dz_texture: &'a Texture,
        amp_dyx_dyz_texture: &'a Texture,
    ) -> Self {
        let texture_bind_group_layout = device.create_bind_group_layout(
            "Time-dependent spectrum - texture bind group layout",
            &[
                // h0_texture (array)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                // waves_data_texture (array)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                // amp_dx_dz_texture (array)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                // amp_dyx_dyz_texture (array)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
            ],
        );

        let textures_bind_group = device.create_bind_group(
            "Time-dependent spectrum - texture bind group",
            &texture_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&h0_texture.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&waves_data_texture.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&amp_dx_dz_texture.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&amp_dyx_dyz_texture.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
            ],
        );

        let shader = unsafe {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Time-dependent spectrum shader"),
                source: ShaderSource::Wgsl(
                    include_str!("shaders/time_dependent_spectrum.wgsl").into(),
                ),
            })
        };

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Time-dependent spectrum pipeline layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });

        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("Time-dependent spectrum pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("calculate_amplitudes"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            size,
            layers,
            textures_bind_group,
            pipeline,
        }
    }

    pub fn dispatch(&self, encoder: &mut CommandEncoder, time: f32) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Calculate time-dependent spectrum"),
            timestamp_writes: None,
        });

        let params = Params { time };

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[params]));
        // Dispatch for all layers at once
        compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, self.layers);
    }
}
