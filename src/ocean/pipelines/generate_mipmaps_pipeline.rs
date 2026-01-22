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

pub struct GenerateMipmapsPipeline {
    size: u32,
    textures_bind_group: BindGroup,
    pipeline: ComputePipeline,
}

impl GenerateMipmapsPipeline {
    pub fn init<'a>(
        device: &RenderDevice,
        size: u32,
        displacement_texture: &'a Texture,
        derivatives_texture: &'a Texture,
    ) -> Self {
        let textures_bind_group_layout = device.create_bind_group_layout(
            "generate mipmaps - texture bind group layout",
            &[
                // displacement
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                // displacement_0
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                // displacement_1
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                // displacement_2
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                // derivatives
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                // derivatives_1
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                // derivatives_2
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                // derivatives_3
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
            ],
        );

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Generate mipmaps pipeline layout"),
            bind_group_layouts: &[&textures_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..8,
            }],
        });

        let shader = unsafe {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Generate mipmaps shader"),
                source: ShaderSource::Wgsl(
                    include_str!("../../shaders/generate_mipmaps.wgsl").into(),
                ),
            })
        };

        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("Generate mipmaps pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let textures_bind_group = device.create_bind_group(
            "Generate mipmaps textures",
            &textures_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&displacement_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&displacement_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 1,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&displacement_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 2,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&displacement_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 3,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&derivatives_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&derivatives_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 1,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&derivatives_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 2,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(&derivatives_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 3,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
            ],
        );

        Self {
            size,
            textures_bind_group,
            pipeline,
        }
    }

    pub fn dispatch(&self, encoder: &mut CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Generate mipmaps"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
        compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, 1);
    }
}
