use bevy::render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType,
        CommandEncoder, ComputePassDescriptor, ComputePipeline, PipelineCompilationOptions,
        PipelineLayoutDescriptor, PushConstantRange, RawComputePipelineDescriptor,
        ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, Texture,
        TextureFormat, TextureView, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::RenderDevice,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
    decay_rate: f32,
    foam_spawn_threshold: f32,
    foam_spawn_strength: f32,
    delta_time: f32,
}

pub struct FoamPersistencePipeline {
    size: u32,
    layers: u32,
    textures_bind_group: BindGroup,
    pipeline: ComputePipeline,
    // Store mip level view for foam persistence (level 1 only)
    #[allow(dead_code)]
    foam_mip1_view: TextureView,
    #[allow(dead_code)]
    foam_mip2_view: TextureView,
}

impl FoamPersistencePipeline {
    pub fn init(
        device: &RenderDevice,
        size: u32,
        displacement_texture: &Texture,
        foam_persistence_texture: &Texture,
    ) -> Self {
        let layers = displacement_texture.depth_or_array_layers();

        let textures_bind_group_layout = device.create_bind_group_layout(
            "Foam persistence - texture bind group layout",
            &[
                // displacement_texture (read only - for Jacobian)
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
                // foam_persistence mip 0 (read-write)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        access: StorageTextureAccess::ReadWrite,
                    },
                    count: None,
                },
                // foam_persistence mip 1 (write)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                // foam_persistence mip 2 (write)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::R32Float,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
            ],
        );

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Foam persistence pipeline layout"),
            bind_group_layouts: &[&textures_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<Parameters>() as u32,
            }],
        });

        let shader = unsafe {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Foam persistence shader"),
                source: ShaderSource::Wgsl(include_str!("shaders/foam_persistence.wgsl").into()),
            })
        };

        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("Foam persistence pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_foam"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create mip level 1 view for foam persistence
        let foam_mip1_view = foam_persistence_texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            base_mip_level: 1,
            mip_level_count: Some(1),
            ..Default::default()
        });

        let foam_mip2_view = foam_persistence_texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            base_mip_level: 2,
            mip_level_count: Some(1),
            ..Default::default()
        });

        let textures_bind_group = device.create_bind_group(
            "Foam persistence - textures",
            &textures_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&displacement_texture.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&foam_persistence_texture.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                // Foam mip level 1
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&foam_mip1_view),
                },
                // Foam mip level 2
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&foam_mip2_view),
                },
            ],
        );

        Self {
            size,
            layers,
            textures_bind_group,
            pipeline,
            foam_mip1_view,
            foam_mip2_view,
        }
    }

    pub fn dispatch(&self, encoder: &mut CommandEncoder, dt: std::time::Duration) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Foam persistence"),
            timestamp_writes: None,
        });

        let parameters = Parameters {
            decay_rate: 0.95,
            foam_spawn_threshold: 1.0,
            foam_spawn_strength: 1.5,
            delta_time: dt.as_secs_f32(),
        };

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));
        // Dispatch for all layers at once
        compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, self.layers);
    }
}
