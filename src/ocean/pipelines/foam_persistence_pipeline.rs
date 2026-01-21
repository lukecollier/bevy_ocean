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
struct Parameters {
    decay_rate: f32,
    foam_spawn_threshold: f32,
    foam_spawn_strength: f32,
    delta_time: f32,
}

pub struct FoamPersistencePipeline {
    size: u32,
    textures_bind_group: BindGroup,
    pipeline: ComputePipeline,
}

impl FoamPersistencePipeline {
    pub fn init(
        device: &RenderDevice,
        size: u32,
        displacement_texture: &Texture,
        foam_persistence_texture: &Texture,
    ) -> Self {
        let textures_bind_group_layout = device.create_bind_group_layout(
            "Foam persistence - texture bind group layout",
            &[
                // displacement_texture (read only - for Jacobian)
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
                // foam_persistence (read-write)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::R32Float,
                        access: StorageTextureAccess::ReadWrite,
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
                range: 0..16, // 4 floats * 4 bytes
            }],
        });

        let shader = unsafe {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Foam persistence shader"),
                source: ShaderSource::Wgsl(
                    include_str!("../../../assets/shaders/foam_persistence.wgsl").into(),
                ),
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

        let textures_bind_group = device.create_bind_group(
            "Foam persistence - textures",
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
                    resource: BindingResource::TextureView(&foam_persistence_texture.create_view(
                        &TextureViewDescriptor {
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

    pub fn dispatch(&self, encoder: &mut CommandEncoder, dt: std::time::Duration) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Foam persistence"),
            timestamp_writes: None,
        });

        let parameters = Parameters {
            decay_rate: 0.92,          // Foam decays slowly
            foam_spawn_threshold: 1.2, // Match the ocean shader threshold
            foam_spawn_strength: 1.0,  // Match the ocean shader multiplier
            delta_time: dt.as_secs_f32(),
        };

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));
        compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, 1);
    }
}
