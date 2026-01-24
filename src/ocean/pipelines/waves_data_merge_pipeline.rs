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
    lambda: f32,
    delta_time: f32,
    layer: u32,
}

pub struct WavesDataMergePipeline {
    size: u32,
    lambda: f32,
    textures_bind_group: BindGroup,
    pipeline: ComputePipeline,
    layers: u32,
}

impl WavesDataMergePipeline {
    pub fn init<'a>(
        device: &RenderDevice,
        size: u32,
        lambda: f32,
        amp_dx_dz_texture: &'a Texture,
        amp_dyx_dyz_texture: &'a Texture,
        displacement_texture: &'a Texture,
        derivatives_texture: &'a Texture,
    ) -> Self {
        let textures_bind_group_layout = device.create_bind_group_layout(
            "Waves data merge - texture bind group layout",
            &[
                // amp_dx_dz_texture
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
                // amp_dyx_dyz_texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                // out_displacement
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2Array,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadWrite,
                    },
                    count: None,
                },
                // out_derivatives
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

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Waves data merge pipeline layout"),
            bind_group_layouts: &[&textures_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<Parameters>() as u32,
            }],
        });

        let shader = unsafe {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Waves data merge shader"),
                source: ShaderSource::Wgsl(
                    include_str!("../../shaders/waves_data_merge.wgsl").into(),
                ),
            })
        };

        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("Waves data merge pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("merge"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let textures_bind_group = device.create_bind_group(
            "Waves data merge - textures",
            &textures_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&amp_dx_dz_texture.create_view(
                        &TextureViewDescriptor {
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&amp_dyx_dyz_texture.create_view(
                        &TextureViewDescriptor {
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&displacement_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&derivatives_texture.create_view(
                        &TextureViewDescriptor {
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            ..Default::default()
                        },
                    )),
                },
            ],
        );

        Self {
            size,
            lambda,
            textures_bind_group,
            pipeline,
            layers: displacement_texture.depth_or_array_layers(),
        }
    }

    // todo: Cleanup layer and use
    pub fn dispatch(&self, encoder: &mut CommandEncoder, dt: std::time::Duration, layer: u32) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Waves data merge"),
            timestamp_writes: None,
        });

        let parameters = Parameters {
            lambda: self.lambda,
            delta_time: dt.as_secs_f32(),
            layer: layer,
        };

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[parameters]));
        compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, 1);
    }
}
