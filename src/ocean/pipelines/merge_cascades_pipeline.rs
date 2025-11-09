use std::rc::Rc;

use bevy::render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType,
        CommandEncoder, ComputePassDescriptor, ComputePipeline, Extent3d,
        PipelineCompilationOptions, PipelineLayoutDescriptor, PushConstantRange,
        RawComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
        StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::RenderDevice,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
    lambda: f32,
    delta_time: f32,
}

pub struct MergeCascadesPipeline {
    size: u32,

    pub cascade_0: Rc<Texture>,
    pub cascade_1: Rc<Texture>,
    pub cascade_2: Rc<Texture>,

    pub merged_displacement: Texture,

    textures_bind_group: BindGroup,
    pipeline: ComputePipeline,
}

impl MergeCascadesPipeline {
    pub fn init(
        device: &RenderDevice,
        size: u32,
        cascade_0: Rc<Texture>,
        cascade_1: Rc<Texture>,
        cascade_2: Rc<Texture>,
    ) -> Self {
        let textures_bind_group_layout = device.create_bind_group_layout(
            "Waves data merge - texture bind group layout",
            &[
                // cascade_0
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
                // cascade_1
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
                // cascade_2
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                // merged_displacement
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
            ],
        );

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Cascade merge pipeline layout"),
            bind_group_layouts: &[&textures_bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..8,
            }],
        });

        let shader = unsafe {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Merge cascades shader"),
                source: ShaderSource::Wgsl(
                    include_str!("../../../assets/shaders/waves_data_merge.wgsl").into(),
                ),
            })
        };

        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("Merge cascades pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("merge"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let texture_size = Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };

        let merged_displacement = device.create_texture(&TextureDescriptor {
            label: Some("Cascade 0"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let textures_bind_group = device.create_bind_group(
            "Merge cascades bind group",
            &textures_bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&cascade_0.create_view(
                        &TextureViewDescriptor {
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&cascade_1.create_view(
                        &TextureViewDescriptor {
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&cascade_2.create_view(
                        &TextureViewDescriptor {
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&merged_displacement.create_view(
                        &TextureViewDescriptor {
                            ..Default::default()
                        },
                    )),
                },
            ],
        );

        Self {
            size,
            merged_displacement,
            cascade_0,
            cascade_1,
            cascade_2,
            textures_bind_group,
            pipeline,
        }
    }

    pub fn dispatch(&self, encoder: &mut CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Waves data merge"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.textures_bind_group, &[]);
        compute_pass.dispatch_workgroups(self.size / 16, self.size / 16, 1);
    }
}
