use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{texture_2d, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    shader::ShaderRef,
};
use bevy_rand::{global::GlobalRng, prelude::WyRand};
use bytemuck::{Pod, Zeroable};
use rand::prelude::*;
use std::borrow::Cow;

const OCEAN_SHADER_PATH: &str = "shaders/ocean_shader.wgsl";
const INITIAL_SPECTRUM_PATH: &str = "shaders/initial_spectrum.wgsl";
const TIME_DEPENDENT_SPECTRUM_PATH: &str = "shaders/time_dependent_spectrum.wgsl";
const FFT_PATH: &str = "shaders/fft.wgsl";
const WAVES_DATA_MERGE_PATH: &str = "shaders/waves_data_merge.wgsl";
const SIZE: u32 = 64;

pub struct OceanPlugin;

/// A custom [`ExtendedMaterial`] that creates animated water ripples.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct OceanMaterial {
    #[texture(0)]
    #[sampler(2)]
    pub t_displacement: Handle<Image>,
    #[texture(1)]
    pub t_derivatives: Handle<Image>,
}

impl Material for OceanMaterial {
    fn fragment_shader() -> ShaderRef {
        OCEAN_SHADER_PATH.into()
    }

    fn vertex_shader() -> ShaderRef {
        OCEAN_SHADER_PATH.into()
    }
}

// problamo, if we want this to cascade we need to have the ocean_pipeline be outside of the plugin
// so we can run (dispatch) it multiple times
#[derive(Resource, Clone, ExtractResource)]
struct OceanImages {
    // our initial noise
    noise: Handle<Image>,

    h0_texture: Handle<Image>,
    waves_data_texture: Handle<Image>,
    h0k_texture: Handle<Image>,

    amp_dx_dz_dy_dxz_texture: Handle<Image>,
    amp_dyx_dyz_dxx_dzz_texture: Handle<Image>,

    fft_precompute_texture: Handle<Image>,
    fft_buffer_a_1_texture: Handle<Image>,
    fft_buffer_b_1_texture: Handle<Image>,

    out_displacement_image: Handle<Image>,
    out_derivatives_image: Handle<Image>,

    debug_texture: Handle<Image>,
}

impl OceanImages {
    fn debug(assets: Res<Assets<Image>>, images: Res<OceanImages>) {
        if let Some(image) = assets.get(&images.debug_texture) {}
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TimeDependentSpectrumPushParamaters {
    time: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FFTPushParamaters {
    ping_pong: u32,
    step: u32,
    size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WavesDataMergePushParamaters {
    lambda: f32,
    delta_time: f32,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct OceanParameters {
    size: u32,
    length_scale: f32,
    cut_off_low: f32,
    cut_off_high: f32,
    gravity_acceleration: f32,
    depth: f32,
}

impl Default for OceanParameters {
    fn default() -> Self {
        Self {
            size: 256u32,
            length_scale: 150.0,
            cut_off_low: 0.0001,
            cut_off_high: 9999.0,
            gravity_acceleration: 9.81,
            depth: 500.0,
        }
    }
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
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
    fn jonswap_alpha(g: f32, fetch: f32, wind_speed: f32) -> f32 {
        0.076 * f32::powf(g * fetch / wind_speed / wind_speed, -0.22)
    }

    fn jonswap_peak_frequency(g: f32, fetch: f32, wind_speed: f32) -> f32 {
        22.0 * f32::powf(wind_speed * fetch / g / g, -0.33)
    }
}

impl Default for SpectrumParamers {
    fn default() -> Self {
        let wind_direction = 200.0;
        let swell: f32 = 0.7;
        let fetch = 100000.0;
        let wind_speed = 0.5;
        let peak_enhancement = 3.3;
        Self {
            scale: 1.0,
            angle: wind_direction / 180.0 * std::f32::consts::PI,
            spread_blend: 1.0,
            swell: swell.clamp(0.01, 1.0),
            alpha: Self::jonswap_alpha(9.81, fetch, wind_speed),
            peak_omega: Self::jonswap_peak_frequency(9.81, fetch, wind_speed),
            gamma: peak_enhancement,
            short_waves_fade: 0.01,
        }
    }
}

struct OceanNode {
    state: OceanState,
}

enum OceanState {
    Loading,
    Init,
    TimeDependentSpectrum,
    // we use ping pong buffers here, need to figure out why lol
    InverseFFT(bool),
    WavesDataMerge,
    Pause,
}

impl Default for OceanNode {
    fn default() -> Self {
        Self {
            state: OceanState::Loading,
        }
    }
}

// we need to do this 4 times, hmmmmmmmm
impl render_graph::Node for OceanNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<OceanPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            OceanState::Loading => {
                let init_pipelines = [
                    pipeline.initial_spectrum_pipeline,
                    pipeline.calculate_conjugated_spectrum_pipeline,
                    pipeline.fft_precompute_pipeline,
                ];
                let mut loaded = 0;
                for current_pipeline in init_pipelines {
                    match pipeline_cache.get_compute_pipeline_state(current_pipeline) {
                        CachedPipelineState::Queued => {}
                        CachedPipelineState::Creating(_) => {}
                        CachedPipelineState::Ok(_) => {
                            loaded += 1;
                        }
                        CachedPipelineState::Err(pipeline_cache_error) => {
                            panic!(
                                "error getting pipeline with id: {} with error: {}",
                                current_pipeline.id(),
                                pipeline_cache_error
                            );
                        }
                    }
                }
                if loaded >= init_pipelines.len() {
                    self.state = OceanState::Init;
                }
            }
            OceanState::Init => {
                if let CachedPipelineState::Ok(_) = pipeline_cache
                    .get_compute_pipeline_state(pipeline.time_dependent_spectrum_pipeline)
                {
                    self.state = OceanState::TimeDependentSpectrum;
                }
                if let CachedPipelineState::Err(_) = pipeline_cache
                    .get_compute_pipeline_state(pipeline.time_dependent_spectrum_pipeline)
                {
                    error!("time dependent pipeline loaded");
                    panic!();
                }
            }
            OceanState::TimeDependentSpectrum => {
                let init_pipelines = [
                    pipeline.fft_swap_pipeline,
                    pipeline.fft_inverse_vertical_pipeline,
                    pipeline.fft_inverse_horizontal_pipeline,
                    pipeline.fft_visualize_pipeline,
                    pipeline.fft_permute_pipeline,
                ];
                let mut loaded = 0;
                for current_pipeline in init_pipelines {
                    match pipeline_cache.get_compute_pipeline_state(current_pipeline) {
                        CachedPipelineState::Queued => {}
                        CachedPipelineState::Creating(_) => {}
                        CachedPipelineState::Ok(_) => {
                            loaded += 1;
                        }
                        CachedPipelineState::Err(pipeline_cache_error) => {
                            panic!(
                                "error getting pipeline with id: {} with error: {}",
                                current_pipeline.id(),
                                pipeline_cache_error
                            );
                        }
                    }
                }
                if loaded >= init_pipelines.len() {
                    let ref mut ping_pong = world.resource_mut::<PingPong>();
                    self.state = OceanState::InverseFFT(ping_pong.ping);
                    // self.state = OceanState::TimeDependentSpectrum;
                    ping_pong.ping = !ping_pong.ping;
                }
            }
            OceanState::InverseFFT(_) => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.waves_data_merge_pipeline)
                {
                    self.state = OceanState::WavesDataMerge;
                }
                if let CachedPipelineState::Err(err) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.waves_data_merge_pipeline)
                {
                    error!("error for waves data merge: {:?}", err);
                    panic!();
                }
            }
            OceanState::WavesDataMerge => {
                self.state = OceanState::TimeDependentSpectrum;
            }
            OceanState::Pause => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        // select the pipeline based on the current state
        match self.state {
            OceanState::Loading => {}
            // i guess we only need to run the init here then transition to update?
            OceanState::Init => {
                let bind_groups = &world.resource::<OceanImageBindGroups>().0;
                let pipeline_cache = world.resource::<PipelineCache>();
                let pipeline = world.resource::<OceanPipeline>();

                {
                    let init_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.initial_spectrum_pipeline)
                        .unwrap();
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("Initial Spectrum"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_bind_group(0, &bind_groups[0], &[]);
                    pass.set_bind_group(1, &bind_groups[1], &[]);
                    pass.set_pipeline(init_pipeline);
                    let workgroup_size = 16u32;
                    let groups_x = (SIZE + workgroup_size - 1) / workgroup_size;
                    let groups_y = (SIZE + workgroup_size - 1) / workgroup_size;
                    pass.dispatch_workgroups(groups_x, groups_y, 1);
                }

                {
                    let conjugated_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.calculate_conjugated_spectrum_pipeline)
                        .unwrap();
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("Initial Conjugated Spectrum"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_bind_group(0, &bind_groups[0], &[]);
                    pass.set_bind_group(1, &bind_groups[1], &[]);
                    pass.set_pipeline(conjugated_pipeline);
                    let workgroup_size = 16u32;
                    let groups_x = (SIZE + workgroup_size - 1) / workgroup_size;
                    let groups_y = (SIZE + workgroup_size - 1) / workgroup_size;
                    pass.dispatch_workgroups(groups_x, groups_y, 1);
                }

                {
                    let fft_precompute_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.fft_precompute_pipeline)
                        .unwrap();
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("FFT Precompute"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_bind_group(0, &bind_groups[3], &[]);
                    pass.set_pipeline(fft_precompute_pipeline);
                    pass.set_push_constants(
                        0,
                        bytemuck::bytes_of(&FFTPushParamaters {
                            size: SIZE as u32,
                            ping_pong: 0,
                            step: 0,
                        }),
                    );
                    let log_size = (SIZE as f64).log(2.0) as u32;

                    pass.dispatch_workgroups(log_size, SIZE / 2 / 8, 1);
                }
            }
            OceanState::TimeDependentSpectrum => {
                let time = world.resource::<Time>();
                let bind_groups = &world.resource::<OceanImageBindGroups>().0;
                let pipeline_cache = world.resource::<PipelineCache>();
                let pipeline = world.resource::<OceanPipeline>();

                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("Time Dependent Spectrum"),
                            timestamp_writes: None,
                        });
                let time_dependent_spectrum_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.time_dependent_spectrum_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[2], &[]);
                pass.set_pipeline(time_dependent_spectrum_pipeline);
                pass.set_push_constants(
                    0,
                    bytemuck::bytes_of(&TimeDependentSpectrumPushParamaters {
                        time: time.elapsed_secs(),
                    }),
                );
                let workgroup_size = 16u32;
                let groups_x = (SIZE + workgroup_size - 1) / workgroup_size;
                let groups_y = (SIZE + workgroup_size - 1) / workgroup_size;
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            }
            OceanState::InverseFFT(ping) => {
                let log_size = (SIZE as f64).log(2.0) as u32;
                let bind_groups = &world.resource::<OceanImageBindGroups>().0;
                let pipeline_cache = world.resource::<PipelineCache>();
                let pipeline = world.resource::<OceanPipeline>();

                {
                    let fft_inverse_horizontal_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.fft_inverse_horizontal_pipeline)
                        .unwrap();
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("Inverse FFT - Horizontal"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_bind_group(0, &bind_groups[3], &[]);
                    pass.set_pipeline(fft_inverse_horizontal_pipeline);
                    for i in 0..log_size {
                        pass.set_push_constants(
                            0,
                            bytemuck::bytes_of(&FFTPushParamaters {
                                size: SIZE as u32,
                                ping_pong: ping as u32,
                                step: i,
                            }),
                        );
                        pass.dispatch_workgroups(SIZE / 16, SIZE / 16, 1);
                    }
                }

                {
                    let fft_inverse_vertical_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.fft_inverse_vertical_pipeline)
                        .unwrap();
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("Inverse FFT - Vertical"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_bind_group(0, &bind_groups[3], &[]);
                    pass.set_pipeline(fft_inverse_vertical_pipeline);
                    for i in 0..log_size {
                        pass.set_push_constants(
                            0,
                            bytemuck::bytes_of(&FFTPushParamaters {
                                size: SIZE as u32,
                                ping_pong: ping as u32,
                                step: i,
                            }),
                        );
                        pass.dispatch_workgroups(SIZE / 16, SIZE / 16, 1);
                    }
                }

                if ping {
                    let fft_swap_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.fft_swap_pipeline)
                        .unwrap();
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("Inverse FFT - Swap"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_bind_group(0, &bind_groups[3], &[]);
                    pass.set_pipeline(fft_swap_pipeline);
                    pass.set_push_constants(
                        0,
                        bytemuck::bytes_of(&FFTPushParamaters {
                            size: SIZE as u32,
                            ping_pong: ping as u32,
                            step: 0,
                        }),
                    );
                    pass.dispatch_workgroups(SIZE / 16, SIZE / 16, 1);
                }
                {
                    let fft_permute_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.fft_permute_pipeline)
                        .unwrap();
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("Inverse FFT - Permute"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_bind_group(0, &bind_groups[3], &[]);
                    pass.set_pipeline(fft_permute_pipeline);
                    pass.set_push_constants(
                        0,
                        bytemuck::bytes_of(&FFTPushParamaters {
                            size: SIZE as u32,
                            ping_pong: 0,
                            step: 0,
                        }),
                    );
                    pass.dispatch_workgroups(SIZE / 16, SIZE / 16, 1);
                }
                {
                    let fft_visualize_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.fft_visualize_pipeline)
                        .unwrap();
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("Inverse FFT - Visualize"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_bind_group(0, &bind_groups[3], &[]);
                    pass.set_pipeline(fft_visualize_pipeline);
                    pass.set_push_constants(
                        0,
                        bytemuck::bytes_of(&FFTPushParamaters {
                            size: SIZE as u32,
                            ping_pong: 0,
                            step: 0,
                        }),
                    );
                    pass.dispatch_workgroups(SIZE / 16, SIZE / 16, 1);
                }
            }
            OceanState::WavesDataMerge => {
                let time = world.resource::<Time>();
                let bind_groups = &world.resource::<OceanImageBindGroups>().0;
                let pipeline_cache = world.resource::<PipelineCache>();
                let pipeline = world.resource::<OceanPipeline>();
                let waves_data_merge_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.waves_data_merge_pipeline)
                    .unwrap();
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("Waves Data Merge"),
                            timestamp_writes: None,
                        });
                pass.set_bind_group(0, &bind_groups[4], &[]);
                pass.set_pipeline(waves_data_merge_pipeline);
                pass.set_push_constants(
                    0,
                    bytemuck::bytes_of(&WavesDataMergePushParamaters {
                        lambda: 1.2,
                        delta_time: time.delta_secs(),
                    }),
                );
                pass.dispatch_workgroups(SIZE / 16, SIZE / 16, 1);
            }
            OceanState::Pause => {}
        }
        Ok(())
    }
}

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup, spawn_debug_textures).chain());

        app.add_plugins(MaterialPlugin::<OceanMaterial>::default());

        app.add_plugins((
            ExtractResourcePlugin::<OceanImages>::default(),
            ExtractResourcePlugin::<OceanParameters>::default(),
            ExtractResourcePlugin::<SpectrumParamers>::default(),
        ));

        app.add_systems(Update, (update_ocean_time, OceanImages::debug));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_ocean_pipeline)
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(OceanLabel::InitialSpectrum, OceanNode::default());
        render_graph.add_node_edge(
            OceanLabel::InitialSpectrum,
            bevy::render::graph::CameraDriverLabel,
        );
    }
}

#[derive(Resource, Default)]
struct PingPong {
    ping: bool,
}

#[derive(Resource)]
struct OceanPipeline {
    initial_spectrum_bind_group_layout_0: BindGroupLayout,
    initial_spectrum_bind_group_layout_1: BindGroupLayout,

    initial_spectrum_pipeline: CachedComputePipelineId,
    calculate_conjugated_spectrum_pipeline: CachedComputePipelineId,

    time_dependent_spectrum_bind_group_layout_0: BindGroupLayout,
    time_dependent_spectrum_pipeline: CachedComputePipelineId,

    fft_bind_group_layout_0: BindGroupLayout,
    fft_precompute_pipeline: CachedComputePipelineId,
    fft_inverse_horizontal_pipeline: CachedComputePipelineId,
    fft_inverse_vertical_pipeline: CachedComputePipelineId,
    fft_swap_pipeline: CachedComputePipelineId,
    fft_permute_pipeline: CachedComputePipelineId,
    // for debugging
    fft_visualize_pipeline: CachedComputePipelineId,

    waves_data_merge_bind_group_layout_0: BindGroupLayout,
    waves_data_merge_pipeline: CachedComputePipelineId,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
enum OceanLabel {
    InitialSpectrum,
}

#[derive(Resource)]
struct OceanImageBindGroups([BindGroup; 5]);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<OceanPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    ocean_images: Res<OceanImages>,
    ocean_uniforms: Res<OceanParameters>,
    spectrum_parameters: Res<SpectrumParamers>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let view_noise = gpu_images.get(&ocean_images.noise).unwrap();
    let view_h0_texture = gpu_images.get(&ocean_images.h0_texture).unwrap();
    let view_waves_data_texture = gpu_images.get(&ocean_images.waves_data_texture).unwrap();
    let view_h0k_texture = gpu_images.get(&ocean_images.h0k_texture).unwrap();

    let view_amp_dx_dz_dy_dxz_texture = gpu_images
        .get(&ocean_images.amp_dx_dz_dy_dxz_texture)
        .unwrap();
    let view_amp_dyx_dyz_dxx_dzz_texture = gpu_images
        .get(&ocean_images.amp_dyx_dyz_dxx_dzz_texture)
        .unwrap();

    let view_fft_precompute_texture = gpu_images
        .get(&ocean_images.fft_precompute_texture)
        .unwrap();
    let view_fft_buffer_a_1_texture = gpu_images
        .get(&ocean_images.fft_buffer_a_1_texture)
        .unwrap();
    let view_fft_buffer_b_1_texture = gpu_images
        .get(&ocean_images.fft_buffer_b_1_texture)
        .unwrap();

    let view_out_displacements_texture = gpu_images
        .get(&ocean_images.out_displacement_image)
        .unwrap();
    let view_out_derivatives_texture = gpu_images.get(&ocean_images.out_derivatives_image).unwrap();

    let view_debug_texture = gpu_images.get(&ocean_images.debug_texture).unwrap();

    let mut ocean_params_uniform_buffer = UniformBuffer::from(ocean_uniforms.into_inner());
    ocean_params_uniform_buffer.write_buffer(&render_device, &queue);

    let mut spectrum_params_uniform_buffer = UniformBuffer::from(spectrum_parameters.into_inner());
    spectrum_params_uniform_buffer.write_buffer(&render_device, &queue);

    let initial_spectrum_bind_group_0 = render_device.create_bind_group(
        Some("ocean_bind_group_0"),
        &pipeline.initial_spectrum_bind_group_layout_0,
        &BindGroupEntries::sequential((
            &view_noise.texture_view,
            &view_h0_texture.texture_view,
            &view_waves_data_texture.texture_view,
            &view_h0k_texture.texture_view,
        )),
    );
    let initial_spectrum_bind_group_1 = render_device.create_bind_group(
        Some("initial_spectrum_bind_group_1"),
        &pipeline.initial_spectrum_bind_group_layout_1,
        &BindGroupEntries::sequential((
            &ocean_params_uniform_buffer,
            &spectrum_params_uniform_buffer,
        )),
    );

    let time_dependent_bind_group_0 = render_device.create_bind_group(
        Some("time_dependent_bind_group"),
        &pipeline.time_dependent_spectrum_bind_group_layout_0,
        &BindGroupEntries::sequential((
            &view_h0_texture.texture_view,
            &view_waves_data_texture.texture_view,
            &view_amp_dx_dz_dy_dxz_texture.texture_view,
            &view_amp_dyx_dyz_dxx_dzz_texture.texture_view,
        )),
    );

    let fft_bind_group = render_device.create_bind_group(
        Some("fft_bind_group_0"),
        &pipeline.fft_bind_group_layout_0,
        &BindGroupEntries::sequential((
            &view_fft_precompute_texture.texture_view,
            &view_amp_dx_dz_dy_dxz_texture.texture_view,
            &view_fft_buffer_a_1_texture.texture_view,
            &view_amp_dyx_dyz_dxx_dzz_texture.texture_view,
            &view_fft_buffer_b_1_texture.texture_view,
            &view_debug_texture.texture_view,
        )),
    );

    let waves_data_merge_bind_group = render_device.create_bind_group(
        Some("waves_data_merge_bind_group_0"),
        &pipeline.waves_data_merge_bind_group_layout_0,
        &BindGroupEntries::sequential((
            &view_fft_buffer_a_1_texture.texture_view,
            &view_fft_buffer_b_1_texture.texture_view,
            &view_out_displacements_texture.texture_view,
            &view_out_derivatives_texture.texture_view,
        )),
    );

    commands.insert_resource(OceanImageBindGroups([
        initial_spectrum_bind_group_0,
        initial_spectrum_bind_group_1,
        time_dependent_bind_group_0,
        fft_bind_group,
        waves_data_merge_bind_group,
    ]));
}

#[derive(Component)]
struct DebugMesh;

fn spawn_debug_textures(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OceanMaterial>>,
    ocean_images: Res<OceanImages>,
) {
    commands.spawn((
        DebugMesh,
        MeshMaterial3d(materials.add(OceanMaterial {
            t_displacement: ocean_images.out_displacement_image.clone(),
            t_derivatives: ocean_images.out_derivatives_image.clone(),
        })),
        Mesh3d(
            meshes.add(
                Plane3d::new(Vec3::Y, Vec2::new(128., 128.))
                    .mesh()
                    .subdivisions(1024),
            ),
        ),
        Transform::from_xyz(0., 0., 0.),
    ));
}

pub fn generate_noise_data<R: Rng + ?Sized>(rng: &mut R, size: usize) -> Vec<f32> {
    let mut buf: Vec<f32> = vec![0.; 4 * size * size];
    for i in 0..4 * size * size {
        buf[i] = rng.random();
    }

    return buf;
}

fn setup(
    mut commands: Commands,
    mut rng: Single<&mut WyRand, With<GlobalRng>>,
    mut images: ResMut<Assets<Image>>,
) {
    let mut image = Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba32Float);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let h0_image = images.add(image.clone());
    let waves_data_image = images.add(image.clone());
    let h0k_image = images.add(image.clone());

    let amp_dx_dz_dy_dxz_image = images.add(image.clone());
    let amp_dyx_dyz_dxx_dzz_image = images.add(image.clone());

    let fft_precompute_image = images.add(image.clone());
    let buffer_a_1_image = images.add(image.clone());
    let buffer_b_1_image = images.add(image.clone());

    let out_displacements_image = images.add(image.clone());
    let out_derivatives_image = images.add(image.clone());

    let mut debug_image = Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba8Unorm);
    debug_image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    debug_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let debug_texture = images.add(debug_image.clone());

    let noise_data: Vec<u8> =
        bytemuck::cast_slice::<f32, u8>(&generate_noise_data(&mut rng, SIZE as usize)).to_vec();

    let size = Extent3d {
        width: SIZE,
        height: SIZE,
        depth_or_array_layers: 1,
    };
    let mut noise_image = Image::new(
        size,
        TextureDimension::D2,
        noise_data,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    noise_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let noise_handle = images.add(noise_image.clone());

    // create our textures, these will be written by the compute shaders
    commands.insert_resource(OceanImages {
        h0_texture: h0_image,
        noise: noise_handle,
        waves_data_texture: waves_data_image,
        h0k_texture: h0k_image,
        amp_dx_dz_dy_dxz_texture: amp_dx_dz_dy_dxz_image,
        amp_dyx_dyz_dxx_dzz_texture: amp_dyx_dyz_dxx_dzz_image,
        fft_precompute_texture: fft_precompute_image,
        fft_buffer_a_1_texture: buffer_a_1_image,
        fft_buffer_b_1_texture: buffer_b_1_image,
        debug_texture,
        out_displacement_image: out_displacements_image,
        out_derivatives_image,
    });

    commands.insert_resource(OceanParameters::default());
    commands.insert_resource(SpectrumParamers::default());
}

fn init_ocean_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let initial_spectrum_bind_group_layout_0 = render_device.create_bind_group_layout(
        "initial_spectrum_bindgroup_0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
            ),
        ),
    );
    let initial_spectrum_bind_group_layout_1 = render_device.create_bind_group_layout(
        "initial_spectrum_bindgroup_1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<OceanParameters>(false),
                uniform_buffer::<SpectrumParamers>(false),
            ),
        ),
    );
    let shader = asset_server.load(INITIAL_SPECTRUM_PATH);
    let initial_spectrum_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![
                initial_spectrum_bind_group_layout_0.clone(),
                initial_spectrum_bind_group_layout_1.clone(),
            ],
            shader: shader.clone(),
            entry_point: Some(Cow::from("calculate_initial_spectrum")),
            ..default()
        });

    let calculate_conjugated_spectrum_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![
                initial_spectrum_bind_group_layout_0.clone(),
                initial_spectrum_bind_group_layout_1.clone(),
            ],
            shader: shader,
            entry_point: Some(Cow::from("calculate_conjugated_spectrum")),
            ..default()
        });

    let time_dependent_spectrum_bind_group_layout_0 = render_device.create_bind_group_layout(
        "time_dependent_spectrum_bindgroup_0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let shader = asset_server.load(TIME_DEPENDENT_SPECTRUM_PATH);
    let time_dependent_spectrum_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![time_dependent_spectrum_bind_group_layout_0.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("calculate_amplitudes")),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<TimeDependentSpectrumPushParamaters>() as u32,
            }],
            ..default()
        });

    let fft_bind_group_layout_0 = render_device.create_bind_group_layout(
        "fft_bindgroup_0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let shader = asset_server.load(FFT_PATH);
    let fft_precompute_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![fft_bind_group_layout_0.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("calculat_twiddle_factors_and_input_indices")),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<FFTPushParamaters>() as u32,
            }],
            ..default()
        });

    let fft_inverse_horizontal_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![fft_bind_group_layout_0.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("horizontal_step_inverse_fft")),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<FFTPushParamaters>() as u32,
            }],
            ..default()
        });

    let fft_inverse_vertical_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![fft_bind_group_layout_0.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("vertical_step_inverse_fft")),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<FFTPushParamaters>() as u32,
            }],
            ..default()
        });

    let fft_swap_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![fft_bind_group_layout_0.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("swap")),
        push_constant_ranges: vec![PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<FFTPushParamaters>() as u32,
        }],
        ..default()
    });
    let fft_permute_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![fft_bind_group_layout_0.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("permute")),
        push_constant_ranges: vec![PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<FFTPushParamaters>() as u32,
        }],
        ..default()
    });

    // todo: Probs remove this eventually
    let fft_visualize_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![fft_bind_group_layout_0.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("visualize_heightmap")),
        push_constant_ranges: vec![PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<FFTPushParamaters>() as u32,
        }],
        ..default()
    });

    let waves_data_merge_bind_group_layout_0 = render_device.create_bind_group_layout(
        "waves_data_merge_bindgroup_0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let shader = asset_server.load(WAVES_DATA_MERGE_PATH);
    let waves_data_merge_pipeline =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![waves_data_merge_bind_group_layout_0.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::from("merge")),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<WavesDataMergePushParamaters>() as u32,
            }],
            ..default()
        });

    commands.insert_resource(PingPong::default());
    commands.insert_resource(OceanPipeline {
        initial_spectrum_bind_group_layout_0,
        initial_spectrum_bind_group_layout_1,
        initial_spectrum_pipeline,
        calculate_conjugated_spectrum_pipeline,
        time_dependent_spectrum_bind_group_layout_0,
        time_dependent_spectrum_pipeline,
        fft_bind_group_layout_0,
        fft_precompute_pipeline,
        fft_inverse_horizontal_pipeline,
        fft_inverse_vertical_pipeline,
        fft_swap_pipeline,
        fft_visualize_pipeline,
        fft_permute_pipeline,
        waves_data_merge_bind_group_layout_0,
        waves_data_merge_pipeline,
    });
}

fn update_ocean_time(
    mut ocean_uniforms: ResMut<OceanParameters>,
    time: Res<Time>, // Bevy time resource
) {
    // Update the simulation time
    // ocean_uniforms.time += time.delta_secs();
}
