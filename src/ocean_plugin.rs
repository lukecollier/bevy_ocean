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

use crate::ocean::{OceanCascade, OceanCascadeParameters};

const OCEAN_SHADER_PATH: &str = "shaders/ocean_shader.wgsl";
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

#[derive(Resource, Clone, ExtractResource)]
struct OceanImages {
    displacement_0: Handle<Image>,
    displacement_1: Handle<Image>,
    displacement_2: Handle<Image>,
}

impl OceanImages {
    // todo: Draw our displacement map to the UI
    fn debug(assets: Res<Assets<Image>>, images: Res<OceanImages>) {}
}

struct OceanNode {
    state: OceanState,
}

#[derive(Debug)]
enum OceanState {
    Init(usize),
    Run,
}

// we need to do this 4 times, hmmmmmmmm
impl render_graph::Node for OceanNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<OceanPipeline>();
        // todo: I think this stuff need's to like run or something?
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            OceanState::Init(0) => {
                info!("inited");
                self.state = OceanState::Init(1);
            }
            OceanState::Init(_) => {
                info!("inited");
                self.state = OceanState::Run;
            }
            OceanState::Run => {
                debug!("run");
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline = world.resource::<OceanPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_queue = world.resource::<RenderQueue>();
        let time = world.resource::<Time>();
        let mut encoder = render_context.command_encoder();
        // select the pipeline based on the current state
        match self.state {
            OceanState::Init(1) => {
                pipeline.ocean_surface.init(&mut encoder, render_queue);
                info!("ran_init");
            }
            OceanState::Init(_) => {}
            OceanState::Run => {
                info!("run");
                pipeline.ocean_surface.dispatch(
                    &mut encoder,
                    render_queue,
                    time.elapsed_secs(),
                    time.delta(),
                );
            }
        }
        Ok(())
    }
}

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup, spawn_debug_textures).chain());

        app.add_plugins(MaterialPlugin::<OceanMaterial>::default());

        app.add_plugins((ExtractResourcePlugin::<OceanImages>::default(),));

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(RenderStartup, (init_ocean_pipeline).chain());

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(
            OceanLabel,
            OceanNode {
                state: OceanState::Init(0),
            },
        );
        render_graph.add_node_edge(OceanLabel, bevy::render::graph::CameraDriverLabel);
    }
}

#[derive(Resource)]
struct OceanPipeline {
    ocean_surface: OceanCascade,
}

#[derive(Resource)]
struct OceanPipelineInit {
    init: bool,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct OceanLabel;

#[derive(Component)]
struct DebugMesh;

fn spawn_debug_textures(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OceanMaterial>>,
    ocean_images: Res<OceanImages>,
) {
}

pub fn generate_noise_data<R: Rng + ?Sized>(rng: &mut R, size: usize) -> Vec<f32> {
    let mut buf: Vec<f32> = vec![0.; 4 * size * size];
    for i in 0..4 * size * size {
        buf[i] = rng.random();
    }

    return buf;
}

fn setup(mut commands: Commands, mut image_assets: ResMut<Assets<Image>>) {
    let texture_size = Extent3d {
        width: SIZE,
        height: SIZE,
        depth_or_array_layers: 1,
    };
    let texture_descriptor = TextureDescriptor {
        label: Some("Displacement"),
        size: texture_size,
        mip_level_count: 4,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::COPY_SRC
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING,
        view_formats: &[TextureFormat::Rgba32Float],
    };
    let image = Image {
        data: None,
        texture_descriptor: texture_descriptor,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        ..Default::default()
    };
    let displacement_0_texture = image_assets.add(image.clone());
    let displacement_1_texture = image_assets.add(image.clone());
    let displacement_2_texture = image_assets.add(image);
    // create our textures, these will be written by the compute shaders
    commands.insert_resource(OceanImages {
        displacement_0: displacement_0_texture,
        displacement_1: displacement_1_texture,
        displacement_2: displacement_2_texture,
    });
}

// we need to re-init everytime the OceanParams change
fn init_ocean_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    // todo: We should use the cache and the bevy things I GUESS
    _pipeline_cache: Res<PipelineCache>,
    ocean_images: Res<OceanImages>,
    render_assets: Res<RenderAssets<GpuImage>>,
) {
    let ocean_params = OceanCascadeParameters {
        size: SIZE,
        wind_speed: 10.0,
        wind_direction: 180.0,
        swell: 0.3,
    };

    let displacement_0_texture = render_assets.get(&ocean_images.displacement_0).unwrap();
    let displacement_1_texture = render_assets.get(&ocean_images.displacement_1).unwrap();
    let displacement_2_texture = render_assets.get(&ocean_images.displacement_2).unwrap();
    // is this right for us? I DUNNO MAYBE IT'LL WORK
    let ocean_resources = OceanPipeline {
        ocean_surface: OceanCascade::new(
            &render_device,
            SIZE,
            ocean_params,
            &displacement_0_texture.texture,
            &displacement_1_texture.texture,
            &displacement_2_texture.texture,
        ),
    };
    commands.insert_resource(ocean_resources);
}
