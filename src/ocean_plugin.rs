use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    ecs::schedule::common_conditions::{not, resource_exists},
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    light::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
    render::{
        Render, RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    shader::ShaderRef,
};
use rand::prelude::*;

use crate::colors::{fog, ocean, sky, sun};
use crate::ocean::{OceanCascade, OceanCascadeParameters};

const OCEAN_SHADER_PATH: &str = "embedded://bevy_ocean/shaders/ocean_shader.wgsl";

#[derive(Clone, Copy)]
pub enum Quality {
    Ultra = 1024,
    High = 512,
    Medium = 256,
    Low = 128,
    VeryLow = 64,
}

pub struct OceanPlugin {
    quality: Quality,
}

impl Default for OceanPlugin {
    fn default() -> Self {
        Self {
            quality: Quality::Low,
        }
    }
}

#[derive(Resource, ExtractResource, Clone)]
pub struct OceanSettings {
    quality: Quality,
}

/// Ocean rendering parameters that can be modified at runtime.
/// Changes to this resource will be reflected in the water simulation.
#[derive(Resource, Clone, Copy, Debug, ShaderType)]
pub struct OceanParams {
    /// Scale of wave displacement (0.0 - 2.0, default 0.6)
    pub displacement_scale: f32,
    /// Strength of surface normals (0.0 - 2.0, default 0.8)
    pub normal_strength: f32,
    /// Jacobian threshold for foam (0.0 - 2.0, default 1.0)
    pub foam_threshold: f32,
    /// Foam intensity multiplier (0.0 - 5.0, default 2.0)
    pub foam_multiplier: f32,
    /// Foam texture tiling scale (1.0 - 20.0, default 8.0)
    pub foam_tile_scale: f32,
    /// Water surface roughness for PBR (0.0 - 1.0, default 0.05)
    pub roughness: f32,
    /// Sun/light intensity (0.0 - 10.0, default 3.0)
    pub light_intensity: f32,
    /// Subsurface scattering intensity (0.0 - 1.0, default 0.4)
    pub sss_intensity: f32,
    /// Sun direction vector (will be normalized in shader)
    pub sun_direction: Vec3,
    /// Fog color (matches sky horizon, updated by day_night_cycle)
    pub fog_color: Vec3,
    /// Distance where fog starts (default 500.0)
    pub fog_start: f32,
    /// Distance where fog is fully opaque (default 5000.0)
    pub fog_end: f32,

    // Ocean colors
    /// Deep ocean color (looking straight down)
    pub deep_color: Vec3,
    /// Shallow ocean color (at grazing angles)
    pub shallow_color: Vec3,
    /// Sky reflection color during day
    pub sky_day: Vec3,
    /// Sky reflection color at night
    pub sky_night: Vec3,
    /// Sun color for specular highlights
    pub sun_color: Vec3,
    /// Subsurface scattering color (turquoise glow)
    pub sss_color: Vec3,
    /// Foam highlight color
    pub foam_color: Vec3,
    /// Ambient light color
    pub ambient_color: Vec3,
}

impl Default for OceanParams {
    fn default() -> Self {
        Self {
            displacement_scale: 1.0,
            normal_strength: 0.3,
            foam_threshold: 1.0,
            foam_multiplier: 2.0,
            foam_tile_scale: 8.0,
            roughness: 0.1,
            light_intensity: 3.0,
            sss_intensity: 1.0,
            sun_direction: Vec3::new(0.3, 0.8, 0.2),
            fog_color: fog::COLOR_DAY,
            fog_start: 8192.0,
            fog_end: 32768.0,
            deep_color: ocean::DEEP,
            shallow_color: ocean::SHALLOW,
            sky_day: sky::REFLECTION_DAY,
            sky_night: sky::REFLECTION_NIGHT,
            sun_color: sun::COLOR_OCEAN,
            sss_color: ocean::SSS,
            foam_color: ocean::FOAM,
            ambient_color: ocean::AMBIENT,
        }
    }
}

/// A custom [`ExtendedMaterial`] that creates animated water ripples.
/// Uses 3 cascades for multi-scale wave detail:
/// - Cascade 0: 500m scale (large ocean swells)
/// - Cascade 1: 85m scale (medium waves)
/// - Cascade 2: 10m scale (small details/ripples)
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct OceanMaterial {
    // Cascade 0 - large scale (500m)
    #[texture(0)]
    #[sampler(6)]
    pub t_displacement_0: Handle<Image>,
    #[texture(1)]
    pub t_derivatives_0: Handle<Image>,
    // Cascade 1 - medium scale (85m)
    #[texture(2)]
    pub t_displacement_1: Handle<Image>,
    #[texture(3)]
    pub t_derivatives_1: Handle<Image>,
    // Cascade 2 - small scale (10m)
    #[texture(4)]
    pub t_displacement_2: Handle<Image>,
    #[texture(5)]
    pub t_derivatives_2: Handle<Image>,
    // Foam texture
    #[texture(7)]
    #[sampler(8)]
    pub t_foam: Handle<Image>,
    // Ocean parameters uniform
    #[uniform(9)]
    pub params: OceanParams,
    // Foam persistence textures (computed each frame)
    #[texture(10)]
    pub t_foam_persistence_0: Handle<Image>,
    #[texture(11)]
    pub t_foam_persistence_1: Handle<Image>,
    #[texture(12)]
    pub t_foam_persistence_2: Handle<Image>,
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
pub struct OceanImages {
    pub displacement_0: Handle<Image>,
    pub displacement_1: Handle<Image>,
    pub displacement_2: Handle<Image>,
    pub derivatives_0: Handle<Image>,
    pub derivatives_1: Handle<Image>,
    pub derivatives_2: Handle<Image>,
    pub foam_persistence_0: Handle<Image>,
    pub foam_persistence_1: Handle<Image>,
    pub foam_persistence_2: Handle<Image>,
}

#[derive(Component)]
#[require(Transform)]
pub struct OceanCamera;

/// Component storing the grid snap size for each ocean LOD ring
/// Each ring snaps to its own cell size to prevent vertex swimming
#[derive(Component)]
struct OceanSnapSize(f32);

impl OceanCamera {
    fn spawn_ocean(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<OceanMaterial>>,
        ocean_images: Res<OceanImages>,
        ocean_params: Res<OceanParams>,
        asset_server: Res<AssetServer>,
    ) {
        // Load foam texture
        let foam_texture: Handle<Image> =
            asset_server.load("embedded://bevy_ocean/textures/foam.png");

        // Create shared ocean material
        let ocean_material = materials.add(OceanMaterial {
            t_displacement_0: ocean_images.displacement_0.clone(),
            t_derivatives_0: ocean_images.derivatives_0.clone(),
            t_displacement_1: ocean_images.displacement_1.clone(),
            t_derivatives_1: ocean_images.derivatives_1.clone(),
            t_displacement_2: ocean_images.displacement_2.clone(),
            t_derivatives_2: ocean_images.derivatives_2.clone(),
            t_foam: foam_texture.clone(),
            params: *ocean_params,
            t_foam_persistence_0: ocean_images.foam_persistence_0.clone(),
            t_foam_persistence_1: ocean_images.foam_persistence_1.clone(),
            t_foam_persistence_2: ocean_images.foam_persistence_2.clone(),
        });

        // LOD rings configuration: (inner_half_size, outer_half_size, subdivisions)
        // Each ring is a square frame that fills the gap between sizes
        // inner_half_size=0 creates a solid square (center patch)
        // subdivisions = cells along the full outer edge (like a plane)
        let lod_rings = [
            (0.0, 256.0, 1024),            // Ring 0: Center square, highest detail
            (256.0, 2048.0 + 256.0, 1024), // Ring 1: Medium detail frame
            (2048.0 + 256., 8192.0, 512),  // Ring 2: Low detail frame
            (8192.0, 32768.0, 256),        // Ring 3: Very low detail, extends to horizon
        ];

        for (inner_half, outer_half, subdivisions) in lod_rings {
            let mesh = Self::create_square_ring_mesh(inner_half, outer_half, subdivisions);

            // Calculate cell size for this ring: total_size / subdivisions
            let cell_size = (outer_half * 2.0) / subdivisions as f32;

            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(ocean_material.clone()),
                Transform::from_translation(Vec3::ZERO),
                OceanSurface,
                OceanSnapSize(cell_size),
                NotShadowCaster,
                NotShadowReceiver,
            ));
        }
    }
    /// Make the ocean mesh follow the camera's XZ position for infinite ocean illusion
    /// Each ring snaps to its own cell size to prevent vertex swimming
    fn ocean_follow_camera(
        camera_query: Query<&Transform, With<OceanCamera>>,
        mut ocean_query: Query<
            (&mut Transform, &OceanSnapSize),
            (With<OceanSurface>, Without<OceanCamera>),
        >,
    ) {
        let Some(camera_transform) = camera_query.iter().next() else {
            return;
        };

        for (mut ocean_transform, snap_size) in &mut ocean_query {
            // Snap camera position to this ring's grid cell size
            let snap = snap_size.0;
            let snapped_x = (camera_transform.translation.x / snap).floor() * snap;
            let snapped_z = (camera_transform.translation.z / snap).floor() * snap;

            // Follow camera on XZ plane (snapped), keep Y at 0
            ocean_transform.translation.x = snapped_x;
            ocean_transform.translation.z = snapped_z;
        }
    }

    /// Generate a square ring (frame) mesh for ocean LOD
    /// inner_half_size: half-width of the inner cutout (0.0 for solid square)
    /// outer_half_size: half-width of the outer edge
    /// subdivisions: number of cells along the full outer edge (like a plane)
    ///
    /// Creates a grid matching a full plane's subdivisions, but skips
    /// triangles for quads entirely inside the inner square.
    fn create_square_ring_mesh(
        inner_half_size: f32,
        outer_half_size: f32,
        subdivisions: u32,
    ) -> Mesh {
        use bevy::mesh::{Indices, PrimitiveTopology};

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let step = (2.0 * outer_half_size) / subdivisions as f32;
        let verts_per_row = subdivisions + 1;

        // Generate full grid of vertices (same as a plane would have)
        for j in 0..=subdivisions {
            for i in 0..=subdivisions {
                let x = -outer_half_size + i as f32 * step;
                let z = -outer_half_size + j as f32 * step;

                positions.push([x, 0.0, z]);
                normals.push([0.0, 1.0, 0.0]);
                uvs.push([x, z]);
            }
        }

        // Generate triangles, skipping quads entirely inside the inner square
        for j in 0..subdivisions {
            for i in 0..subdivisions {
                // Calculate quad bounds
                let x_min = -outer_half_size + i as f32 * step;
                let x_max = x_min + step;
                let z_min = -outer_half_size + j as f32 * step;
                let z_max = z_min + step;

                // Skip if quad is entirely inside the inner square (with small epsilon)
                let eps = step * 0.01;
                let inside = x_min >= -inner_half_size + eps
                    && x_max <= inner_half_size - eps
                    && z_min >= -inner_half_size + eps
                    && z_max <= inner_half_size - eps;

                if !inside {
                    // Vertex indices for this quad
                    let bl = j * verts_per_row + i; // bottom-left
                    let br = j * verts_per_row + i + 1; // bottom-right
                    let tl = (j + 1) * verts_per_row + i; // top-left
                    let tr = (j + 1) * verts_per_row + i + 1; // top-right

                    // Two triangles (CCW winding for +Y normal)
                    indices.push(bl);
                    indices.push(tl);
                    indices.push(br);

                    indices.push(br);
                    indices.push(tl);
                    indices.push(tr);
                }
            }
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(Indices::U32(indices));
        mesh
    }
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
        // Wait for pipeline to be created
        let Some(_pipeline) = world.get_resource::<OceanPipeline>() else {
            return;
        };

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            OceanState::Init(0) => {
                self.state = OceanState::Init(1);
            }
            OceanState::Init(_) => {
                self.state = OceanState::Run;
            }
            OceanState::Run => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        // Wait for pipeline to be created
        let Some(pipeline) = world.get_resource::<OceanPipeline>() else {
            return Ok(());
        };
        let render_queue = world.resource::<RenderQueue>();
        let time = world.resource::<Time>();
        let mut encoder = render_context.command_encoder();
        // select the pipeline based on the current state
        match self.state {
            OceanState::Init(1) => {
                pipeline.ocean_surface.init(&mut encoder, render_queue);
            }
            OceanState::Init(_) => {}
            OceanState::Run => {
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
        let strip_prefix = "src/";
        embedded_asset!(app, strip_prefix, "./shaders/ocean_shader.wgsl");
        embedded_asset!(app, strip_prefix, "./textures/foam.png");
        // Insert default ocean parameters resource
        app.init_resource::<OceanParams>();
        app.insert_resource(OceanSettings {
            quality: self.quality,
        });

        app.add_systems(Startup, (setup, OceanCamera::spawn_ocean).chain());
        app.add_systems(Update, OceanCamera::ocean_follow_camera);

        // Sync ocean params to materials every frame
        app.add_systems(Update, sync_ocean_params);

        app.add_plugins(MaterialPlugin::<OceanMaterial>::default());

        app.add_plugins((ExtractResourcePlugin::<OceanImages>::default(),));
        app.add_plugins((ExtractResourcePlugin::<OceanSettings>::default(),));

        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(OceanSettings {
            quality: self.quality,
        });
        // Run init_ocean_pipeline in ExtractCommands phase so it runs after extraction
        render_app.add_systems(
            Render,
            init_ocean_pipeline.run_if(not(resource_exists::<OceanPipeline>)),
        );

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

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct OceanLabel;

/// Marker component for ocean surface entities
#[derive(Component)]
pub struct OceanSurface;

/// System to sync OceanParams resource to all ocean materials
fn sync_ocean_params(
    ocean_params: Res<OceanParams>,
    mut materials: ResMut<Assets<OceanMaterial>>,
    query: Query<&MeshMaterial3d<OceanMaterial>, With<OceanSurface>>,
) {
    if !ocean_params.is_changed() {
        return;
    }

    for material_handle in query.iter() {
        if let Some(material) = materials.get_mut(material_handle) {
            material.params = *ocean_params;
        }
    }
}

pub fn generate_noise_data<R: Rng + ?Sized>(rng: &mut R, size: usize) -> Vec<f32> {
    let mut buf: Vec<f32> = vec![0.; 4 * size * size];
    for i in 0..4 * size * size {
        buf[i] = rng.random();
    }

    return buf;
}

pub fn setup(
    mut commands: Commands,
    mut image_assets: ResMut<Assets<Image>>,
    settings: Res<OceanSettings>,
) {
    let texture_size = Extent3d {
        width: settings.quality as u32,
        height: settings.quality as u32,
        depth_or_array_layers: 1,
    };
    let displacement_descriptor = TextureDescriptor {
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
    let derivatives_descriptor = TextureDescriptor {
        label: Some("Derivatives"),
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
    let repeat_sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..Default::default()
    });
    let displacement_image = Image {
        data: None,
        texture_descriptor: displacement_descriptor,
        sampler: repeat_sampler.clone(),
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        ..Default::default()
    };
    let derivatives_image = Image {
        data: None,
        texture_descriptor: derivatives_descriptor,
        sampler: repeat_sampler.clone(),
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        ..Default::default()
    };

    // Foam persistence texture - single channel R32Float
    let foam_persistence_descriptor = TextureDescriptor {
        label: Some("Foam Persistence"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::COPY_SRC
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING,
        view_formats: &[TextureFormat::R32Float],
    };
    let foam_persistence_image = Image {
        data: None,
        texture_descriptor: foam_persistence_descriptor,
        sampler: repeat_sampler,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        ..Default::default()
    };

    let displacement_0_texture = image_assets.add(displacement_image.clone());
    let displacement_1_texture = image_assets.add(displacement_image.clone());
    let displacement_2_texture = image_assets.add(displacement_image);
    let derivatives_0_texture = image_assets.add(derivatives_image.clone());
    let derivatives_1_texture = image_assets.add(derivatives_image.clone());
    let derivatives_2_texture = image_assets.add(derivatives_image);
    let foam_persistence_0_texture = image_assets.add(foam_persistence_image.clone());
    let foam_persistence_1_texture = image_assets.add(foam_persistence_image.clone());
    let foam_persistence_2_texture = image_assets.add(foam_persistence_image);

    commands.insert_resource(OceanImages {
        displacement_0: displacement_0_texture,
        displacement_1: displacement_1_texture,
        displacement_2: displacement_2_texture,
        derivatives_0: derivatives_0_texture,
        derivatives_1: derivatives_1_texture,
        derivatives_2: derivatives_2_texture,
        foam_persistence_0: foam_persistence_0_texture,
        foam_persistence_1: foam_persistence_1_texture,
        foam_persistence_2: foam_persistence_2_texture,
    });
}

// we need to re-init everytime the OceanParams change
fn init_ocean_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    // todo: We should use the cache and the bevy things I GUESS
    pipeline_cache: Res<PipelineCache>,
    ocean_images: Option<Res<OceanImages>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    settings: Res<OceanSettings>,
) {
    // Wait for the resource to be extracted from main world
    let Some(ocean_images) = ocean_images else {
        return;
    };
    let ocean_params = OceanCascadeParameters {
        size: settings.quality as u32,
        wind_speed: 10.0,
        wind_direction: 180.0,
        swell: 0.3,
    };

    let displacement_0_texture = render_assets.get(&ocean_images.displacement_0).unwrap();
    let displacement_1_texture = render_assets.get(&ocean_images.displacement_1).unwrap();
    let displacement_2_texture = render_assets.get(&ocean_images.displacement_2).unwrap();
    let derivatives_0_texture = render_assets.get(&ocean_images.derivatives_0).unwrap();
    let derivatives_1_texture = render_assets.get(&ocean_images.derivatives_1).unwrap();
    let derivatives_2_texture = render_assets.get(&ocean_images.derivatives_2).unwrap();
    let foam_persistence_0_texture = render_assets.get(&ocean_images.foam_persistence_0).unwrap();
    let foam_persistence_1_texture = render_assets.get(&ocean_images.foam_persistence_1).unwrap();
    let foam_persistence_2_texture = render_assets.get(&ocean_images.foam_persistence_2).unwrap();

    let ocean_resources = OceanPipeline {
        ocean_surface: OceanCascade::new(
            &render_device,
            settings.quality as u32,
            ocean_params,
            &displacement_0_texture.texture,
            &displacement_1_texture.texture,
            &displacement_2_texture.texture,
            &derivatives_0_texture.texture,
            &derivatives_1_texture.texture,
            &derivatives_2_texture.texture,
            &foam_persistence_0_texture.texture,
            &foam_persistence_1_texture.texture,
            &foam_persistence_2_texture.texture,
        ),
    };
    commands.insert_resource(ocean_resources);
}
