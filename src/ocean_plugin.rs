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
    shader::{ShaderDefVal, ShaderRef},
};
use rand::prelude::*;

use crate::colors::{fog, ocean, sky, sun};
use crate::ocean::{OceanSurface, OceanSurfaceCascadeData, OceanSurfaceParameters};

const OCEAN_SHADER_PATH: &str = "embedded://bevy_ocean/shaders/ocean_shader.wgsl";
const NUMBER_OF_CASCADES: u32 = 3;

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
    params: OceanParams,
    wind_speed: f32,
    wind_direction: f32,
    swell: f32,
    choppiness: f32,
    depth: f32,
}

impl Default for OceanPlugin {
    fn default() -> Self {
        Self {
            quality: Quality::Low,
            params: OceanParams::default(),
            wind_speed: 10.0,
            wind_direction: 180.0,
            swell: 0.3,
            choppiness: 0.8,
            depth: 500.0,
        }
    }
}

impl OceanPlugin {
    pub fn calm() -> Self {
        Self {
            quality: Quality::Low,
            params: OceanParams::calm(),
            wind_speed: 3.0,
            wind_direction: 180.0,
            swell: 0.1,
            choppiness: 0.4,
            depth: 500.0,
        }
    }
}

#[derive(Resource, ExtractResource, Clone)]
pub struct OceanSettings {
    /// Layers
    pub number_of_cascades: u32,
    pub quality: Quality,
    /// Wind speed in m/s - controls wave energy (default 10.0)
    pub wind_speed: f32,
    /// Wind direction in degrees (0-360, default 180.0)
    pub wind_direction: f32,
    /// Swell contribution from distant storms (0.0-1.0, default 0.3)
    pub swell: f32,
    /// Choppiness - horizontal displacement intensity (0.0-1.0, default 0.8)
    pub choppiness: f32,
    /// Water depth in meters - affects wave dispersion in shallow water (default 500.0)
    pub depth: f32,
}

/// Ocean rendering parameters that can be modified at runtime.
/// Changes to this resource will be reflected in the water simulation.
#[derive(Resource, ExtractResource, Clone, Copy, Debug, ShaderType)]
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
    pub _padding: f32, // Or more if needed
    pub cascades: [CascadeParams; 8],
    pub cascade_count: u32,
}

impl OceanParams {
    pub fn default_cascade() -> [CascadeParams; 8] {
        [
            CascadeParams {
                length_scale: 500.,
                jacobian_strength: 0.6,
                lod_cutoff: 5000.,
                foam_strength: 0.6,
            },
            CascadeParams {
                length_scale: 85.,
                jacobian_strength: 0.17,
                lod_cutoff: 1000.,
                foam_strength: 0.25,
            },
            CascadeParams {
                length_scale: 10.,
                jacobian_strength: 0.23,
                lod_cutoff: 200.,
                foam_strength: 0.16,
            },
            CascadeParams::default(),
            CascadeParams::default(),
            CascadeParams::default(),
            CascadeParams::default(),
            CascadeParams::default(),
        ]
    }
    pub fn calm() -> Self {
        Self {
            displacement_scale: 1.0,
            normal_strength: 0.15,
            foam_threshold: 1.3,
            foam_multiplier: 0.5,
            foam_tile_scale: 8.0,
            roughness: 0.05,
            light_intensity: 3.0,
            sss_intensity: 0.6,
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
            ..Default::default()
        }
    }
}

impl Default for OceanParams {
    fn default() -> Self {
        let cascade_params = [
            CascadeParams {
                length_scale: 500.,
                jacobian_strength: 0.6,
                lod_cutoff: 0.,
                foam_strength: 0.6,
            },
            CascadeParams {
                length_scale: 85.,
                jacobian_strength: 0.17,
                lod_cutoff: 2000.,
                foam_strength: 0.25,
            },
            CascadeParams {
                length_scale: 10.,
                jacobian_strength: 0.23,
                lod_cutoff: 300.,
                foam_strength: 0.16,
            },
            CascadeParams::default(),
            CascadeParams::default(),
            CascadeParams::default(),
            CascadeParams::default(),
            CascadeParams::default(),
        ];
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
            cascades: cascade_params,
            cascade_count: cascade_params.len() as u32,
            _padding: 0.,
        }
    }
}

#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct CascadeParams {
    pub length_scale: f32,
    pub jacobian_strength: f32,
    pub lod_cutoff: f32,
    pub foam_strength: f32, // Single padding float to align to 16 bytes
}

/// A custom [`ExtendedMaterial`] that creates animated water ripples.
/// Uses 3 cascades for multi-scale wave detail:
/// - Cascade 0: 500m scale (large ocean swells)
/// - Cascade 1: 85m scale (medium waves)
/// - Cascade 2: 10m scale (small details/ripples)
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct OceanMaterial<const N: u32> {
    #[texture(0, dimension = "2d_array")]
    #[sampler(3)]
    pub t_displacements: Handle<Image>,

    #[texture(1, dimension = "2d_array")]
    pub t_derivatives: Handle<Image>,

    // Foam persistence textures (computed each frame)
    #[texture(2, dimension = "2d_array")]
    pub t_foam_persistences: Handle<Image>,
    // Cascade 0 - large scale (500m)
    // Foam texture
    #[texture(4)]
    #[sampler(5)]
    pub t_foam: Handle<Image>,
    // Ocean parameters uniform
    #[uniform(6)]
    pub params: OceanParams,
}

impl<const N: u32> Material for OceanMaterial<N> {
    fn fragment_shader() -> ShaderRef {
        OCEAN_SHADER_PATH.into()
    }

    fn vertex_shader() -> ShaderRef {
        OCEAN_SHADER_PATH.into()
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if N > 8 {
            panic!("cascades go up to a maximum of 8");
        }
        if let Some(fragment) = &mut descriptor.fragment {
            fragment
                .shader_defs
                .push(ShaderDefVal::UInt("NUMBER_OF_CASCADES".to_string(), N));
        }

        Ok(())
    }
}

#[derive(Resource, Clone, ExtractResource)]
pub struct OceanImages {
    pub displacement_image: Handle<Image>,
    pub derivative_image: Handle<Image>,
    pub foam_persistence_image: Handle<Image>,
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
        mut materials: ResMut<Assets<OceanMaterial<NUMBER_OF_CASCADES>>>,
        ocean_images: Res<OceanImages>,
        ocean_params: Res<OceanParams>,
        ocean_settings: Res<OceanSettings>,
        asset_server: Res<AssetServer>,
    ) {
        // Load foam texture
        let foam_texture: Handle<Image> =
            asset_server.load("embedded://bevy_ocean/textures/foam.png");

        // Create shared ocean material
        let ocean_material = materials.add(OceanMaterial {
            t_foam: foam_texture.clone(),
            params: *ocean_params,
            t_displacements: ocean_images.displacement_image.clone(),
            t_derivatives: ocean_images.derivative_image.clone(),
            t_foam_persistences: ocean_images.foam_persistence_image.clone(),
        });
        // LOD configuration: each ring has (inner_half, outer_half, subdivisions_per_chunk)
        // Ring 0 is a single center square (the parent)
        // Ring 1+ are 8 plane chunks arranged around the previous ring (children)
        // Uses powers of 2 so boundaries align exactly on vertex grids
        let lod_rings: [(f32, f32, u32); 3] = [
            (0.0, 1024.0, 512),              // Ring 0: Center square
            (1024.0, 4096.0 + 1024.0, 512),  // Ring 1: 8 chunks
            (4096.0 + 1024.0, 10000.0, 256), // Ring 2: 8 chunks
        ];

        // First, spawn the center plane as parent - all other meshes will be children
        let (_, center_outer, center_subdivs) = lod_rings[0];
        let center_cell_size = (center_outer * 2.0) / center_subdivs as f32;
        let center_plane = Plane3d::new(Vec3::Y, Vec2::splat(center_outer));
        let center_mesh = center_plane.mesh().subdivisions(center_subdivs).build();

        let parent_entity = commands
            .spawn((
                Mesh3d(meshes.add(center_mesh)),
                MeshMaterial3d(ocean_material.clone()),
                Transform::from_translation(Vec3::ZERO),
                OceanSurfaceMarker,
                OceanSnapSize(center_cell_size),
                NotShadowCaster,
                NotShadowReceiver,
            ))
            .id();

        // Now spawn outer rings as children of the center plane
        for (ring_idx, (inner_half, outer_half, subdivisions)) in lod_rings.iter().enumerate() {
            // Skip ring 0 (already spawned as parent)
            if *inner_half == 0.0 {
                continue;
            }

            let side_width = outer_half - inner_half;
            let side_length = inner_half * 2.0;
            let side_subdivs_width = *subdivisions;
            let side_subdivs_length = ((side_length / side_width) * *subdivisions as f32) as u32;

            // Side chunks (N, S, E, W) - rectangles with center offset
            let side_chunks = [
                // North
                (
                    0.0,
                    *inner_half + side_width / 2.0,
                    *inner_half,
                    side_width / 2.0,
                    side_subdivs_length,
                    side_subdivs_width,
                ),
                // South
                (
                    0.0,
                    -(*inner_half + side_width / 2.0),
                    *inner_half,
                    side_width / 2.0,
                    side_subdivs_length,
                    side_subdivs_width,
                ),
                // East
                (
                    *inner_half + side_width / 2.0,
                    0.0,
                    side_width / 2.0,
                    *inner_half,
                    side_subdivs_width,
                    side_subdivs_length,
                ),
                // West
                (
                    -(*inner_half + side_width / 2.0),
                    0.0,
                    side_width / 2.0,
                    *inner_half,
                    side_subdivs_width,
                    side_subdivs_length,
                ),
            ];

            // Corner chunks (NE, NW, SE, SW)
            let corner_offset = *inner_half + side_width / 2.0;
            let corner_half = side_width / 2.0;
            let corner_chunks = [
                (corner_offset, corner_offset),
                (-corner_offset, corner_offset),
                (corner_offset, -corner_offset),
                (-corner_offset, -corner_offset),
            ];

            // Spawn side chunks as children
            for (cx, cz, half_x, half_z, subdivs_x, subdivs_z) in side_chunks {
                let plane = Plane3d::new(Vec3::Y, Vec2::new(half_x, half_z));
                let mesh = plane.mesh().subdivisions(subdivs_x.max(subdivs_z)).build();
                let child = commands
                    .spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(ocean_material.clone()),
                        Transform::from_xyz(cx, 0.0, cz),
                        NotShadowCaster,
                        NotShadowReceiver,
                    ))
                    .id();
                commands.entity(parent_entity).add_child(child);
            }

            // Spawn corner chunks as children
            for (cx, cz) in corner_chunks {
                let plane = Plane3d::new(Vec3::Y, Vec2::splat(corner_half));
                let mesh = plane.mesh().subdivisions(*subdivisions).build();
                let child = commands
                    .spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(ocean_material.clone()),
                        Transform::from_xyz(cx, 0.0, cz),
                        NotShadowCaster,
                        NotShadowReceiver,
                    ))
                    .id();
                commands.entity(parent_entity).add_child(child);
            }

            // Create stitch strips as children
            let (_, prev_outer, prev_subdivisions) = lod_rings[ring_idx - 1];
            let fine_step = if ring_idx == 1 {
                (prev_outer * 2.0) / prev_subdivisions as f32
            } else {
                (prev_outer - lod_rings[ring_idx - 2].1) / prev_subdivisions as f32
            };
            let coarse_step = (outer_half - inner_half) / *subdivisions as f32;

            for direction in [
                StitchDirection::North,
                StitchDirection::South,
                StitchDirection::East,
                StitchDirection::West,
            ] {
                let stitch_mesh = Self::create_stitch_strip(
                    *inner_half,
                    fine_step,
                    coarse_step,
                    *inner_half,
                    *outer_half,
                    direction,
                );

                let child = commands
                    .spawn((
                        Mesh3d(meshes.add(stitch_mesh)),
                        MeshMaterial3d(ocean_material.clone()),
                        Transform::from_translation(Vec3::ZERO),
                        NotShadowCaster,
                        NotShadowReceiver,
                    ))
                    .id();
                commands.entity(parent_entity).add_child(child);
            }
        }
    }
    /// Make the ocean parent mesh follow the camera's XZ position for infinite ocean illusion
    /// Only the parent (center plane) has OceanSnapSize - children follow automatically
    fn ocean_follow_camera(
        camera_query: Query<&Transform, With<OceanCamera>>,
        mut ocean_query: Query<
            (&mut Transform, &OceanSnapSize),
            (With<OceanSurfaceMarker>, Without<OceanCamera>),
        >,
    ) {
        let Some(camera_transform) = camera_query.iter().next() else {
            return;
        };

        for (mut ocean_transform, snap_size) in &mut ocean_query {
            // Snap to grid based on center plane's cell size
            let snap = snap_size.0;
            let snapped_x = (camera_transform.translation.x / snap).floor() * snap;
            let snapped_z = (camera_transform.translation.z / snap).floor() * snap;

            ocean_transform.translation.x = snapped_x;
            ocean_transform.translation.z = snapped_z;
        }
    }

    /// Create a stitch strip that connects a fine (inner) ring edge to a coarse (outer) ring edge
    /// This bridges the vertex density gap between LOD levels (corners handled by corner chunks)
    fn create_stitch_strip(
        boundary: f32,       // The boundary position (e.g., 1024.0)
        fine_step: f32,      // Step size of finer (inner) ring
        coarse_step: f32,    // Step size of coarser (outer) ring
        fine_extent: f32,    // Half-extent of the edge to stitch
        _coarse_extent: f32, // Half-extent of coarse ring (unused)
        direction: StitchDirection,
    ) -> Mesh {
        use bevy::mesh::{Indices, PrimitiveTopology};

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Simple stitch strip - no corner handling (corners covered by corner chunks)
        // Fine edge: at boundary with fine_step spacing
        // Coarse edge: at boundary with coarse_step spacing (same position, different density)

        let fine_count = ((fine_extent * 2.0) / fine_step).round() as u32 + 1;
        let coarse_count = ((fine_extent * 2.0) / coarse_step).round() as u32 + 1;
        let ratio = (coarse_step / fine_step).round() as u32;

        match direction {
            StitchDirection::North => {
                let fine_z = boundary;
                let coarse_z = boundary; // Same position, will be offset by chunk

                // Fine edge vertices
                for i in 0..fine_count {
                    let x = -fine_extent + i as f32 * fine_step;
                    positions.push([x, 0.0, fine_z]);
                    normals.push([0.0, 1.0, 0.0]);
                    uvs.push([x, fine_z]);
                }

                // Coarse edge vertices (offset by coarse_step into the coarse ring)
                for i in 0..coarse_count {
                    let x = -fine_extent + i as f32 * coarse_step;
                    positions.push([x, 0.0, coarse_z + coarse_step]);
                    normals.push([0.0, 1.0, 0.0]);
                    uvs.push([x, coarse_z + coarse_step]);
                }

                Self::generate_stitch_triangles(
                    &mut indices,
                    0,
                    fine_count,
                    coarse_count,
                    ratio,
                    false,
                );
            }
            StitchDirection::South => {
                let fine_z = -boundary;
                let coarse_z = -boundary;

                for i in 0..fine_count {
                    let x = -fine_extent + i as f32 * fine_step;
                    positions.push([x, 0.0, fine_z]);
                    normals.push([0.0, 1.0, 0.0]);
                    uvs.push([x, fine_z]);
                }

                for i in 0..coarse_count {
                    let x = -fine_extent + i as f32 * coarse_step;
                    positions.push([x, 0.0, coarse_z - coarse_step]);
                    normals.push([0.0, 1.0, 0.0]);
                    uvs.push([x, coarse_z - coarse_step]);
                }

                Self::generate_stitch_triangles(
                    &mut indices,
                    0,
                    fine_count,
                    coarse_count,
                    ratio,
                    true,
                );
            }
            StitchDirection::East => {
                let fine_x = boundary;
                let coarse_x = boundary;

                for i in 0..fine_count {
                    let z = -fine_extent + i as f32 * fine_step;
                    positions.push([fine_x, 0.0, z]);
                    normals.push([0.0, 1.0, 0.0]);
                    uvs.push([fine_x, z]);
                }

                for i in 0..coarse_count {
                    let z = -fine_extent + i as f32 * coarse_step;
                    positions.push([coarse_x + coarse_step, 0.0, z]);
                    normals.push([0.0, 1.0, 0.0]);
                    uvs.push([coarse_x + coarse_step, z]);
                }

                Self::generate_stitch_triangles(
                    &mut indices,
                    0,
                    fine_count,
                    coarse_count,
                    ratio,
                    true,
                );
            }
            StitchDirection::West => {
                let fine_x = -boundary;
                let coarse_x = -boundary;

                for i in 0..fine_count {
                    let z = -fine_extent + i as f32 * fine_step;
                    positions.push([fine_x, 0.0, z]);
                    normals.push([0.0, 1.0, 0.0]);
                    uvs.push([fine_x, z]);
                }

                for i in 0..coarse_count {
                    let z = -fine_extent + i as f32 * coarse_step;
                    positions.push([coarse_x - coarse_step, 0.0, z]);
                    normals.push([0.0, 1.0, 0.0]);
                    uvs.push([coarse_x - coarse_step, z]);
                }

                Self::generate_stitch_triangles(
                    &mut indices,
                    0,
                    fine_count,
                    coarse_count,
                    ratio,
                    false,
                );
            }
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(Indices::U32(indices));
        mesh
    }

    /// Generate triangles connecting fine edge to coarse edge with fan pattern
    fn generate_stitch_triangles(
        indices: &mut Vec<u32>,
        fine_start: u32,
        fine_count: u32,
        coarse_count: u32,
        ratio: u32,
        reverse_winding: bool,
    ) {
        let coarse_start = fine_count;

        // For each segment between coarse vertices
        for c in 0..(coarse_count - 1) {
            let c0 = coarse_start + c;
            let c1 = coarse_start + c + 1;

            // Fine vertices that span between these two coarse vertices
            let f_base = fine_start + c * ratio;

            // Create fan of triangles from c0 to the fine vertices
            for f in 0..ratio {
                let f0 = f_base + f;
                let f1 = f_base + f + 1;

                if reverse_winding {
                    indices.push(c0);
                    indices.push(f0);
                    indices.push(f1);
                } else {
                    indices.push(c0);
                    indices.push(f1);
                    indices.push(f0);
                }
            }

            // Final triangle connecting last fine vertex to c1
            let f_last = f_base + ratio;
            if reverse_winding {
                indices.push(c0);
                indices.push(f_last);
                indices.push(c1);
            } else {
                indices.push(c0);
                indices.push(c1);
                indices.push(f_last);
            }
        }
    }
}

/// Direction for stitch strips between LOD rings
#[derive(Clone, Copy)]
enum StitchDirection {
    North, // +Z edge
    South, // -Z edge
    East,  // +X edge
    West,  // -X edge
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
        let Some(_ocean_params) = world.get_resource::<OceanParams>() else {
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
        app.insert_resource(self.params);
        app.insert_resource(OceanSettings {
            number_of_cascades: NUMBER_OF_CASCADES,
            quality: self.quality,
            wind_speed: self.wind_speed,
            wind_direction: self.wind_direction,
            swell: self.swell,
            choppiness: self.choppiness,
            depth: self.depth,
        });

        app.add_systems(Startup, (setup, OceanCamera::spawn_ocean).chain());
        app.add_systems(Update, OceanCamera::ocean_follow_camera);

        // Sync ocean params to materials every frame
        app.add_systems(Update, sync_ocean_params);

        app.add_plugins(MaterialPlugin::<OceanMaterial<NUMBER_OF_CASCADES>>::default());

        app.add_plugins((ExtractResourcePlugin::<OceanImages>::default(),));
        app.add_plugins((ExtractResourcePlugin::<OceanSettings>::default(),));
        app.add_plugins((ExtractResourcePlugin::<OceanParams>::default(),));

        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(OceanSettings {
            number_of_cascades: NUMBER_OF_CASCADES,
            quality: self.quality,
            wind_speed: self.wind_speed,
            wind_direction: self.wind_direction,
            swell: self.swell,
            choppiness: self.choppiness,
            depth: self.depth,
        });
        render_app.insert_resource(self.params);
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
    ocean_surface: OceanSurface<3>,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct OceanLabel;

/// Marker component for ocean surface entities
#[derive(Component)]
pub struct OceanSurfaceMarker;

/// System to sync OceanParams resource to all ocean materials
fn sync_ocean_params(
    ocean_params: Res<OceanParams>,
    mut materials: ResMut<Assets<OceanMaterial<3>>>,
    query: Query<&MeshMaterial3d<OceanMaterial<3>>, With<OceanSurfaceMarker>>,
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
        depth_or_array_layers: settings.number_of_cascades,
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

    let displacement_texture_array = image_assets.add(displacement_image);
    let derivatives_texture_array = image_assets.add(derivatives_image);
    let foam_persistence_texture_array = image_assets.add(foam_persistence_image);

    commands.insert_resource(OceanImages {
        displacement_image: displacement_texture_array,
        derivative_image: derivatives_texture_array,
        foam_persistence_image: foam_persistence_texture_array,
    });
}

// we need to re-init everytime the OceanParams change
fn init_ocean_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    // todo: We should use the cache and the bevy things I GUESS
    _pipeline_cache: Res<PipelineCache>,
    ocean_images: Option<Res<OceanImages>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    settings: Res<OceanSettings>,
) {
    // Wait for the resource to be extracted from main world
    let Some(ocean_images) = ocean_images else {
        return;
    };
    let ocean_params = OceanSurfaceParameters {
        size: settings.quality as u32,
        wind_speed: settings.wind_speed,
        wind_direction: settings.wind_direction,
        swell: settings.swell,
        choppiness: settings.choppiness,
        depth: settings.depth,
    };

    let displacement_texture = render_assets.get(&ocean_images.displacement_image).unwrap();
    let derivatives_texture = render_assets.get(&ocean_images.derivative_image).unwrap();
    let foam_persistence_texture = render_assets
        .get(&ocean_images.foam_persistence_image)
        .unwrap();

    let cascade_data = [
        OceanSurfaceCascadeData {
            displacement: &displacement_texture.texture,
            derivatives: &derivatives_texture.texture,
            foam_persistence: &foam_persistence_texture.texture,
            length_scale: 500.,
        },
        OceanSurfaceCascadeData {
            displacement: &displacement_texture.texture,
            derivatives: &derivatives_texture.texture,
            foam_persistence: &foam_persistence_texture.texture,
            length_scale: 85.,
        },
        OceanSurfaceCascadeData {
            displacement: &displacement_texture.texture,
            derivatives: &derivatives_texture.texture,
            foam_persistence: &foam_persistence_texture.texture,
            length_scale: 10.,
        },
    ];
    let ocean_resources = OceanPipeline {
        ocean_surface: OceanSurface::new(
            &render_device,
            settings.quality as u32,
            ocean_params,
            cascade_data,
        ),
    };
    commands.insert_resource(ocean_resources);
}
