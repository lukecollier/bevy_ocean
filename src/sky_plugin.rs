use bevy::{asset::embedded_asset, prelude::*, render::render_resource::*, shader::ShaderRef};

use crate::colors::{ocean, scatter, sky, sun, water};

const SKY_SHADER_PATH: &str = "embedded://bevy_ocean/shaders/sky_shader.wgsl";

pub struct SkyPlugin;

/// Sky rendering parameters that can be modified at runtime.
#[derive(Resource, Clone, Copy, Debug, ShaderType)]
pub struct SkyParams {
    // Sky gradient colors
    /// Horizon color during day
    pub horizon_day: Vec3,
    /// Zenith (overhead) color during day
    pub zenith_day: Vec3,
    /// Horizon color at night
    pub horizon_night: Vec3,
    /// Zenith color at night
    pub zenith_night: Vec3,

    // Sun disc settings
    /// Angular radius of the sun disc
    pub sun_size: f32,
    /// Intensity of the glow around the sun
    pub sun_glow_intensity: f32,
    /// Falloff exponent for sun glow
    pub sun_glow_falloff: f32,

    // Sun state (updated by day_night_cycle)
    /// Normalized sun direction vector
    pub sun_direction: Vec3,
    /// Sun color (RGB)
    pub sun_color: Vec3,
    /// Sun intensity (0.0 at night, 1.0+ during day)
    pub sun_intensity: f32,
    /// Sun core color (very bright center)
    pub sun_core_color: Vec3,

    // Water colors for distant water rendering
    /// Deep water color during day
    pub water_deep_day: Vec3,
    /// Deep water color at night
    pub water_deep_night: Vec3,
    /// Shallow water color during day
    pub water_shallow_day: Vec3,
    /// Shallow water color at night
    pub water_shallow_night: Vec3,

    // Atmospheric scatter colors
    /// Warm scatter color (sunset/sunrise)
    pub scatter_warm: Vec3,
    /// Cool scatter color (night)
    pub scatter_cool: Vec3,
}

impl Default for SkyParams {
    fn default() -> Self {
        Self {
            horizon_day: sky::HORIZON_DAY,
            zenith_day: sky::ZENITH_DAY,
            horizon_night: sky::HORIZON_NIGHT,
            zenith_night: sky::ZENITH_NIGHT,
            sun_size: 0.03,
            sun_glow_intensity: 0.5,
            sun_glow_falloff: 2.0,
            sun_direction: Vec3::new(0.3, 0.8, 0.2).normalize(),
            sun_color: sun::COLOR_ZENITH,
            sun_intensity: 1.0,
            sun_core_color: sun::COLOR_CORE,
            water_deep_day: water::DEEP_DAY,
            water_deep_night: water::DEEP_NIGHT,
            water_shallow_day: water::SHALLOW_DAY,
            water_shallow_night: water::SHALLOW_NIGHT,
            scatter_warm: scatter::HORIZON_WARM,
            scatter_cool: scatter::HORIZON_COOL,
        }
    }
}

/// Material for the procedural sky dome
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct SkyMaterial {
    #[uniform(0)]
    pub params: SkyParams,
}

impl Material for SkyMaterial {
    fn fragment_shader() -> ShaderRef {
        SKY_SHADER_PATH.into()
    }

    fn vertex_shader() -> ShaderRef {
        SKY_SHADER_PATH.into()
    }
}

/// Marker component for the sky dome entity
#[derive(Component)]
pub struct SkyDome;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        let strip_prefix = "src/";
        embedded_asset!(app, strip_prefix, "./shaders/sky_shader.wgsl");

        app.init_resource::<SkyParams>();
        app.add_plugins(MaterialPlugin::<SkyMaterial>::default());
        app.add_systems(Startup, spawn_sky_dome);
        app.add_systems(Update, sync_sky_params);
    }
}

/// Spawn the sky dome as a large inverted sphere
fn spawn_sky_dome(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SkyMaterial>>,
    sky_params: Res<SkyParams>,
) {
    // Create a large sphere mesh for the sky dome
    let mut sky_mesh = Sphere::new(10000.0).mesh().uv(64, 32);

    // Invert winding so we can see the mesh from inside
    sky_mesh.invert_winding().unwrap();

    commands.spawn((
        Mesh3d(meshes.add(sky_mesh)),
        MeshMaterial3d(materials.add(SkyMaterial {
            params: *sky_params,
        })),
        Transform::from_translation(Vec3::ZERO),
        SkyDome,
    ));
}

/// System to sync SkyParams resource to sky materials
fn sync_sky_params(
    sky_params: Res<SkyParams>,
    mut materials: ResMut<Assets<SkyMaterial>>,
    query: Query<&MeshMaterial3d<SkyMaterial>, With<SkyDome>>,
) {
    if !sky_params.is_changed() {
        return;
    }

    for material_handle in query.iter() {
        if let Some(material) = materials.get_mut(material_handle) {
            material.params = *sky_params;
        }
    }
}
