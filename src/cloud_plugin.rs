use bevy::{asset::embedded_asset, prelude::*, render::render_resource::*, shader::ShaderRef};

use crate::colors::{cloud, sun};

const CLOUD_SHADER_PATH: &str = "embedded://bevy_ocean/shaders/cloud_shader.wgsl";

pub struct CloudPlugin;

/// Cloud rendering parameters that can be modified at runtime.
#[derive(Resource, Clone, Copy, Debug, ShaderType)]
pub struct CloudParams {
    // Cloud settings
    /// Cloud coverage (0.0-1.0, default: 0.5)
    pub coverage: f32,
    /// Cloud density (0.0-1.0, default: 0.8)
    pub density: f32,
    /// Cloud scroll speed (UV units per second, default: (0.01, 0.005))
    pub scroll_speed: Vec2,
    /// UV scale for cloud noise (default: 0.001)
    pub scale: f32,
    /// Edge softness for cloud blending (default: 0.3)
    pub softness: f32,
    /// Cloud layer altitude (default: 500.0)
    pub altitude: f32,
    /// Animation time (updated each frame)
    pub time: f32,

    // Sun state (updated by day_night_cycle)
    /// Normalized sun direction vector
    pub sun_direction: Vec3,
    /// Sun color (RGB)
    pub sun_color: Vec3,
    /// Sun intensity (0.0 at night, 1.0+ during day)
    pub sun_intensity: f32,

    // Cloud colors
    /// Cloud base color (white)
    pub cloud_base: Vec3,
    /// Cloud ambient color during day
    pub cloud_ambient_day: Vec3,
    /// Cloud ambient color at night
    pub cloud_ambient_night: Vec3,
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            coverage: 0.5,
            density: 0.8,
            scroll_speed: Vec2::new(0.01, 0.005),
            scale: 0.001,
            softness: 0.3,
            altitude: 500.0,
            time: 0.0,
            sun_direction: Vec3::new(0.3, 0.8, 0.2).normalize(),
            sun_color: sun::COLOR_ZENITH,
            sun_intensity: 1.0,
            cloud_base: cloud::BASE,
            cloud_ambient_day: cloud::AMBIENT_DAY,
            cloud_ambient_night: cloud::AMBIENT_NIGHT,
        }
    }
}

/// Material for the procedural cloud dome
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CloudMaterial {
    #[uniform(0)]
    pub params: CloudParams,
}

impl Material for CloudMaterial {
    fn fragment_shader() -> ShaderRef {
        CLOUD_SHADER_PATH.into()
    }

    fn vertex_shader() -> ShaderRef {
        CLOUD_SHADER_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

/// Marker component for the cloud dome entity
#[derive(Component)]
pub struct CloudDome;

impl Plugin for CloudPlugin {
    fn build(&self, app: &mut App) {
        let strip_prefix = "src/";
        embedded_asset!(app, strip_prefix, "./shaders/cloud_shader.wgsl");

        app.init_resource::<CloudParams>();
        app.add_plugins(MaterialPlugin::<CloudMaterial>::default());
        app.add_systems(Startup, spawn_cloud_dome);
        app.add_systems(Update, sync_cloud_params);
    }
}

/// Spawn the cloud dome as a slightly smaller inverted sphere (inside sky dome)
fn spawn_cloud_dome(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CloudMaterial>>,
    cloud_params: Res<CloudParams>,
) {
    // Create a sphere slightly smaller than the sky dome
    let mut cloud_mesh = Sphere::new(9900.0).mesh().uv(64, 32);

    // Invert winding so we can see the mesh from inside
    cloud_mesh.invert_winding().unwrap();

    commands.spawn((
        Mesh3d(meshes.add(cloud_mesh)),
        MeshMaterial3d(materials.add(CloudMaterial {
            params: *cloud_params,
        })),
        Transform::from_translation(Vec3::ZERO),
        CloudDome,
    ));
}

/// System to sync CloudParams resource to cloud materials
fn sync_cloud_params(
    cloud_params: Res<CloudParams>,
    mut materials: ResMut<Assets<CloudMaterial>>,
    query: Query<&MeshMaterial3d<CloudMaterial>, With<CloudDome>>,
) {
    if !cloud_params.is_changed() {
        return;
    }

    for material_handle in query.iter() {
        if let Some(material) = materials.get_mut(material_handle) {
            material.params = *cloud_params;
        }
    }
}
