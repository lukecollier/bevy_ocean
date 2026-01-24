use bevy::prelude::*;

use crate::cloud_plugin::CloudParams;
use crate::colors::{fog, sky, sun};
use crate::ocean_plugin::OceanParams;
use crate::sky_plugin::SkyParams;

/// Marker component for the sun directional light.
#[derive(Component)]
pub struct SunLight;

/// Configuration for the day/night cycle.
#[derive(Resource, Clone, Copy)]
pub struct DayNightConfig {
    /// Full day/night cycle period in seconds.
    pub cycle_period: f32,
    /// Maximum illuminance of the sun light.
    pub max_illuminance: f32,
}

impl Default for DayNightConfig {
    fn default() -> Self {
        Self {
            cycle_period: 60.0,
            max_illuminance: 1000.0,
        }
    }
}

/// Plugin that manages the day/night cycle, synchronizing sun direction,
/// color, and intensity across ocean, sky, cloud, and directional light.
pub struct DayNightCyclePlugin {
    pub config: DayNightConfig,
}

impl Default for DayNightCyclePlugin {
    fn default() -> Self {
        Self {
            config: DayNightConfig::default(),
        }
    }
}

impl DayNightCyclePlugin {
    pub fn new(cycle_period: f32) -> Self {
        Self {
            config: DayNightConfig {
                cycle_period,
                ..default()
            },
        }
    }
}

impl Plugin for DayNightCyclePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config);
        app.add_systems(Startup, spawn_sun_light);
        app.add_systems(Update, day_night_cycle);
    }
}

fn spawn_sun_light(mut commands: Commands, config: Res<DayNightConfig>) {
    commands.spawn((
        DirectionalLight {
            illuminance: config.max_illuminance,
            shadows_enabled: true,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(0.2, -0.8, -0.5), Vec3::Y),
        SunLight,
    ));
}

fn day_night_cycle(
    config: Res<DayNightConfig>,
    ocean: Option<ResMut<OceanParams>>,
    sky: Option<ResMut<SkyParams>>,
    cloud: Option<ResMut<CloudParams>>,
    mut sun_light_query: Query<(&mut DirectionalLight, &mut Transform), With<SunLight>>,
    time: Res<Time>,
) {
    let angle = (time.elapsed_secs() / config.cycle_period) * std::f32::consts::TAU;

    // Rotate sun around the X axis (east-west arc across the sky)
    // Y = sin(angle) gives height (-1 to 1)
    // Z = cos(angle) gives depth
    let direction = Vec3::new(
        0.2,         // Slight offset so it's not perfectly overhead
        angle.sin(), // Height: goes from -1 (below) to 1 (zenith)
        angle.cos(), // Depth: creates the arc
    )
    .normalize();

    // Sun intensity based on height (0 when below horizon, 1 at zenith)
    let intensity = direction.y.max(0.0);

    // Sun color: warm at horizon, white at zenith
    let horizon_factor = 1.0 - direction.y.abs().sqrt();
    let color = Vec3::lerp(sun::COLOR_ZENITH, sun::COLOR_HORIZON, horizon_factor);

    // Update ocean plugin if present
    if let Some(mut ocean) = ocean {
        ocean.sun_direction = direction;
        ocean.fog_color = Vec3::lerp(fog::COLOR_NIGHT, fog::COLOR_DAY, intensity);
    }

    // Update sky plugin if present
    if let Some(mut sky) = sky {
        sky.sun_direction = direction;
        sky.sun_color = color;
        sky.sun_intensity = intensity;
    }

    // Update cloud plugin if present
    if let Some(mut cloud) = cloud {
        cloud.sun_direction = direction;
        cloud.sun_color = color;
        cloud.sun_intensity = intensity;
        cloud.time = time.elapsed_secs();
    }

    // Update directional light to match sun
    for (mut light, mut transform) in sun_light_query.iter_mut() {
        // Light direction is opposite to sun direction (light points toward the scene)
        *transform = Transform::default().looking_to(-direction, Vec3::Y);

        // Blend light color between day and night sky colors
        let light_color = Vec3::lerp(sky::HORIZON_NIGHT, color, intensity);
        light.color = Color::linear_rgb(light_color.x, light_color.y, light_color.z);

        // Scale illuminance based on sun intensity (dimmer at night/horizon)
        light.illuminance = config.max_illuminance * intensity;
    }
}
