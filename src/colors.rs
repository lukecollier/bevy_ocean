//! Centralized day/night color definitions for the ocean scene.
//!
//! All atmospheric and water colors are defined here to ensure consistency
//! across sky, ocean, clouds, and fog systems.

use bevy::prelude::Vec3;

/// Sky colors
pub mod sky {
    use super::*;

    /// Horizon color during daytime (soft sky blue)
    pub const HORIZON_DAY: Vec3 = Vec3::new(0.5, 0.7, 0.9);
    /// Zenith (overhead) color during daytime (deep blue)
    pub const ZENITH_DAY: Vec3 = Vec3::new(0.2, 0.4, 0.8);
    /// Horizon color at night (dark blue-gray)
    pub const HORIZON_NIGHT: Vec3 = Vec3::new(0.05, 0.05, 0.1);
    /// Zenith color at night (near black)
    pub const ZENITH_NIGHT: Vec3 = Vec3::new(0.01, 0.01, 0.03);
    /// Sky reflection on ocean during day
    pub const REFLECTION_DAY: Vec3 = Vec3::new(0.6, 0.75, 0.9);
    /// Sky reflection on ocean at night
    pub const REFLECTION_NIGHT: Vec3 = Vec3::new(0.05, 0.08, 0.15);
}

/// Sun colors
pub mod sun {
    use super::*;

    /// Sun color at zenith (white-ish)
    pub const COLOR_ZENITH: Vec3 = Vec3::new(1.0, 0.95, 0.9);
    /// Sun color at horizon (orange/red for sunset/sunrise)
    pub const COLOR_HORIZON: Vec3 = Vec3::new(1.0, 0.5, 0.2);
    /// Sun color for ocean specular (slightly warmer)
    pub const COLOR_OCEAN: Vec3 = Vec3::new(1.0, 0.95, 0.85);
    /// Sun core color (very bright white-yellow)
    pub const COLOR_CORE: Vec3 = Vec3::new(1.0, 1.0, 0.95);
}

/// Water colors (used in sky shader for distant water)
pub mod water {
    use super::*;

    /// Deep water color during day (sky shader distant water)
    pub const DEEP_DAY: Vec3 = Vec3::new(0.02, 0.08, 0.15);
    /// Deep water color at night (sky shader distant water)
    pub const DEEP_NIGHT: Vec3 = Vec3::new(0.01, 0.02, 0.04);
    /// Shallow/horizon water color during day (sky shader)
    pub const SHALLOW_DAY: Vec3 = Vec3::new(0.06, 0.18, 0.28);
    /// Shallow/horizon water color at night (sky shader)
    pub const SHALLOW_NIGHT: Vec3 = Vec3::new(0.03, 0.05, 0.08);
}

/// Ocean mesh colors (used in ocean shader for actual water mesh)
pub mod ocean {
    use super::*;

    /// Deep ocean color (looking straight down)
    pub const DEEP: Vec3 = Vec3::new(0.02, 0.05, 0.12);
    /// Shallow ocean color (at grazing angles)
    pub const SHALLOW: Vec3 = Vec3::new(0.05, 0.18, 0.28);
    /// Subsurface scattering color (turquoise glow through waves)
    pub const SSS: Vec3 = Vec3::new(0.1, 0.4, 0.35);
    /// Foam color (white)
    pub const FOAM: Vec3 = Vec3::new(1.0, 1.0, 1.0);
    /// Ambient light color
    pub const AMBIENT: Vec3 = Vec3::new(0.02, 0.03, 0.05);
}

/// Fog colors (matches horizon for seamless blending)
pub mod fog {
    use super::*;

    /// Fog color during day (matches sky horizon)
    pub const COLOR_DAY: Vec3 = sky::HORIZON_DAY;
    /// Fog color at night (matches sky horizon)
    pub const COLOR_NIGHT: Vec3 = sky::HORIZON_NIGHT;
}

/// Atmospheric scattering colors
pub mod scatter {
    use super::*;

    /// Warm scatter color for horizon during day (sunset/sunrise tint)
    pub const HORIZON_WARM: Vec3 = Vec3::new(1.0, 0.6, 0.3);
    /// Cool scatter color for horizon at night
    pub const HORIZON_COOL: Vec3 = Vec3::new(0.3, 0.2, 0.4);
}

/// Cloud colors
pub mod cloud {
    use super::*;

    /// Cloud base color (white)
    pub const BASE: Vec3 = Vec3::new(1.0, 1.0, 1.0);
    /// Cloud ambient color during day
    pub const AMBIENT_DAY: Vec3 = Vec3::new(0.8, 0.85, 0.9);
    /// Cloud ambient color at night
    pub const AMBIENT_NIGHT: Vec3 = Vec3::new(0.3, 0.35, 0.4);
}

/// Helper to interpolate between night and day colors
#[inline]
pub fn lerp_day_night(night: Vec3, day: Vec3, sun_intensity: f32) -> Vec3 {
    Vec3::lerp(night, day, sun_intensity)
}
