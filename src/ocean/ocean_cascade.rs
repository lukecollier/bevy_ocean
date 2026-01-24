use bevy::render::render_resource::CommandEncoder;
use bevy::render::render_resource::Texture;
use bevy::render::renderer::RenderDevice;
use bevy::render::renderer::RenderQueue;

use crate::ocean::OceanSpectrumParameters;
use crate::ocean::OceanSurface;

pub struct OceanCascade<const N: usize> {
    pub cascades: [OceanSurface; N],
}

#[derive(Clone, Copy)]
pub struct OceanCascadeParameters {
    pub size: u32,
    pub wind_speed: f32,
    pub wind_direction: f32,
    pub swell: f32,
    /// Choppiness/lambda - horizontal displacement intensity (0.0-1.0, default 0.8)
    pub choppiness: f32,
    /// Water depth in meters - affects wave dispersion in shallow water (default 500.0)
    pub depth: f32,
}

pub struct OceanCascadeData<'a> {
    pub displacement: &'a Texture,
    pub derivatives: &'a Texture,
    pub foam_persistence: &'a Texture,
    pub length_scale: f32,
}

impl<const N: usize> OceanCascade<N> {
    pub fn new(
        device: &RenderDevice,
        size: u32,
        params: OceanCascadeParameters,
        cascade_data: [OceanCascadeData; N],
    ) -> Self {
        assert!(N > 1);
        let surface_params = OceanSpectrumParameters {
            size: params.size,
            wind_speed: params.wind_speed,
            wind_direction: params.wind_direction,
            swell: params.swell,
            delta: params.choppiness,
            depth: params.depth,
            ..Default::default()
        };
        let cascades: [OceanSurface; N] = std::array::from_fn(|idx| {
            let OceanCascadeData {
                displacement,
                derivatives,
                foam_persistence,
                length_scale,
            } = cascade_data[idx];
            let params = if idx == 0 {
                // first cascade
                let next_length_scale = cascade_data[idx + 1].length_scale;
                let boundary_low = 0.0001;
                let boundary_high = 2.0 * std::f32::consts::PI / next_length_scale * 6.0;
                OceanSpectrumParameters {
                    cut_off_low: boundary_low,
                    cut_off_high: boundary_high,
                    length_scale: length_scale,
                    ..surface_params
                }
            } else if idx == N - 1 {
                // last cascade
                let prev_length_scale = cascade_data[idx - 1].length_scale;
                let boundary_low = 2.0 * std::f32::consts::PI / prev_length_scale * 6.0;
                let boundary_high = 9999.0;
                OceanSpectrumParameters {
                    cut_off_low: boundary_low,
                    cut_off_high: boundary_high,
                    length_scale: length_scale,
                    ..surface_params
                }
            } else {
                // inbetween cascade
                let next_length_scale = cascade_data[idx + 1].length_scale;
                let boundary_low = 2.0 * std::f32::consts::PI / length_scale * 6.0;
                let boundary_high = 2.0 * std::f32::consts::PI / next_length_scale * 6.0;
                OceanSpectrumParameters {
                    cut_off_low: boundary_low,
                    cut_off_high: boundary_high,
                    length_scale: length_scale,
                    ..surface_params
                }
            };
            OceanSurface::new(
                device,
                size,
                params,
                displacement,
                derivatives,
                foam_persistence,
            )
        });
        Self { cascades }
    }

    pub fn init(&self, encoder: &mut CommandEncoder, queue: &RenderQueue) {
        for surface in &self.cascades {
            surface.init(encoder, queue);
        }
    }

    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        queue: &RenderQueue,
        time: f32,
        dt: std::time::Duration,
    ) {
        for surface in &self.cascades {
            surface.dispatch(encoder, queue, time, dt);
        }
    }
}
