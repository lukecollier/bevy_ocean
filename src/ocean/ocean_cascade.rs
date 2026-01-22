use bevy::render::render_resource::CommandEncoder;
use bevy::render::render_resource::Texture;
use bevy::render::renderer::RenderDevice;
use bevy::render::renderer::RenderQueue;

use crate::ocean::OceanSpectrumParameters;
use crate::ocean::OceanSurface;

pub struct OceanCascade {
    pub cascade_0: OceanSurface,
    pub cascade_1: OceanSurface,
    pub cascade_2: OceanSurface,
}

#[derive(Clone, Copy)]
pub struct OceanCascadeParameters {
    pub size: u32,
    pub wind_speed: f32,
    pub wind_direction: f32,
    pub swell: f32,
}

impl OceanCascade {
    pub fn new(
        device: &RenderDevice,
        size: u32,
        params: OceanCascadeParameters,
        displacement_0: &Texture,
        displacement_1: &Texture,
        displacement_2: &Texture,
        derivatives_0: &Texture,
        derivatives_1: &Texture,
        derivatives_2: &Texture,
        foam_persistence_0: &Texture,
        foam_persistence_1: &Texture,
        foam_persistence_2: &Texture,
    ) -> Self {
        let surface_params = OceanSpectrumParameters {
            size: params.size,
            wind_speed: params.wind_speed,
            wind_direction: params.wind_direction,
            swell: params.swell,
            ..Default::default()
        };

        let length_scale_0 = 500.0;
        let length_scale_1 = 85.0;
        let length_scale_2 = 10.0;

        let boundary_1 = 2.0 * std::f32::consts::PI / length_scale_1 * 6.0;
        let boundary_2 = 2.0 * std::f32::consts::PI / length_scale_2 * 6.0;

        let params_0 = OceanSpectrumParameters {
            cut_off_low: 0.0001,
            cut_off_high: boundary_1,
            length_scale: length_scale_0,
            ..surface_params
        };

        let params_1 = OceanSpectrumParameters {
            cut_off_low: boundary_1,
            cut_off_high: boundary_2,
            length_scale: length_scale_1,
            ..surface_params
        };

        let params_2 = OceanSpectrumParameters {
            cut_off_low: boundary_2,
            cut_off_high: 9999.0,
            length_scale: length_scale_2,
            ..surface_params
        };

        let cascade_0 = OceanSurface::new(
            device,
            size,
            params_0,
            displacement_0,
            derivatives_0,
            foam_persistence_0,
        );
        let cascade_1 = OceanSurface::new(
            device,
            size,
            params_1,
            displacement_1,
            derivatives_1,
            foam_persistence_1,
        );
        let cascade_2 = OceanSurface::new(
            device,
            size,
            params_2,
            displacement_2,
            derivatives_2,
            foam_persistence_2,
        );

        Self {
            cascade_0,
            cascade_1,
            cascade_2,
        }
    }

    pub fn init(&self, encoder: &mut CommandEncoder, queue: &RenderQueue) {
        self.cascade_0.init(encoder, queue);
        self.cascade_1.init(encoder, queue);
        self.cascade_2.init(encoder, queue);
    }

    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        queue: &RenderQueue,
        time: f32,
        dt: std::time::Duration,
    ) {
        self.cascade_0.dispatch(encoder, queue, time, dt);
        self.cascade_1.dispatch(encoder, queue, time, dt);
        // self.cascade_2.dispatch(encoder, queue, time, dt);
    }
}
