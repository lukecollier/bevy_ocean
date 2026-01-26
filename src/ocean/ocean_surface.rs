use bevy::render::render_resource::CommandEncoder;
use bevy::render::render_resource::Extent3d;
use bevy::render::render_resource::Texture;
use bevy::render::render_resource::TextureDescriptor;
use bevy::render::render_resource::TextureDimension;
use bevy::render::render_resource::TextureFormat;
use bevy::render::render_resource::TextureUsages;
use bevy::render::renderer::RenderDevice;
use bevy::render::renderer::RenderQueue;

use crate::ocean::pipelines::FFT;
use crate::ocean::pipelines::FoamPersistencePipeline;
use crate::ocean::pipelines::InitialSpectrumPipeline;
use crate::ocean::pipelines::TimeDependentSpectrumPipeline;
use crate::ocean::pipelines::WavesDataMergePipeline;
use crate::ocean::OceanSpectrumParameters;

/// Internal struct for per-cascade initialization
struct CascadeInitializer {
    params: OceanSpectrumParameters,
    parameters_changed: bool,
    initial_spectrum_pipeline: InitialSpectrumPipeline,
}

impl CascadeInitializer {
    fn new(
        device: &RenderDevice,
        layer: u32,
        params: OceanSpectrumParameters,
        h0_texture: &Texture,
        waves_data_texture: &Texture,
        h0k_texture: &Texture,
    ) -> Self {
        let initial_spectrum_pipeline = InitialSpectrumPipeline::init(
            params,
            layer,
            device,
            h0k_texture,
            waves_data_texture,
            h0_texture,
        );

        Self {
            params,
            initial_spectrum_pipeline,
            parameters_changed: false,
        }
    }

    fn init(&self, encoder: &mut CommandEncoder, queue: &RenderQueue) {
        self.initial_spectrum_pipeline.dispatch(encoder, queue);
    }

    fn reinit_if_changed(&self, encoder: &mut CommandEncoder, queue: &RenderQueue) {
        if self.parameters_changed {
            self.initial_spectrum_pipeline.dispatch(encoder, queue);
        }
    }

    #[allow(dead_code)]
    fn change_parameters(&mut self, parameters: OceanSpectrumParameters) {
        self.params = parameters;
        self.parameters_changed = true;
    }
}

/// Ocean surface simulation with multiple cascades.
///
/// Manages FFT-based ocean wave simulation using texture arrays to process
/// all cascades in a single dispatch call per pipeline stage.
pub struct OceanSurface<const N: usize> {
    /// Per-cascade initialization (different parameters per cascade)
    cascades: [CascadeInitializer; N],

    /// Per-frame pipelines (process all cascades at once)
    time_dependent_spectrum_pipeline: TimeDependentSpectrumPipeline,
    fft: FFT,
    pub waves_data_merge_pipeline: WavesDataMergePipeline,
    pub foam_persistence_pipeline: FoamPersistencePipeline,
}

#[derive(Clone, Copy)]
pub struct OceanSurfaceParameters {
    pub size: u32,
    pub wind_speed: f32,
    pub wind_direction: f32,
    pub swell: f32,
    /// Choppiness/lambda - horizontal displacement intensity (0.0-1.0, default 0.8)
    pub choppiness: f32,
    /// Water depth in meters - affects wave dispersion in shallow water (default 500.0)
    pub depth: f32,
}

pub struct OceanSurfaceCascadeData<'a> {
    pub displacement: &'a Texture,
    pub derivatives: &'a Texture,
    pub foam_persistence: &'a Texture,
    pub length_scale: f32,
}

impl<const N: usize> OceanSurface<N> {
    pub fn new(
        device: &RenderDevice,
        size: u32,
        params: OceanSurfaceParameters,
        cascade_data: [OceanSurfaceCascadeData; N],
    ) -> Self {
        assert!(N > 1);
        let layers = N as u32;

        // Create shared intermediate texture arrays
        let texture_size = Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        };

        let h0_texture = device.create_texture(&TextureDescriptor {
            label: Some("H0 texture array"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let h0k_texture = device.create_texture(&TextureDescriptor {
            label: Some("H0k texture array"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let waves_data_texture = device.create_texture(&TextureDescriptor {
            label: Some("Waves Data texture array"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let amp_dx_dz_texture = device.create_texture(&TextureDescriptor {
            label: Some("Dx / Dz texture array"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let amp_dyx_dyz_texture = device.create_texture(&TextureDescriptor {
            label: Some("Dyx / Dyz texture array"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let surface_params = OceanSpectrumParameters {
            size: params.size,
            wind_speed: params.wind_speed,
            wind_direction: params.wind_direction,
            swell: params.swell,
            delta: params.choppiness,
            depth: params.depth,
            ..Default::default()
        };

        // Create per-cascade initializers (for initialization with different parameters)
        let cascades: [CascadeInitializer; N] = std::array::from_fn(|idx| {
            let OceanSurfaceCascadeData {
                length_scale, ..
            } = cascade_data[idx];
            let params = if idx == 0 {
                // first cascade
                let next_length_scale = cascade_data[idx + 1].length_scale;
                let boundary_low = 0.0001;
                let boundary_high = 2.0 * std::f32::consts::PI / next_length_scale * 6.0;
                OceanSpectrumParameters {
                    cut_off_low: boundary_low,
                    cut_off_high: boundary_high,
                    length_scale,
                    ..surface_params
                }
            } else if idx == N - 1 {
                // last cascade - boundary_low matches previous cascade's boundary_high
                let boundary_low = 2.0 * std::f32::consts::PI / length_scale * 6.0;
                let boundary_high = 9999.0;
                OceanSpectrumParameters {
                    cut_off_low: boundary_low,
                    cut_off_high: boundary_high,
                    length_scale,
                    ..surface_params
                }
            } else {
                // inbetween cascade - boundary_low matches previous cascade's boundary_high
                let next_length_scale = cascade_data[idx + 1].length_scale;
                let boundary_low = 2.0 * std::f32::consts::PI / length_scale * 6.0;
                let boundary_high = 2.0 * std::f32::consts::PI / next_length_scale * 6.0;
                OceanSpectrumParameters {
                    cut_off_low: boundary_low,
                    cut_off_high: boundary_high,
                    length_scale,
                    ..surface_params
                }
            };
            CascadeInitializer::new(
                device,
                idx as u32, // layer
                params,
                &h0_texture,
                &waves_data_texture,
                &h0k_texture,
            )
        });

        // Create per-frame pipelines (process all layers at once)
        let time_dependent_spectrum_pipeline = TimeDependentSpectrumPipeline::init(
            size,
            layers,
            device,
            &h0_texture,
            &waves_data_texture,
            &amp_dx_dz_texture,
            &amp_dyx_dyz_texture,
        );

        let fft = FFT::init(size, layers, device, &amp_dx_dz_texture, &amp_dyx_dyz_texture);

        // Use displacement texture from first cascade (they're all the same shared array)
        let displacement_texture = cascade_data[0].displacement;
        let derivatives_texture = cascade_data[0].derivatives;
        let foam_persistence_texture = cascade_data[0].foam_persistence;

        let waves_data_merge_pipeline = WavesDataMergePipeline::init(
            device,
            size,
            params.choppiness, // lambda/delta
            &amp_dx_dz_texture,
            &amp_dyx_dyz_texture,
            displacement_texture,
            derivatives_texture,
        );

        let foam_persistence_pipeline = FoamPersistencePipeline::init(
            device,
            size,
            displacement_texture,
            foam_persistence_texture,
        );

        Self {
            cascades,
            time_dependent_spectrum_pipeline,
            fft,
            waves_data_merge_pipeline,
            foam_persistence_pipeline,
        }
    }

    pub fn init(&self, encoder: &mut CommandEncoder, queue: &RenderQueue) {
        // Initialize spectrum for each cascade (writes to different layers)
        for cascade in &self.cascades {
            cascade.init(encoder, queue);
        }
        // Precompute FFT twiddle factors (same for all layers)
        self.fft.precompute(encoder);
    }

    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        queue: &RenderQueue,
        time: f32,
        dt: std::time::Duration,
    ) {
        // Reinit any cascades with changed parameters
        for cascade in &self.cascades {
            cascade.reinit_if_changed(encoder, queue);
        }

        // Run per-frame simulation (processes all layers at once)
        self.time_dependent_spectrum_pipeline.dispatch(encoder, time);
        self.fft.dispatch(encoder);
        self.waves_data_merge_pipeline.dispatch(encoder, dt);
        self.foam_persistence_pipeline.dispatch(encoder, dt);
    }
}
