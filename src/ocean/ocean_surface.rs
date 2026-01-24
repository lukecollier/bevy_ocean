use bevy::render::render_resource::CommandEncoder;
use bevy::render::render_resource::Extent3d;
use bevy::render::render_resource::Texture;
use bevy::render::render_resource::TextureDescriptor;
use bevy::render::render_resource::TextureDimension;
use bevy::render::render_resource::TextureFormat;
use bevy::render::render_resource::TextureUsages;
use bevy::render::renderer::RenderDevice;
use bevy::render::renderer::RenderQueue;

use crate::ocean::ocean_parameters::OceanSpectrumParameters;
use crate::ocean::pipelines::FFT;
use crate::ocean::pipelines::FoamPersistencePipeline;
use crate::ocean::pipelines::GenerateMipmapsPipeline;
use crate::ocean::pipelines::InitialSpectrumPipeline;
use crate::ocean::pipelines::TimeDependentSpectrumPipeline;
use crate::ocean::pipelines::WavesDataMergePipeline;

pub struct OceanSurface {
    params: OceanSpectrumParameters,
    parameters_changed: bool,

    // pipelines
    initial_spectrum_pipeline: InitialSpectrumPipeline,
    time_dependent_spectrum_pipeline: TimeDependentSpectrumPipeline,
    fft: FFT,
    waves_data_merge_pipeline: WavesDataMergePipeline,
    foam_persistence_pipeline: FoamPersistencePipeline,
    // generate_mipmaps_pipeline: GenerateMipmapsPipeline,
}

impl OceanSurface {
    pub fn new(
        device: &RenderDevice,
        size: u32,
        params: OceanSpectrumParameters,
        // these 3 are actually going to be texture arrays
        displacement_texture: &Texture,
        derivatives_texture: &Texture,
        foam_persistence_texture: &Texture,
    ) -> OceanSurface {
        let texture_size = Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };

        let h0_texture = device.create_texture(&TextureDescriptor {
            label: Some("H0 texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let h0k_texture = device.create_texture(&TextureDescriptor {
            label: Some("H0k texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let waves_data_texture = device.create_texture(&TextureDescriptor {
            label: Some("Waves Data texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let amp_dx_dz_texture = device.create_texture(&TextureDescriptor {
            label: Some("Dx / Dz"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let amp_dyx_dyz_texture = device.create_texture(&TextureDescriptor {
            label: Some("Dyx / Dyz"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        let initial_spectrum_pipeline = InitialSpectrumPipeline::init(
            params,
            device,
            &h0k_texture,
            &waves_data_texture,
            &h0_texture,
        );

        let time_dependent_spectrum_pipeline = TimeDependentSpectrumPipeline::init(
            size,
            device,
            &h0_texture,
            &waves_data_texture,
            &amp_dx_dz_texture,
            &amp_dyx_dyz_texture,
        );

        let fft = FFT::init(size, device, &amp_dx_dz_texture, &amp_dyx_dyz_texture);

        let waves_data_merge_pipeline = WavesDataMergePipeline::init(
            device,
            size,
            params.delta,
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

        // let generate_mipmaps_pipeline =
        //     GenerateMipmapsPipeline::init(device, size, displacement_texture, derivatives_texture);

        OceanSurface {
            params,
            initial_spectrum_pipeline,
            time_dependent_spectrum_pipeline,
            fft,
            waves_data_merge_pipeline,
            foam_persistence_pipeline,
            // generate_mipmaps_pipeline,
            parameters_changed: false,
        }
    }

    pub fn init(&self, encoder: &mut CommandEncoder, queue: &RenderQueue) {
        self.initial_spectrum_pipeline.dispatch(encoder, &queue);
        self.fft.precompute(encoder);
    }

    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        queue: &RenderQueue,
        time: f32,
        dt: std::time::Duration,
    ) {
        if self.parameters_changed {
            self.initial_spectrum_pipeline.dispatch(encoder, queue);
        }

        self.time_dependent_spectrum_pipeline
            .dispatch(encoder, time);

        self.fft.dispatch(encoder);

        self.waves_data_merge_pipeline.dispatch(encoder, dt);
        self.foam_persistence_pipeline.dispatch(encoder, dt);
        // self.generate_mipmaps_pipeline.dispatch(encoder);
    }

    pub fn change_parameters(&mut self, parameters: OceanSpectrumParameters) {
        self.params = parameters;
        self.parameters_changed = true;
    }
}
