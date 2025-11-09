use bevy::asset::Asset;
use bevy::asset::AssetServer;
use bevy::asset::Handle;
use bevy::ecs::system::Res;
use bevy::image::Image;
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
use crate::ocean::pipelines::GenerateMipmapsPipeline;
use crate::ocean::pipelines::InitialSpectrumPipeline;
use crate::ocean::pipelines::TimeDependentSpectrumPipeline;
use crate::ocean::pipelines::WavesDataMergePipeline;

pub struct OceanSurface {
    params: OceanSpectrumParameters,
    parameters_changed: bool,

    h0_texture: Texture,
    h0k_texture: Texture,
    waves_data_texture: Texture,

    amp_dx_dz_texture: Texture,
    amp_dyx_dyz_texture: Texture,

    displacement_texture: Texture,
    derivatives_texture: Texture,

    // pipelines
    initial_spectrum_pipeline: InitialSpectrumPipeline,
    time_dependent_spectrum_pipeline: TimeDependentSpectrumPipeline,
    fft: FFT,
    waves_data_merge_pipeline: WavesDataMergePipeline,
    generate_mipmaps_pipeline: GenerateMipmapsPipeline,
}

impl OceanSurface {
    // todo: We should pass in the displacements here,these are the only two textures that bevy
    // will need to read
    pub fn new(device: &RenderDevice, size: u32, params: OceanSpectrumParameters) -> OceanSurface {
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

        let displacement_texture = device.create_texture(&TextureDescriptor {
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
        });

        let derivatives_texture = device.create_texture(&TextureDescriptor {
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
        });

        let initial_spectrum_pipeline = InitialSpectrumPipeline::init(
            size,
            params,
            &device,
            &h0k_texture,
            &waves_data_texture,
            &h0_texture,
        );

        let time_dependent_spectrum_pipeline = TimeDependentSpectrumPipeline::init(
            size,
            &device,
            &h0_texture,
            &waves_data_texture,
            &amp_dx_dz_texture,
            &amp_dyx_dyz_texture,
        );

        let fft = FFT::init(size, &device, &amp_dx_dz_texture, &amp_dyx_dyz_texture);

        let waves_data_merge_pipeline = WavesDataMergePipeline::init(
            &device,
            size,
            1.2,
            &amp_dx_dz_texture,
            &amp_dyx_dyz_texture,
            &displacement_texture,
            &derivatives_texture,
        );

        let generate_mipmaps_pipeline = GenerateMipmapsPipeline::init(
            &device,
            size,
            &displacement_texture,
            &derivatives_texture,
        );

        OceanSurface {
            h0k_texture,
            waves_data_texture,
            h0_texture,
            amp_dx_dz_texture,
            amp_dyx_dyz_texture,
            displacement_texture,
            derivatives_texture,

            params,
            initial_spectrum_pipeline,
            time_dependent_spectrum_pipeline,
            fft,
            waves_data_merge_pipeline,
            generate_mipmaps_pipeline,
            parameters_changed: false,
        }
    }

    pub fn init(&self, encoder: &mut CommandEncoder, queue: &RenderQueue) {
        self.initial_spectrum_pipeline.dispatch(encoder, &queue);
        self.fft.precompute(encoder);
    }

    pub fn dispatch(
        &mut self,
        encoder: &mut CommandEncoder,
        queue: &RenderQueue,
        time: f32,
        dt: std::time::Duration,
    ) {
        if self.parameters_changed {
            self.initial_spectrum_pipeline.dispatch(encoder, queue);
            self.parameters_changed = false;
        }

        self.time_dependent_spectrum_pipeline
            .dispatch(encoder, time + 10000.0);

        self.fft.dispatch(encoder);

        self.waves_data_merge_pipeline.dispatch(encoder, dt);
        self.generate_mipmaps_pipeline.dispatch(encoder);
    }

    // todo: These should come from the plugin as a Handle<Image>
    pub fn displacement_texture(&self) -> &Texture {
        &self.displacement_texture
    }

    pub fn derivatives_texture(&self) -> &Texture {
        &self.derivatives_texture
    }

    pub fn change_parameters(&mut self, parameters: OceanSpectrumParameters) {
        self.params = parameters;
        self.parameters_changed = true;
    }
}
