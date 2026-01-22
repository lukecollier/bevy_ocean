use bevy::core_pipeline::Skybox;
use bevy::prelude::*;
use bevy::render::render_resource::{TextureViewDescriptor, TextureViewDimension};
use bevy::ui_widgets::{SliderPrecision, SliderStep, observe, slider_self_update};
use bevy::{
    feathers::{
        FeathersPlugins,
        controls::{ButtonProps, ButtonVariant, SliderProps, button, slider},
        dark_theme::create_dark_theme,
        rounded_corners::RoundedCorners,
        theme::{ThemeBackgroundColor, ThemedText, UiTheme},
        tokens,
    },
    prelude::*,
    ui_widgets::ValueChange,
};
use bevy_flycam::{FlyCam, PlayerPlugin};
use bevy_rand::{plugin::EntropyPlugin, prelude::WyRand};

use bevy_ocean::ocean_plugin::{
    OceanImages, OceanMaterial, OceanParams, OceanPlugin, OceanSurface, setup,
};

#[derive(Resource, Default)]
struct LoadingSkybox {
    loading: bool,
    cubemap: Handle<Image>,
}

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlayerPlugin)
        .add_plugins(EntropyPlugin::<WyRand>::default())
        .add_plugins(OceanPlugin)
        .insert_resource(UiTheme(create_dark_theme()))
        .add_plugins(FeathersPlugins)
        .add_systems(Startup, startup_skybox)
        .add_systems(Startup, spawn_debug_textures.after(setup))
        .add_systems(Startup, (startup_ui))
        .add_systems(Update, update_skybox)
        .run()
}

fn startup_skybox(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(LoadingSkybox {
        loading: true,
        cubemap: asset_server.load("sky.png"),
    });
}
fn spawn_debug_textures(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OceanMaterial>>,
    ocean_images: Res<OceanImages>,
    ocean_params: Res<OceanParams>,
    asset_server: Res<AssetServer>,
) {
    // Create a large plane mesh for the ocean
    let ocean_mesh = Mesh::from(
        Plane3d::default()
            .mesh()
            .size(1024.0, 1024.0)
            .subdivisions(2048),
    );

    // Load foam texture
    let foam_texture: Handle<Image> = asset_server.load("embedded://bevy_ocean/textures/foam.png");

    // Spawn the ocean entity with the OceanMaterial
    // Using all 3 cascades for multi-scale wave detail
    commands.spawn((
        Mesh3d(meshes.add(ocean_mesh)),
        MeshMaterial3d(materials.add(OceanMaterial {
            // Cascade 0 - large scale (500m)
            t_displacement_0: ocean_images.displacement_0.clone(),
            t_derivatives_0: ocean_images.derivatives_0.clone(),
            // Cascade 1 - medium scale (85m)
            t_displacement_1: ocean_images.displacement_1.clone(),
            t_derivatives_1: ocean_images.derivatives_1.clone(),
            // Cascade 2 - small scale (10m)
            t_displacement_2: ocean_images.displacement_2.clone(),
            t_derivatives_2: ocean_images.derivatives_2.clone(),
            // Foam texture
            t_foam: foam_texture.clone(),
            // Ocean parameters
            params: *ocean_params,
            // Foam persistence textures
            t_foam_persistence_0: ocean_images.foam_persistence_0.clone(),
            t_foam_persistence_1: ocean_images.foam_persistence_1.clone(),
            t_foam_persistence_2: ocean_images.foam_persistence_2.clone(),
        })),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        OceanSurface,
    ));
}

fn update_skybox(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut loading_skybox: ResMut<LoadingSkybox>,
    camera_q: Single<Entity, With<FlyCam>>,
    mut images: ResMut<Assets<Image>>,
) {
    if !asset_server.is_loaded(&loading_skybox.cubemap) || loading_skybox.loading == false {
        return;
    }
    loading_skybox.loading = false;
    let image = images.get_mut(&loading_skybox.cubemap).unwrap();
    image.reinterpret_stacked_2d_as_array(6);
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });

    commands.entity(camera_q.into_inner()).insert(Skybox {
        image: loading_skybox.cubemap.clone(),
        brightness: 1000.,
        ..Default::default()
    });
}

fn startup_ui(mut commands: Commands, params: ResMut<OceanParams>) {
    commands.spawn((
        Node {
            padding: UiRect::px(6., 6., 6., 6.),
            width: percent(20),
            height: Val::Auto,
            align_items: AlignItems::Start,
            justify_content: JustifyContent::Start,
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            ..default()
        },
        ThemeBackgroundColor(tokens::WINDOW_BG),
        children![
            app_slider(
                "ds",
                params.displacement_scale,
                0.,
                2.,
                2,
                |ref mut params, value| { params.displacement_scale = value }
            ),
            app_slider(
                "ns",
                params.normal_strength,
                0.,
                2.,
                2,
                |ref mut params, value| { params.normal_strength = value }
            ),
            app_slider(
                "ft",
                params.foam_threshold,
                0.,
                2.,
                2,
                |ref mut params, value| { params.foam_threshold = value }
            ),
            app_slider(
                "fm",
                params.foam_multiplier,
                0.,
                2.,
                2,
                |ref mut params, value| { params.foam_multiplier = value }
            ),
            app_slider(
                "fts",
                params.foam_tile_scale,
                0.,
                20.,
                1,
                |ref mut params, value| { params.foam_tile_scale = value }
            ),
            app_slider("r", params.roughness, 0., 1., 1, |ref mut params, value| {
                params.roughness = value
            }),
            app_slider(
                "li",
                params.light_intensity,
                0.,
                10.,
                1,
                |ref mut params, value| { params.light_intensity = value }
            ),
            app_slider(
                "si",
                params.sss_intensity,
                0.,
                1.,
                1,
                |ref mut params, value| { params.sss_intensity = value }
            ),
        ],
    ));
}

fn make_slider(
    op: fn(ResMut<OceanParams>, f32) -> (),
) -> impl FnMut(On<ValueChange<f32>>, ResMut<OceanParams>) -> () {
    move |value_change, controls| {
        op(controls, value_change.value);
    }
}

fn app_slider(
    label: &str,
    value: f32,
    min: f32,
    max: f32,
    precision: i32,
    op: fn(ResMut<OceanParams>, f32) -> (),
) -> impl Bundle {
    (
        Node {
            width: Val::Percent(100.),
            height: Val::Auto,
            align_items: AlignItems::Start,
            justify_content: JustifyContent::Start,
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            row_gap: px(10),
            ..default()
        },
        children![
            (Text::new(label), ThemedText),
            (
                // Node {
                //     width: Val::Percent(100.),
                //     ..default()
                // },
                slider(
                    SliderProps { min, max, value },
                    (SliderStep(1.), SliderPrecision(precision)),
                ),
                observe(slider_self_update),
                observe(make_slider(op)) // Spawn((Text::new("Normal"), ThemedText)),
            )
        ],
    )
}
