use bevy::dev_tools::fps_overlay::FpsOverlayPlugin;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::feathers::{
    FeathersPlugins,
    controls::{SliderProps, slider},
    dark_theme::create_dark_theme,
    theme::{ThemeBackgroundColor, ThemedText, UiTheme},
    tokens,
};
use bevy::prelude::*;
use bevy::ui_widgets::{SliderPrecision, SliderStep, ValueChange, observe, slider_self_update};
use bevy_flycam::PlayerPlugin;
use bevy_rand::{plugin::EntropyPlugin, prelude::WyRand};

use bevy_ocean::cloud_plugin::CloudPlugin;
use bevy_ocean::day_night_plugin::DayNightCyclePlugin;
use bevy_ocean::ocean_plugin::{OceanParams, OceanPlugin};
use bevy_ocean::sky_plugin::SkyPlugin;

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlayerPlugin)
        .add_plugins(EntropyPlugin::<WyRand>::default())
        .add_plugins(OceanPlugin::default())
        .add_plugins(SkyPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(FpsOverlayPlugin::default())
        .add_plugins(CloudPlugin)
        .add_plugins(DayNightCyclePlugin::new(60.0 * 60.0))
        .insert_resource(UiTheme(create_dark_theme()))
        .add_plugins(FeathersPlugins)
        .add_systems(Startup, startup_ui)
        .run()
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
