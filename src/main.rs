use bevy::prelude::*;
use bevy_flycam::PlayerPlugin;
use bevy_rand::{plugin::EntropyPlugin, prelude::WyRand};

use bevy_ocean::ocean_plugin::OceanPlugin;

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PlayerPlugin)
        .add_plugins(EntropyPlugin::<WyRand>::default())
        .add_plugins(OceanPlugin)
        .run()
}
