#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view
}

// Cloud parameters uniform
struct CloudParamsUniform {
    coverage: f32,
    density: f32,
    scroll_speed: vec2<f32>,
    scale: f32,
    softness: f32,
    altitude: f32,
    time: f32,
    sun_direction: vec3<f32>,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    // Cloud colors
    cloud_base: vec3<f32>,
    cloud_ambient_day: vec3<f32>,
    cloud_ambient_night: vec3<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var<uniform> params: CloudParamsUniform;

struct CloudVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) view_ray: vec3<f32>,
}

@vertex
fn vertex(in: Vertex) -> CloudVertexOutput {
    var model = mesh_functions::get_world_from_local(in.instance_index);

    // Position the cloud dome centered on the camera
    let camera_pos = view.world_position;
    let local_pos = vec4<f32>(in.position, 1.0);
    var world_pos = mesh_functions::mesh_position_local_to_world(model, local_pos);

    // Keep cloud dome centered on camera

    world_pos = vec4<f32>(world_pos.xyz + camera_pos, world_pos.w);

    var out: CloudVertexOutput;
    out.position = position_world_to_clip(world_pos.xyz);
    out.world_position = world_pos;
    out.view_ray = normalize(world_pos.xyz - camera_pos);

    return out;
}

// Hash function for noise
fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// Value noise with smooth interpolation
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Smooth interpolation (smoothstep)
    let u = f * f * (3.0 - 2.0 * f);

    // Sample 4 corners
    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));

    // Bilinear interpolation
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractional Brownian Motion (layered noise)
fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var total_amplitude = 0.0;

    // 3 octaves for clouds
    for (var i = 0; i < 3; i = i + 1) {
        value = value + noise(p * frequency) * amplitude;
        total_amplitude = total_amplitude + amplitude;
        amplitude = amplitude * 0.5;
        frequency = frequency * 2.0;
    }

    return value / total_amplitude;
}

@fragment
fn fragment(mesh: CloudVertexOutput) -> @location(0) vec4<f32> {
    let ray = normalize(mesh.view_ray);

    // Early out for below-horizon rays (no clouds there)
    if (ray.y < 0.05) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Calculate intersection with cloud plane at altitude
    // ray starts at camera (origin for our purposes), intersects plane y = altitude
    // t = altitude / ray.y (time along ray to reach cloud plane)
    let t = params.altitude / ray.y;

    // Get intersection point in world XZ coordinates
    let cloud_pos = ray.xz * t;

    // Apply UV scale and scrolling animation
    let scroll_offset = params.scroll_speed * params.time;
    let uv = cloud_pos * params.scale + scroll_offset;

    // Sample FBM noise for cloud shape
    let cloud_noise = fbm(uv);

    // Apply coverage threshold with soft edges
    // Higher coverage = more clouds visible
    let threshold = 1.0 - params.coverage;
    let cloud_shape = smoothstep(threshold - params.softness, threshold + params.softness, cloud_noise);

    // Skip fully transparent pixels
    if (cloud_shape < 0.01) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Cloud base color
    var cloud_color = params.cloud_base;

    // Sun direction
    let sun_dir = normalize(params.sun_direction);

    // Lighting: clouds facing sun are brighter
    // Use a simple directional light approximation
    let sun_factor = dot(ray, sun_dir) * 0.5 + 0.5;  // Remap to 0-1

    // Add subtle warm tint from sun
    let sun_tint = params.sun_color * sun_factor * params.sun_intensity;
    cloud_color = mix(cloud_color * 0.7, cloud_color, sun_factor);
    cloud_color = cloud_color + sun_tint * 0.2;

    // Ambient sky color influence (darker at night)
    let ambient = mix(params.cloud_ambient_night, params.cloud_ambient_day, params.sun_intensity);
    cloud_color = cloud_color * ambient;

    // Edge glow when sun is behind clouds (silver lining effect)
    let back_light = saturate(dot(-ray, sun_dir));
    let edge_glow = pow(back_light, 4.0) * (1.0 - cloud_shape) * params.sun_intensity * 0.5;
    cloud_color = cloud_color + params.sun_color * edge_glow;

    // Final alpha based on density and shape
    let alpha = cloud_shape * params.density;

    // Fade clouds near horizon to prevent hard cutoff
    let horizon_fade = smoothstep(0.05, 0.15, ray.y);

    return vec4<f32>(cloud_color, alpha * horizon_fade);
}
