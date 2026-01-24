#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view
}

// Sky parameters uniform
struct SkyParamsUniform {
    horizon_day: vec3<f32>,
    zenith_day: vec3<f32>,
    horizon_night: vec3<f32>,
    zenith_night: vec3<f32>,
    sun_size: f32,
    sun_glow_intensity: f32,
    sun_glow_falloff: f32,
    sun_direction: vec3<f32>,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    sun_core_color: vec3<f32>,
    // Water colors for distant water rendering
    water_deep_day: vec3<f32>,
    water_deep_night: vec3<f32>,
    water_shallow_day: vec3<f32>,
    water_shallow_night: vec3<f32>,
    // Atmospheric scatter colors
    scatter_warm: vec3<f32>,
    scatter_cool: vec3<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var<uniform> params: SkyParamsUniform;

struct SkyVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) view_ray: vec3<f32>,
}

@vertex
fn vertex(in: Vertex) -> SkyVertexOutput {
    var model = mesh_functions::get_world_from_local(in.instance_index);

    // Position the sky dome centered on the camera
    let camera_pos = view.world_position;
    let local_pos = vec4<f32>(in.position, 1.0);
    var world_pos = mesh_functions::mesh_position_local_to_world(model, local_pos);

    // Keep sky dome centered on camera (follows camera movement)
    world_pos = vec4<f32>(world_pos.xyz + camera_pos, world_pos.w);

    var out: SkyVertexOutput;
    out.position = position_world_to_clip(world_pos.xyz);
    out.world_position = world_pos;
    // View ray is the direction from camera to this sky vertex
    out.view_ray = normalize(world_pos.xyz - camera_pos);

    return out;
}

@fragment
fn fragment(mesh: SkyVertexOutput) -> @location(0) vec4<f32> {
    // Normalize view ray (direction we're looking)
    let ray = normalize(mesh.view_ray);
    let camera_y = view.world_position.y;

    // Fixed y=0 world-space horizon calculation
    // Ray hits y=0 plane when: camera_y + t * ray.y = 0, so t = -camera_y / ray.y
    // Water visible when camera above water (camera_y > 0) and ray points down (ray.y < 0)
    // and would hit the plane (t > 0, which is guaranteed when signs differ)
    let is_water = camera_y > 0.0 && ray.y < 0.0;

    // Calculate the angle to the y=0 horizon from current camera height
    // At height h, looking at angle θ below horizontal hits y=0
    // The horizon line is at θ = 0 (horizontal) for a flat plane
    // But we want the horizon BLEND to be at y=0 world space

    // For rays pointing down, calculate where they'd hit y=0
    // Normalize the "horizon angle" relative to camera height for smooth blending
    // horizon_ray_y is the ray.y value that would graze y=0 at infinity (always 0 for flat plane)
    let horizon_ray_y = 0.0;

    // Height factor for sky gradient (0 at horizon, 1 at zenith)
    let height = saturate(ray.y);

    // Blend between day and night colors based on sun intensity
    let horizon_color = mix(params.horizon_night, params.horizon_day, params.sun_intensity);
    let zenith_color = mix(params.zenith_night, params.zenith_day, params.sun_intensity);

    // Create smooth gradient from horizon to zenith
    let gradient_factor = pow(height, 0.4);
    var sky_color = mix(horizon_color, zenith_color, gradient_factor);

    // Water colors (from uniform params)
    let deep_water = mix(params.water_deep_night, params.water_deep_day, params.sun_intensity);
    let shallow_water = mix(params.water_shallow_night, params.water_shallow_day, params.sun_intensity);

    // Water gradient based on view angle (deeper = looking more down)
    let water_depth = saturate(-ray.y * 3.0);  // 0 at horizon, 1 looking straight down
    var water_color = mix(shallow_water, deep_water, water_depth);

    // At horizon, water should blend into the sky color (atmospheric perspective)
    // Fresnel reflection is strongest at grazing angles (near horizon)
    let fresnel = pow(1.0 - water_depth, 3.0);
    water_color = mix(water_color, horizon_color, fresnel);

    // Horizon haze - blend sky and water at the fixed y=0 horizon line
    // Use ray.y directly since horizon is always at ray.y = 0 for flat plane
    let horizon_blend = smoothstep(-0.02, 0.02, ray.y);
    var final_color = mix(water_color, sky_color, horizon_blend);

    // Sun direction (normalized)
    let sun_dir = normalize(params.sun_direction);

    // Calculate angle between view ray and sun direction
    let sun_dot = dot(ray, sun_dir);

    // Sun visibility: fade in/out near horizon, always visible when above
    let sun_visible = smoothstep(-0.1, 0.1, sun_dir.y);

    // Sun disc: bright core with soft edge
    let sun_angle = acos(clamp(sun_dot, -1.0, 1.0));
    let sun_disc = 1.0 - smoothstep(params.sun_size * 0.7, params.sun_size, sun_angle);

    // Inner bright core (smaller, more intense)
    let sun_core = 1.0 - smoothstep(params.sun_size * 0.3, params.sun_size * 0.5, sun_angle);

    // Sun glow: multiple layers for realistic atmospheric scattering
    let glow_inner = pow(saturate(sun_dot), params.sun_glow_falloff) * params.sun_glow_intensity;
    let glow_outer = pow(saturate(sun_dot), params.sun_glow_falloff * 0.5) * params.sun_glow_intensity * 0.3;

    // Combine glows
    let sun_glow = glow_inner + glow_outer;

    // Add sun glow to sky (only above horizon)
    final_color = final_color + params.sun_color * sun_glow * sun_visible * horizon_blend;

    // Add sun disc with bright core (only in sky, not water)
    let disc_color = params.sun_color * 3.0;
    let core_color = params.sun_core_color * 5.0;
    final_color = mix(final_color, disc_color, sun_disc * sun_visible * horizon_blend);
    final_color = mix(final_color, core_color, sun_core * sun_visible * horizon_blend);

    // Sun reflection on water
    if (is_water) {
        // Reflect sun direction across horizon plane
        let reflected_sun_dir = vec3<f32>(sun_dir.x, -sun_dir.y, sun_dir.z);
        let reflection_dot = dot(ray, reflected_sun_dir);

        // Stretched reflection (sun path on water)
        let stretch_factor = 1.0 + water_depth * 4.0;  // More stretch when looking down
        let reflected_angle = acos(clamp(reflection_dot, -1.0, 1.0));
        let sun_reflection = 1.0 - smoothstep(params.sun_size * 0.5, params.sun_size * stretch_factor, reflected_angle);

        // Sun glitter/reflection glow on water
        let water_glow = pow(saturate(reflection_dot), params.sun_glow_falloff * 0.5) * params.sun_glow_intensity * 0.5;

        // Add reflection (brighter near horizon due to fresnel)
        let reflection_intensity = fresnel * sun_visible;
        final_color = final_color + params.sun_color * water_glow * reflection_intensity;
        final_color = final_color + params.sun_color * sun_reflection * reflection_intensity * 2.0;
    }

    // Slight color shift near horizon (atmospheric scattering)
    let horizon_scatter = pow(1.0 - abs(ray.y), 4.0) * params.sun_intensity * 0.15;
    let scatter_color = mix(params.scatter_cool, params.scatter_warm, sun_visible);
    final_color = final_color + scatter_color * horizon_scatter;

    return vec4<f32>(final_color, 1.0);
}
