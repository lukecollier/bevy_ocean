#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
    pbr_functions::apply_pbr_lighting,
    mesh_view_bindings::view
}

// Cascade 0 - large scale (500m)
@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var t_displacement_0: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1)
var t_derivatives_0: texture_2d<f32>;

// Cascade 1 - medium scale (85m)
@group(#{MATERIAL_BIND_GROUP}) @binding(2)
var t_displacement_1: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3)
var t_derivatives_1: texture_2d<f32>;

// Cascade 2 - small scale (10m)
@group(#{MATERIAL_BIND_GROUP}) @binding(4)
var t_displacement_2: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5)
var t_derivatives_2: texture_2d<f32>;

// Sampler for ocean textures (repeat mode)
@group(#{MATERIAL_BIND_GROUP}) @binding(6)
var s_ocean: sampler;

// Foam texture
@group(#{MATERIAL_BIND_GROUP}) @binding(7)
var t_foam: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8)
var s_foam: sampler;

// Ocean parameters uniform (synced from OceanParams resource)
struct OceanParamsUniform {
    displacement_scale: f32,
    normal_strength: f32,
    foam_threshold: f32,
    foam_multiplier: f32,
    foam_tile_scale: f32,
    roughness: f32,
    light_intensity: f32,
    sss_intensity: f32,
    sun_direction: vec3<f32>,
    fog_color: vec3<f32>,
    fog_start: f32,
    fog_end: f32,
    // Ocean colors
    deep_color: vec3<f32>,
    shallow_color: vec3<f32>,
    sky_day: vec3<f32>,
    sky_night: vec3<f32>,
    sun_color: vec3<f32>,
    sss_color: vec3<f32>,
    foam_color: vec3<f32>,
    ambient_color: vec3<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(9)
var<uniform> params: OceanParamsUniform;

// Foam persistence textures (updated by compute shader each frame)
@group(#{MATERIAL_BIND_GROUP}) @binding(10)
var t_foam_persistence_0: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(11)
var t_foam_persistence_1: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(12)
var t_foam_persistence_2: texture_2d<f32>;

// Cascade length scales (matching reference implementation)
const LENGTH_SCALE_0: f32 = 500.0;  // Large ocean swells
const LENGTH_SCALE_1: f32 = 85.0;   // Medium waves
const LENGTH_SCALE_2: f32 = 10.0;   // Small details/ripples

// LOD parameters for distance-based cascade fading
const LOD_SCALE: f32 = 15.0;
const MID_DIST_THRESHOLD: f32 = 2000.0;   // Include cascade 1 when closer than this
const NEAR_DIST_THRESHOLD: f32 = 300.0;   // Include cascade 2 when closer than this

struct OceanVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) uv_0: vec2<f32>,      // UV for cascade 0
    @location(2) uv_1: vec2<f32>,      // UV for cascade 1
    @location(3) uv_2: vec2<f32>,      // UV for cascade 2
    @location(4) lod_factors: vec3<f32>, // LOD scale factors for each cascade
    @location(5) jacobian: f32,
}

@vertex
fn vertex(in: Vertex) -> OceanVertexOutput {
    var model = mesh_functions::get_world_from_local(in.instance_index);
    var world_pos = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 1.0)
    );

    // Calculate view distance for LOD-based cascade blending
    let camera_pos = view.world_position;
    let view_dist = length(camera_pos - world_pos.xyz);

    // Calculate LOD scales based on distance (matching reference implementation)
    // Each cascade fades based on: min(LOD_SCALE * LENGTH_SCALE / view_dist, 1.0)
    let lod_c0 = min(LOD_SCALE * LENGTH_SCALE_0 / view_dist, 1.0);
    let lod_c1 = min(LOD_SCALE * LENGTH_SCALE_1 / view_dist, 1.0);
    let lod_c2 = min(LOD_SCALE * LENGTH_SCALE_2 / view_dist, 1.0);

    // Distance thresholds for including cascades
    let include_mid = view_dist < MID_DIST_THRESHOLD;
    let include_near = view_dist < NEAR_DIST_THRESHOLD;

    // Calculate UVs from world position for each cascade
    // This allows meshes to tile seamlessly and scale independently
    let uv_0 = world_pos.xz / LENGTH_SCALE_0;
    let uv_1 = world_pos.xz / LENGTH_SCALE_1;
    let uv_2 = world_pos.xz / LENGTH_SCALE_2;

    // Sample displacement from cascade 0 (always included)
    let d0 = textureSampleLevel(t_displacement_0, s_ocean, uv_0, 0.0);
    var total_displacement = d0.xyz * lod_c0;
    var jacobian = d0.w * 0.6;

    // Add cascade 1 if within mid distance
    if (include_mid) {
        let d1 = textureSampleLevel(t_displacement_1, s_ocean, uv_1, 0.0);
        total_displacement = total_displacement + d1.xyz * lod_c1;
        jacobian = jacobian + d1.w * 0.17;
    }

    // Add cascade 2 if within near distance
    if (include_near) {
        let d2 = textureSampleLevel(t_displacement_2, s_ocean, uv_2, 0.0);
        total_displacement = total_displacement + d2.xyz * lod_c2;
        jacobian = jacobian + d2.w * 0.23;
    }

    // Apply blended displacement (using uniform parameter)
    world_pos.x = world_pos.x + total_displacement.x * params.displacement_scale;
    world_pos.y = world_pos.y + total_displacement.y * params.displacement_scale;
    world_pos.z = world_pos.z + total_displacement.z * params.displacement_scale;

    // Pass LOD factors to fragment shader for per-pixel normal calculation
    var out: OceanVertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);
    out.uv_0 = uv_0;
    out.uv_1 = uv_1;
    out.uv_2 = uv_2;
    out.lod_factors = vec3<f32>(lod_c0, lod_c1, lod_c2);
    out.jacobian = jacobian;

    return out;
}

// Debug modes
const DEBUG_JACOBIAN: bool = false;
const DEBUG_FOAM_TEXTURE: bool = false;

// PBR helper functions
fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ndoth = max(dot(n, h), 0.0);
    let ndoth2 = ndoth * ndoth;
    let denom = ndoth2 * (a2 - 1.0) + 1.0;
    return a2 / (3.14159 * denom * denom);
}

fn geometry_schlick_ggx(ndotv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return ndotv / (ndotv * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let ndotv = max(dot(n, v), 0.0);
    let ndotl = max(dot(n, l), 0.0);
    let ggx1 = geometry_schlick_ggx(ndotv, roughness);
    let ggx2 = geometry_schlick_ggx(ndotl, roughness);
    return ggx1 * ggx2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// Compute geometric roughness from normal variation across the pixel
// This prevents specular aliasing at close range
// Note: With per-pixel normals from texture, we scale down the variance
// contribution to avoid killing specular highlights
fn compute_geometric_roughness(normal: vec3<f32>, base_roughness: f32) -> f32 {
    // Screen-space derivatives of the normal
    let dndu = dpdx(normal);
    let dndv = dpdy(normal);

    // Variance approximation - how much the normal changes across this pixel
    // Scale down significantly for per-pixel normals to preserve specular
    let variance = (dot(dndu, dndu) + dot(dndv, dndv)) * 0.1;

    // Add variance to roughness squared, then sqrt back
    // This blurs specular where normals vary rapidly (close-up detail)
    let adjusted_roughness = sqrt(base_roughness * base_roughness + variance);

    return saturate(adjusted_roughness);
}

@fragment
fn fragment(mesh: OceanVertexOutput) -> @location(0) vec4<f32> {
    // Get view direction (from fragment to camera)
    let camera_pos = view.world_position;
    let view_dir = normalize(camera_pos - mesh.world_position.xyz);
    let view_dist = length(camera_pos - mesh.world_position.xyz);

    // Per-pixel normal calculation: sample derivatives and blend with LOD factors
    let lod_c0 = mesh.lod_factors.x;
    let lod_c1 = mesh.lod_factors.y;
    let lod_c2 = mesh.lod_factors.z;

    // Sample derivatives per-pixel for smooth lighting (always sample all, blend with LOD)
    let deriv_0 = textureSample(t_derivatives_0, s_ocean, mesh.uv_0);
    let deriv_1 = textureSample(t_derivatives_1, s_ocean, mesh.uv_1);
    let deriv_2 = textureSample(t_derivatives_2, s_ocean, mesh.uv_2);

    // Blend derivatives based on LOD factors
    let blended_deriv = deriv_0 * lod_c0 + deriv_1 * lod_c1 + deriv_2 * lod_c2;

    // Compute normal from blended derivatives (per-pixel)
    // Same formula as was used in vertex shader
    let slope_x = blended_deriv.x / (1.0 + blended_deriv.z) * params.normal_strength;
    let slope_z = blended_deriv.y / (1.0 + blended_deriv.w) * params.normal_strength;
    let normal = normalize(vec3<f32>(-slope_x, 1.0, -slope_z));

    // Light direction (sun position in sky)
    let light_dir = normalize(params.sun_direction);
    let half_vec = normalize(light_dir + view_dir);

    // Sun height factor: smoothly fade sun contribution when below horizon
    // light_dir.y < 0 means sun is below horizon
    // Smooth transition from -0.1 to 0.2 for gradual sunrise/sunset
    let sun_height = saturate((light_dir.y + 0.1) / 0.3);

    // PBR parameters for water (using uniform)
    // Geometric roughness prevents specular aliasing at close range
    let roughness = compute_geometric_roughness(normal, params.roughness);
    let f0 = vec3<f32>(0.02);  // Water's base reflectivity (IOR ~1.33)

    // Calculate PBR terms
    let ndotv = max(dot(normal, view_dir), 0.0);
    let ndotl = max(dot(normal, light_dir), 0.0);
    let ndoth = max(dot(normal, half_vec), 0.0);

    // GGX specular
    let ndf = distribution_ggx(normal, half_vec, roughness);
    let g = geometry_smith(normal, view_dir, light_dir, roughness);
    let f = fresnel_schlick(max(dot(half_vec, view_dir), 0.0), f0);

    let numerator = ndf * g * f;
    let denominator = 4.0 * ndotv * ndotl + 0.0001;
    let specular = numerator / denominator;

    // Wrapped diffuse for softer look
    let wrapped_diffuse = max(dot(normal, light_dir) * 0.5 + 0.5, 0.0);

    // Fresnel for environment reflection (separate from specular fresnel)
    let env_fresnel = fresnel_schlick(ndotv, f0);

    // Subsurface scattering approximation (light through wave crests)
    // Light scatters through thin parts of waves, creating a glow effect
    let wave_height = mesh.world_position.y;
    let sss_mask = saturate(dot(view_dir, -light_dir) * 0.5 + 0.5);
    let sss = sss_mask * saturate(wave_height * 0.3 + 0.2) * params.sss_intensity;

    // Ocean colors from params
    let sky_color = mix(params.sky_night, params.sky_day, sun_height);

    // Mix ocean color based on view angle and wave height
    let depth_factor = saturate(1.0 - ndotv + wave_height * 0.1);
    var ocean_color = mix(params.shallow_color, params.deep_color, depth_factor);

    // Add wrapped diffuse lighting (softer shadows), scaled by sun height
    ocean_color = ocean_color * (0.4 + 0.6 * wrapped_diffuse * sun_height);

    // Add subsurface scattering (only when sun is up)
    ocean_color = ocean_color + params.sss_color * sss * sun_height;

    // Add sky reflection based on PBR fresnel
    // Reduce (but don't eliminate) reflection at close range to preserve surface detail
    let reflection_dist_fade = mix(0.3, 1.0, saturate(view_dist / 150.0));
    let reflection_strength = env_fresnel.r;  // Use scalar from fresnel
    ocean_color = mix(ocean_color, sky_color, reflection_strength * 0.5 * reflection_dist_fade);

    // Add PBR sun specular highlight (using uniform light intensity)
    // Fade out when sun is below horizon
    ocean_color = ocean_color + params.sun_color * specular * params.light_intensity * ndotl * sun_height;

    // Sample persistent foam from compute shader (has exponential decay applied)
    // Each cascade contributes foam at its respective scale
    let foam_persistent_0 = textureSample(t_foam_persistence_0, s_ocean, mesh.uv_0).r;
    let foam_persistent_1 = textureSample(t_foam_persistence_1, s_ocean, mesh.uv_1).r;
    let foam_persistent_2 = textureSample(t_foam_persistence_2, s_ocean, mesh.uv_2).r;

    // Blend persistent foam from all cascades with distance-based weighting
    let camera_pos_frag = view.world_position;
    let view_dist_frag = length(camera_pos_frag - mesh.world_position.xyz);
    let lod_foam_0 = min(LOD_SCALE * LENGTH_SCALE_0 / view_dist_frag, 1.0);
    let lod_foam_1 = min(LOD_SCALE * LENGTH_SCALE_1 / view_dist_frag, 1.0);
    let lod_foam_2 = min(LOD_SCALE * LENGTH_SCALE_2 / view_dist_frag, 1.0);

    // Combine persistent foam from cascades
    var base_turbulence = foam_persistent_0 * lod_foam_0 * 0.6;
    if (view_dist_frag < MID_DIST_THRESHOLD) {
        base_turbulence = base_turbulence + foam_persistent_1 * lod_foam_1 * 0.25;
    }
    if (view_dist_frag < NEAR_DIST_THRESHOLD) {
        base_turbulence = base_turbulence + foam_persistent_2 * lod_foam_2 * 0.15;
    }

    // Sample foam texture at multiple scales as noise to break up the pattern
    // Use the cascade UVs for natural multi-scale variation
    let foam_uv_large = mesh.uv_0 * params.foam_tile_scale;
    let foam_uv_medium = mesh.uv_1 * params.foam_tile_scale;
    let foam_uv_small = mesh.uv_2 * params.foam_tile_scale;

    // Use s_ocean sampler (configured with repeat mode) for foam texture
    let noise_large = textureSample(t_foam, s_ocean, foam_uv_large).r;
    let noise_medium = textureSample(t_foam, s_ocean, foam_uv_medium).r;
    let noise_small = textureSample(t_foam, s_ocean, foam_uv_small).r;

    // Combine noise at different scales (like the original's multi-cascade approach)
    let foam_noise = noise_large * 0.6 + noise_medium * 0.25 + noise_small * 0.15;

    // Use noise to modulate turbulence - creates organic foam breakup
    // The noise acts as a threshold mask for where foam appears
    let foam_mask = saturate((base_turbulence - (1.0 - foam_noise) * 0.5) * 2.0);

    // Add foam as highlights (uses foam color from params)
    ocean_color = ocean_color + params.foam_color * foam_mask * 0.8;

    // Add ambient light
    ocean_color = ocean_color + params.ambient_color;

    // Debug mode: visualize jacobian values
    if (DEBUG_JACOBIAN) {
        // Show jacobian mapped to grayscale with foam areas highlighted
        // Red channel = turbulence (where foam would appear)
        // Green channel = jacobian normalized (0.5 = jacobian of 1.0)
        // This helps visualize actual value ranges
        let j = mesh.jacobian;
        let j_normalized = saturate(j * 0.5);  // Map 0-2 range to 0-1
        return vec4(base_turbulence, j_normalized, 0.0, 1.0);
    }

    // Debug mode: visualize foam noise
    if (DEBUG_FOAM_TEXTURE) {
        return vec4(vec3(foam_noise), 1.0);
    }

    // Distance fog - blend to horizon color at distance
    let fog_factor = smoothstep(params.fog_start, params.fog_end, view_dist);
    ocean_color = mix(ocean_color, params.fog_color, fog_factor);

    return vec4(ocean_color, 1.0);
}
