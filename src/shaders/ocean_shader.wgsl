#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
    pbr_functions::apply_pbr_lighting,
    mesh_view_bindings::view
}

override NUMBER_OF_CASCADES: u32 = 3u;

// Cascade 0 - large scale (500m)
// Cascade 1 - medium scale (85m)
// Cascade 2 - small scale (10m)
@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var t_displacements: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1)
var t_derivatives: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2)
var t_foam_persistences: texture_2d_array<f32>;

// Sampler for ocean textures (repeat mode)
@group(#{MATERIAL_BIND_GROUP}) @binding(3)
var s_ocean: sampler;

// Foam texture
@group(#{MATERIAL_BIND_GROUP}) @binding(4)
var t_foam: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5)
var s_foam: sampler;

struct CascadeParams {
    length_scale: f32,
    jacobian_strength: f32,
    lod_cutoff: f32,
    foam_strength: f32,
    // Automatic padding to 16 bytes
}

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
    _padding: f32, 
    cascades: array<CascadeParams, 8>,
    cascade_count: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(6)
var<uniform> params: OceanParamsUniform;

// Cascade length scales (matching reference implementation)
// const LENGTH_SCALE_0: f32 = 500.0;  // Large ocean swells
// const LENGTH_SCALE_1: f32 = 85.0;   // Medium waves
// const LENGTH_SCALE_2: f32 = 10.0;   // Small details/ripples

// LOD parameters for distance-based cascade fading
const LOD_SCALE: f32 = 15.0;
const MID_DIST_THRESHOLD: f32 = 2000.0;   // Include cascade 1 when closer than this
const NEAR_DIST_THRESHOLD: f32 = 300.0;   // Include cascade 2 when closer than this

struct OceanVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) original_xz: vec2<f32>,  // Original world XZ before displacement (for UV calculation)
    @location(2) lod_factors: vec3<f32>,  // LOD scale factors for each cascade
    @location(3) jacobian: f32,
}

@vertex
fn vertex(in: Vertex) -> OceanVertexOutput {
    var model = mesh_functions::get_world_from_local(in.instance_index);
    var world_pos = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 1.0)
    );

    // Store original world XZ before displacement (for UV calculations in fragment shader)
    let original_xz = world_pos.xz;

    // Calculate view distance for LOD-based cascade blending
    let camera_pos = view.world_position;
    let view_dist = max(length(camera_pos - world_pos.xyz), 0.01);
    let displacement_mip_levels = f32(textureNumLevels(t_displacements));

    var total_displacement = vec3(0.);
    var jacobian = 0.;
    // Sample displacement for all cascades
    for (var layer = 0u; layer < NUMBER_OF_CASCADES; layer++) {
      // Distance thresholds for including cascades (lod_cutoff of 0 means always include)
      let cascade_param = params.cascades[layer];
      if (cascade_param.lod_cutoff == 0.0 || view_dist < cascade_param.lod_cutoff) {
        let normalized_distance = view_dist / cascade_param.length_scale;
        // To improve our sampling we use our mip's for the texture sampling.
        let displacement_lod_level = clamp(log2(normalized_distance), 0.0, displacement_mip_levels); 
        // Calculate LOD scales based on distance
        // Each cascade fades based on: min(LOD_SCALE * LENGTH_SCALE / view_dist, 1.0)
        let lod_c0 = min(LOD_SCALE * cascade_param.length_scale / view_dist, 1.0);
        // Calculate UVs from ORIGINAL world position (before displacement)
        let uv = original_xz / cascade_param.length_scale;
        let d0 = textureSampleLevel(t_displacements, s_ocean, uv, layer, displacement_lod_level);
        total_displacement = total_displacement + d0.xyz * lod_c0;
        jacobian = jacobian + d0.w * cascade_param.jacobian_strength;
      }
    }

    // Apply blended displacement (using uniform parameter)
    world_pos.x = world_pos.x + total_displacement.x * params.displacement_scale;
    world_pos.y = world_pos.y + total_displacement.y * params.displacement_scale;
    world_pos.z = world_pos.z + total_displacement.z * params.displacement_scale;

    // Pass data to fragment shader
    var out: OceanVertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);
    out.original_xz = original_xz;  // Pass original position for UV calculations
    out.jacobian = jacobian;

    return out;
}

// Debug modes - set to true to visualize different aspects
const DEBUG_JACOBIAN: bool = false;
const DEBUG_FOAM_TEXTURE: bool = false;
const DEBUG_DISPLACEMENT: bool = false;  // Visualize raw displacement values per cascade

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
    let world_pos = mesh.world_position;
    // Get view direction (from fragment to camera)
    let camera_pos = view.world_position;
    let view_dir = normalize(camera_pos - mesh.world_position.xyz);
    let view_dist = max(length(camera_pos - mesh.world_position.xyz), 0.01);

    // Sample derivatives per-pixel for smooth lighting (always sample all, blend with LOD)

    var blended_deriv = vec4(0.);
    var foam_noise = vec3(0.);
    var base_turbulence = 0.;
    for (var layer = 0u; layer < NUMBER_OF_CASCADES; layer++) {
      let cascade_param = params.cascades[layer];
      // Per-pixel normal calculation: sample derivatives and blend with LOD factors
      let lod_c = min(LOD_SCALE * cascade_param.length_scale / view_dist, 1.0);
      // Use original (pre-displacement) position for UVs, matching old shader behavior
      let uv = mesh.original_xz / params.cascades[layer].length_scale;
      let deriv = textureSample(t_derivatives, s_ocean, uv, layer);

      // Blend derivatives based on LOD factors
      blended_deriv = blended_deriv + deriv * lod_c;

      // Sample persistent foam from compute shader (has exponential decay applied)
      // Each cascade contributes foam at its respective scale
      let foam_persistent = textureSample(t_foam_persistences, s_ocean, uv, layer).r;

      // Blend persistent foam from all cascades with distance-based weighting
      // Use view_dist (with max protection) for consistent LOD calculations
      let lod_foam = min(LOD_SCALE * cascade_param.length_scale / view_dist, 1.0);

      // Combine persistent foam from cascades (lod_cutoff of 0 means always include)
      if (cascade_param.lod_cutoff == 0.0 || view_dist < cascade_param.lod_cutoff) {
        base_turbulence = base_turbulence + foam_persistent * lod_foam * cascade_param.foam_strength;
        // Sample foam texture at multiple scales as noise to break up the pattern
        // Use the cascade UVs for natural multi-scale variation
        let foam_uv = uv * params.foam_tile_scale;

        // Use s_ocean sampler (configured with repeat mode) for foam texture
        let noise = textureSample(t_foam, s_ocean, foam_uv).r;

        // Combine noise at different scales (matching reference - no LOD scaling on noise)
        foam_noise = foam_noise + noise * cascade_param.foam_strength;
      }

    }

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

    // Debug mode: visualize displacement per cascade
    if (DEBUG_DISPLACEMENT) {
      // Sample displacement from each cascade separately for visualization
      var debug_color = vec3(0.0);
      for (var layer = 0u; layer < NUMBER_OF_CASCADES; layer++) {
        let cascade_param = params.cascades[layer];
        let uv = mesh.original_xz / cascade_param.length_scale;
        let d = textureSample(t_displacements, s_ocean, uv, layer);
        // Map displacement.y (vertical) to color channel per cascade
        // Cascade 0 = Red, Cascade 1 = Green, Cascade 2 = Blue
        if (layer == 0u) {
          debug_color.r = saturate(d.y * 0.1 + 0.5);  // Normalize around 0.5
        } else if (layer == 1u) {
          debug_color.g = saturate(d.y * 0.2 + 0.5);
        } else if (layer == 2u) {
          debug_color.b = saturate(d.y * 0.5 + 0.5);
        }
      }
      return vec4(debug_color, 1.0);
    }

    // Distance fog - blend to horizon color at distance
    let fog_factor = smoothstep(params.fog_start, params.fog_end, view_dist);
    ocean_color = mix(ocean_color, params.fog_color, fog_factor);

    return vec4(ocean_color, 1.0);
}
