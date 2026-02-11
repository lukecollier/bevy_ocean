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

// SDF texture for shoreline detection
@group(#{MATERIAL_BIND_GROUP}) @binding(7)
var t_sdf: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8)
var s_sdf: sampler;

// Shore parameters uniform
struct ShoreParamsUniform {
    sdf_origin: vec2<f32>,
    sdf_extent: vec2<f32>,
    depth_scale: f32,
    max_depth: f32,
    blend_start: f32,
    blend_end: f32,
    gerstner_amplitude: f32,
    gerstner_wavelength: f32,
    gerstner_steepness: f32,
    gerstner_num_waves: f32,
    gerstner_speed: f32,
    shore_foam_distance: f32,
    shore_foam_intensity: f32,
    shore_foam_band_freq: f32,
    swell_direction: vec2<f32>,
    time: f32,
    _padding1: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(9)
var<uniform> shore: ShoreParamsUniform;

// LOD parameters for distance-based cascade fading
const LOD_SCALE: f32 = 15.0;
const MID_DIST_THRESHOLD: f32 = 2000.0;   // Include cascade 1 when closer than this
const NEAR_DIST_THRESHOLD: f32 = 300.0;   // Include cascade 2 when closer than this

const PI: f32 = 3.14159265;
const GRAVITY: f32 = 9.81;

// ---- Analytical SDF Functions ----
// Computes a circular island SDF analytically from world position.
// Returns world-space signed distance: positive = water, negative = land.

fn analytical_sdf(world_xz: vec2<f32>) -> f32 {
    let center = shore.sdf_origin + shore.sdf_extent * 0.5;
    let radius = 0.5 * max(shore.sdf_extent.x, shore.sdf_extent.y);
    return length(world_xz - center) - radius;
}

fn analytical_sdf_gradient(world_xz: vec2<f32>) -> vec2<f32> {
    let center = shore.sdf_origin + shore.sdf_extent * 0.5;
    let dir = world_xz - center;
    let len = length(dir);
    if (len < 0.0001) {
        return vec2(0.0, 1.0);
    }
    return dir / len;
}

fn sdf_to_depth(sdf_dist: f32) -> f32 {
    return clamp(sdf_dist * shore.depth_scale, 0.0, shore.max_depth);
}

// ---- Hash-based 2D value noise (deterministic, no textures) ----
fn hash2d(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep interpolation
    let a = hash2d(i);
    let b = hash2d(i + vec2(1.0, 0.0));
    let c = hash2d(i + vec2(0.0, 1.0));
    let d = hash2d(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ---- Shore Wave Functions (Refracted Sum of Sines) ----
// Waves approach from a dominant swell direction and refract toward the
// shore-approaching direction as depth decreases.  Waves break and dissipate
// near the shoreline (amplitude → 0 at depth = 0).

// Shoaling + breaking envelope.
// Waves shoal (amplitude grows) in intermediate depths, then break and
// dissipate as depth approaches 0.  Returns a multiplier on amplitude.
fn shoal_and_break(depth: f32) -> f32 {
    // Breaking: amplitude fades to 0 only at the very shoreline
    let breaking = smoothstep(0.0, 1.0, depth);
    // Shoaling: waves grow as they enter shallow water
    let shoal = clamp(sqrt(shore.max_depth / max(depth, 1.0)), 1.0, 2.5);
    return shoal * breaking;
}

// Returns vec4: xyz = displacement, w = breaking foam (0..1)
fn shore_wave_displacement(
    sdf_grad: vec2<f32>,
    base_amplitude: f32,
    world_xz: vec2<f32>,
    depth: f32,
) -> vec4<f32> {
    var disp = vec3(0.0);
    var foam = 0.0;
    let num_waves = i32(shore.gerstner_num_waves);
    let envelope = shoal_and_break(depth);
    let toward_shore = -sdf_grad;
    let Q = shore.gerstner_steepness;
    let depth_ratio = saturate(depth / shore.max_depth);

    // Gentle spatial amplitude variation (static — doesn't bob)
    let amp_variation = value_noise(world_xz * 0.02) * 0.4 + 0.8; // [0.8, 1.2]

    for (var i = 0; i < num_waves; i++) {
        let fi = f32(i);
        let harmonic = fi + 1.0;
        let wavelength = shore.gerstner_wavelength / harmonic;
        let k = 2.0 * PI / wavelength;
        let w = sqrt(GRAVITY * k) * shore.gerstner_speed;
        let amplitude = base_amplitude / harmonic * envelope * amp_variation;

        // Slight angular spread per harmonic
        let perp = vec2(-toward_shore.y, toward_shore.x);
        let spread_angle = (fi - f32(num_waves - 1) * 0.5) * 0.12;
        let dir = normalize(toward_shore * cos(spread_angle) + perp * sin(spread_angle));

        // Along-crest envelope: creates finite wave segments
        let along_crest = dot(perp, world_xz);
        let env_freq = 0.04 / harmonic;
        let env = max(sin(along_crest * env_freq + fi * 2.39), 0.0);

        // Clean traveling wave
        let phase = dot(dir, world_xz) * k - w * shore.time;

        // Gerstner displacement: horizontal pulls toward crests, vertical is sine
        let qi = Q / (k * amplitude * f32(num_waves) + 0.001);
        disp.x += amplitude * env * qi * dir.x * cos(phase);
        disp.y += amplitude * env * sin(phase);
        disp.z += amplitude * env * qi * dir.y * cos(phase);

        // Foam at crests in shallow water
        let steepness = amplitude * env * k;
        let crest = saturate(sin(phase));
        let break_threshold = mix(0.2, 0.8, depth_ratio);
        foam += smoothstep(break_threshold, break_threshold + 0.1, steepness) * crest;
    }

    return vec4(disp, saturate(foam));
}

fn shore_wave_normal(
    sdf_grad: vec2<f32>,
    base_amplitude: f32,
    world_xz: vec2<f32>,
    depth: f32,
) -> vec3<f32> {
    var dydx = 0.0;
    var dydz = 0.0;
    let num_waves = i32(shore.gerstner_num_waves);
    let envelope = shoal_and_break(depth);
    let toward_shore = -sdf_grad;

    // Must match displacement
    let amp_variation = value_noise(world_xz * 0.02) * 0.4 + 0.8;

    for (var i = 0; i < num_waves; i++) {
        let fi = f32(i);
        let harmonic = fi + 1.0;
        let wavelength = shore.gerstner_wavelength / harmonic;
        let k = 2.0 * PI / wavelength;
        let w = sqrt(GRAVITY * k) * shore.gerstner_speed;
        let amplitude = base_amplitude / harmonic * envelope * amp_variation;

        let perp = vec2(-toward_shore.y, toward_shore.x);
        let spread_angle = (fi - f32(num_waves - 1) * 0.5) * 0.12;
        let dir = normalize(toward_shore * cos(spread_angle) + perp * sin(spread_angle));

        // Along-crest envelope (must match displacement)
        let along_crest = dot(perp, world_xz);
        let env_freq = 0.04 / harmonic;
        let env = max(sin(along_crest * env_freq + fi * 2.39), 0.0);

        // Clean traveling wave — must match displacement
        let phase = dot(dir, world_xz) * k - w * shore.time;
        let dy_dphase = amplitude * env * cos(phase) * k;

        dydx += dy_dphase * dir.x;
        dydz += dy_dphase * dir.y;
    }

    return normalize(vec3(-dydx, 1.0, -dydz));
}

// ---- Inner Shore Zone: Contour-Following Coastal Wave Sets ----
// Wave fronts follow the SDF contour ())) shape around the coast).
// Each wave set spawns at the outer edge and rolls inward over time.
// The shore angle offsets timing so different stretches of coast
// receive sets at different times, creating natural spacing.

const INNER_ZONE_START: f32 = 256.0;
const INNER_ZONE_FADE: f32 = 200.0;
const INNER_NUM_WAVES: i32 = 5;

fn inner_shore_displacement(
    sdf_grad: vec2<f32>,
    world_xz: vec2<f32>,
    sdf_distance: f32,
) -> vec4<f32> {
    var disp = vec3(0.0);
    var foam = 0.0;
    let toward_shore = -sdf_grad;

    // Blend factor: 1 inside 200m, fades to 0 at 256m
    let inner_blend = smoothstep(INNER_ZONE_START, INNER_ZONE_FADE, sdf_distance);

    // Distance-based amplitude: largest at zone start, shrinks toward shore
    let dist_ratio = saturate(sdf_distance / INNER_ZONE_START);
    let dist_amp = mix(0.03, 1.0, dist_ratio);

    // Shore angle: determines when this stretch of coast receives a wave set
    let shore_angle = atan2(sdf_grad.y, sdf_grad.x);

    for (var i = 0; i < INNER_NUM_WAVES; i++) {
        let fi = f32(i);

        // Wavelength cycles over time — each wave has its own slow period
        let cycle_period = 20.0 + fi * 7.0;
        let cycle = sin(shore.time / cycle_period + fi * 2.1) * 0.5 + 0.5;
        let wavelength = mix(12.0, 30.0, cycle);
        let base_amp = 0.15 + fi * 0.02;
        let amplitude = base_amp * inner_blend * dist_amp;
        let k = 2.0 * PI / wavelength;
        let w = sqrt(GRAVITY * k) * shore.gerstner_speed;

        // Phase driven by SDF distance — crests follow shoreline contour
        // Noise-based angle offset: each wave gets a distinct, non-uniform phase
        // per stretch of coast so fronts are staggered/intermittent around the island
        let angle_norm = shore_angle / PI; // -1..1
        let angle_offset = hash2d(vec2(angle_norm * 4.0 + fi * 5.3, fi * 13.7)) * 2.0 * PI;
        let phase = sdf_distance * k - w * shore.time + angle_offset;

        // Gerstner displacement: moves toward shore along SDF gradient
        let Q = 0.3;
        let qi = Q / (k * amplitude * f32(INNER_NUM_WAVES) + 0.001);
        disp.x += amplitude * qi * toward_shore.x * cos(phase);
        disp.y += amplitude * sin(phase);
        disp.z += amplitude * qi * toward_shore.y * cos(phase);

        // Foam on crests — carried all the way to shore
        let crest = saturate(sin(phase));
        foam += crest * amplitude * k * 0.5;
    }

    return vec4(disp, saturate(foam));
}

fn inner_shore_normal(
    sdf_grad: vec2<f32>,
    world_xz: vec2<f32>,
    sdf_distance: f32,
) -> vec3<f32> {
    var dydx = 0.0;
    var dydz = 0.0;
    let toward_shore = -sdf_grad;

    let inner_blend = smoothstep(INNER_ZONE_START, INNER_ZONE_FADE, sdf_distance);

    let dist_ratio = saturate(sdf_distance / INNER_ZONE_START);
    let dist_amp = mix(0.03, 1.0, dist_ratio);

    let shore_angle = atan2(sdf_grad.y, sdf_grad.x);

    for (var i = 0; i < INNER_NUM_WAVES; i++) {
        let fi = f32(i);

        // Must match displacement exactly
        let cycle_period = 20.0 + fi * 7.0;
        let cycle = sin(shore.time / cycle_period + fi * 2.1) * 0.5 + 0.5;
        let wavelength = mix(12.0, 30.0, cycle);
        let base_amp = 0.15 + fi * 0.02;
        let amplitude = base_amp * inner_blend * dist_amp;
        let k = 2.0 * PI / wavelength;
        let w = sqrt(GRAVITY * k) * shore.gerstner_speed;

        let angle_norm = shore_angle / PI;
        let angle_offset = hash2d(vec2(angle_norm * 4.0 + fi * 5.3, fi * 13.7)) * 2.0 * PI;
        let phase = sdf_distance * k - w * shore.time + angle_offset;

        // Normal derivative is along toward-shore direction (matches displacement)
        let dy_dphase = amplitude * cos(phase) * k;
        dydx += dy_dphase * toward_shore.x;
        dydz += dy_dphase * toward_shore.y;
    }

    return normalize(vec3(-dydx, 1.0, -dydz));
}

struct OceanVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) original_xz: vec2<f32>,  // Original world XZ before displacement (for UV calculation)
    @location(2) lod_factors: vec3<f32>,  // LOD scale factors for each cascade
    @location(3) jacobian: f32,
    @location(4) shore_blend: f32,        // 0 = pure FFT, 1 = pure Gerstner
    @location(5) sdf_distance: f32,       // World-space SDF distance for fragment foam
    @location(6) shore_wave_height: f32,  // Breaking foam from wave steepness (0..1)
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

    // --- Shore wave blending ---
    // Analytical SDF: returns world-space signed distance directly
    let sdf_world_dist = analytical_sdf(original_xz);

    // shore_blend: 0 in open ocean (beyond blend_start), 1 near shore (within blend_end)
    // Only computed for water (positive SDF).
    var shore_blend = 0.0;
    if (sdf_world_dist > 0.0) {
        shore_blend = 1.0 - smoothstep(shore.blend_end, shore.blend_start, sdf_world_dist);
    }

    // Land (inside island): no displacement at all
    var blended_displacement = total_displacement;
    var shore_wave_h = 0.0;
    if (sdf_world_dist <= 0.0) {
        blended_displacement = vec3(0.0);
    } else {
        // Transition zone: FFT fully fades to 0 at shore_blend=1.0,
        // replaced entirely by shore sine waves near coast
        if (shore_blend > 0.001) {
            let grad = analytical_sdf_gradient(original_xz);
            let depth = sdf_to_depth(sdf_world_dist);

            let shore_result = shore_wave_displacement(grad, shore.gerstner_amplitude, original_xz, depth);
            let shore_disp = shore_result.xyz;
            shore_wave_h = shore_result.w; // Now carries breaking foam, not raw height
            // FFT fades completely as shore_blend approaches 1.0
            let fft_atten = 1.0 - shore_blend;
            blended_displacement = total_displacement * fft_atten + shore_disp * shore_blend;
        }

        // Inner shore zone: additive pseudorandom coastal waves within 256m
        if (sdf_world_dist < INNER_ZONE_START) {
            let grad = analytical_sdf_gradient(original_xz);
            let inner_result = inner_shore_displacement(grad, original_xz, sdf_world_dist);
            blended_displacement = blended_displacement + inner_result.xyz;
            shore_wave_h = shore_wave_h + inner_result.w;
        }
    }

    // Apply blended displacement (using uniform parameter)
    world_pos.x = world_pos.x + blended_displacement.x * params.displacement_scale;
    world_pos.y = world_pos.y + blended_displacement.y * params.displacement_scale;
    world_pos.z = world_pos.z + blended_displacement.z * params.displacement_scale;

    // Pass data to fragment shader
    var out: OceanVertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);
    out.original_xz = original_xz;  // Pass original position for UV calculations
    out.jacobian = jacobian;
    out.shore_blend = shore_blend;
    out.sdf_distance = sdf_world_dist;
    out.shore_wave_height = shore_wave_h;

    return out;
}

// Debug modes - set to true to visualize different aspects
const DEBUG_JACOBIAN: bool = false;
const DEBUG_FOAM_TEXTURE: bool = false;
const DEBUG_DISPLACEMENT: bool = false;  // Visualize raw displacement values per cascade
const DEBUG_SDF: bool = false;           // Visualize SDF values and blend zones

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
    // Land: render white, skip all shading (early out)
    if (mesh.sdf_distance <= 0.0) {
        return vec4(1.0, 1.0, 1.0, 1.0);
    }

    let world_pos = mesh.world_position;
    // Get view direction (from fragment to camera)
    let camera_pos = view.world_position;
    let view_dir = normalize(camera_pos - mesh.world_position.xyz);
    let view_dist = max(length(camera_pos - mesh.world_position.xyz), 0.01);


    let t_foam_miplevels = f32(textureNumLevels(t_foam_persistences));
    let t_derivs_miplevels = f32(textureNumLevels(t_derivatives));

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
      let normalized_distance = view_dist / cascade_param.length_scale;
      let deriv_lod_level = clamp(log2(normalized_distance), 0.0, t_derivs_miplevels);
      let deriv = textureSampleLevel(t_derivatives, s_ocean, uv, layer, deriv_lod_level);

      // FFT derivatives fully fade to 0 in shore zone — shore sine waves
      // own the surface entirely within blend_end
      let fft_deriv_weight = 1.0 - mesh.shore_blend;
      blended_deriv = blended_deriv + deriv * lod_c * fft_deriv_weight;

      // Combine persistent foam from cascades (lod_cutoff of 0 means always include)
      if (cascade_param.lod_cutoff == 0.0 || view_dist < cascade_param.lod_cutoff) {
        // Sample persistent foam from compute shader (has exponential decay applied)
        // Each cascade contributes foam at its respective scale
        let foam_lod_level = clamp(log2(normalized_distance), 0.0, t_foam_miplevels);
        let foam_persistent = textureSampleLevel(t_foam_persistences, s_ocean, uv, layer, foam_lod_level).r;
        // Blend persistent foam from all cascades with distance-based weighting
        // Use view_dist (with max protection) for consistent LOD calculations
        let lod_foam = min(LOD_SCALE * cascade_param.length_scale / view_dist, 1.0);
        // FFT foam also fades in shore zone — shore foam takes over
        base_turbulence = base_turbulence + foam_persistent * lod_foam * cascade_param.foam_strength * fft_deriv_weight;
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
    var normal = normalize(vec3<f32>(-slope_x, 1.0, -slope_z));

    // Blend shore wave normals over FFT normals — shore waves define the
    // dominant surface shape, remaining FFT derivatives add micro-detail
    if (mesh.shore_blend > 0.001) {
        let sdf_world_dist_frag = analytical_sdf(mesh.original_xz);
        let grad = analytical_sdf_gradient(mesh.original_xz);
        let depth = sdf_to_depth(sdf_world_dist_frag);
        let s_normal = shore_wave_normal(grad, shore.gerstner_amplitude, mesh.original_xz, depth);

        // Multi-octave normal perturbation: FBM-style surface ripple detail
        let wxz = mesh.original_xz;
        let t = shore.time * 0.3;
        // 3 octaves at increasing frequency and decreasing amplitude
        var px = 0.0;
        var pz = 0.0;
        px += (value_noise(wxz * 0.5 + vec2(t, 0.0)) - 0.5) * 0.12;
        pz += (value_noise(wxz * 0.5 + vec2(43.7, 91.3 + t)) - 0.5) * 0.12;
        px += (value_noise(wxz * 1.5 + vec2(t * 1.3, 17.1)) - 0.5) * 0.08;
        pz += (value_noise(wxz * 1.5 + vec2(61.2, t * 1.3)) - 0.5) * 0.08;
        px += (value_noise(wxz * 4.0 + vec2(t * 1.7, 33.9)) - 0.5) * 0.05;
        pz += (value_noise(wxz * 4.0 + vec2(88.4, t * 1.7)) - 0.5) * 0.05;
        let perturbed = normalize(s_normal + vec3(px, 0.0, pz));

        normal = normalize(mix(normal, perturbed, mesh.shore_blend));
    }

    // Inner shore zone normals: blend additively when within 256m
    if (mesh.sdf_distance > 0.0 && mesh.sdf_distance < INNER_ZONE_START) {
        let grad = analytical_sdf_gradient(mesh.original_xz);
        let inner_n = inner_shore_normal(grad, mesh.original_xz, mesh.sdf_distance);
        let inner_blend = smoothstep(INNER_ZONE_START, INNER_ZONE_FADE, mesh.sdf_distance);
        // Blend inner wave normals into existing normal
        normal = normalize(mix(normal, inner_n, inner_blend * 0.5));
    }

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

    // Shore foam: driven by per-harmonic wave steepness vs depth-dependent
    // breaking threshold. Foam is coherent with wave structure — it follows
    // crests, and some waves break early while others reach shore.
    if (mesh.sdf_distance > 0.0 && (mesh.shore_blend > 0.001 || mesh.sdf_distance < INNER_ZONE_START)) {
        let breaking_foam = mesh.shore_wave_height; // Already 0..1 from vertex (includes inner shore waves)

        // Light texture detail for micro-variation (not a random mask)
        let foam_uv = mesh.original_xz * params.foam_tile_scale * 0.03;
        let detail = textureSample(t_foam, s_ocean, foam_uv).r * 0.2 + 0.8;

        let foam_blend = max(mesh.shore_blend, smoothstep(INNER_ZONE_START, INNER_ZONE_FADE, mesh.sdf_distance));
        base_turbulence = base_turbulence + breaking_foam * detail * foam_blend * shore.shore_foam_intensity;

        // Surf-zone foam: persistent foam right at the waterline
        let surf_width = 8.0;
        let surf_foam = (1.0 - smoothstep(0.0, surf_width, mesh.sdf_distance)) * 0.8;
        let surf_detail = textureSample(t_foam, s_ocean, foam_uv * 3.0).r * 0.3 + 0.7;
        base_turbulence = base_turbulence + surf_foam * surf_detail;
    }

    // Use noise to modulate turbulence - creates organic foam breakup
    // The noise acts as a threshold mask for where foam appears
    // Same pipeline for both FFT and shore-driven foam
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

    // Debug mode: visualize SDF values and blend zones
    if (DEBUG_SDF) {
      let sdf_val = mesh.sdf_distance / max(shore.sdf_extent.x, shore.sdf_extent.y);
      // Red = inside island (negative SDF), Green = water, Blue = blend zone
      var debug_color = vec3(0.0);
      if (sdf_val < 0.0) {
          debug_color.r = saturate(-sdf_val * 5.0); // Land
      } else {
          debug_color.g = saturate(sdf_val * 2.0);  // Water distance
      }
      debug_color.b = mesh.shore_blend; // Blend zone visualization
      return vec4(debug_color, 1.0);
    }

    // Distance fog - blend to horizon color at distance
    let fog_factor = smoothstep(params.fog_start, params.fog_end, view_dist);
    ocean_color = mix(ocean_color, params.fog_color, fog_factor);

    return vec4(ocean_color, 1.0);
}
