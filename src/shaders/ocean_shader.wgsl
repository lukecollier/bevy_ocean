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

// ---- Texture-Sampled SDF Functions ----
// Samples the SDF texture (binding 7/8) instead of analytical functions.
// The SDF texture stores UV-space signed distance; we convert to world space.

fn analytical_sdf(world_xz: vec2<f32>) -> f32 {
    let uv = (world_xz - shore.sdf_origin) / shore.sdf_extent;
    let uv_dist = textureSampleLevel(t_sdf, s_sdf, uv, 0.0).r;
    let world_scale = max(shore.sdf_extent.x, shore.sdf_extent.y);
    return uv_dist * world_scale;
}

// Fade factor based on inscribed circle within the SDF texture.
// Returns 1.0 inside 95% of the radius, fades to 0.0 at 100%.
fn sdf_texture_fade(world_xz: vec2<f32>) -> f32 {
    let center = shore.sdf_origin + shore.sdf_extent * 0.5;
    let dist_from_center = length(world_xz - center);
    let radius = min(shore.sdf_extent.x, shore.sdf_extent.y) * 0.5;
    return 1.0 - smoothstep(radius * 0.95, radius, dist_from_center);
}

fn analytical_sdf_gradient(world_xz: vec2<f32>) -> vec2<f32> {
    // Central differences: scale eps with texel size for smooth gradients
    let eps = max(1.0, shore.sdf_extent.x / 256.0);
    let dx = analytical_sdf(world_xz + vec2(eps, 0.0)) - analytical_sdf(world_xz - vec2(eps, 0.0));
    let dy = analytical_sdf(world_xz + vec2(0.0, eps)) - analytical_sdf(world_xz - vec2(0.0, eps));
    let g = vec2(dx, dy);
    let len = length(g);
    if (len < 0.0001) {
        return vec2(0.0, 1.0);
    }
    return g / len;
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

// 3-octave FBM for organic, non-grid-aligned patterns
fn fbm3(p: vec2<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    // Each octave rotates the coordinate to break grid alignment
    let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    var q = p;
    val += amp * value_noise(q); amp *= 0.5; q = rot * q * 2.0;
    val += amp * value_noise(q); amp *= 0.5; q = rot * q * 2.0;
    val += amp * value_noise(q);
    // Normalize to [0, 1]: weights sum to 0.5 + 0.25 + 0.125 = 0.875
    return val / 0.875;
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

// Returns vec4: xyz = displacement, w = Jacobian (surface fold-over detection)
fn shore_wave_displacement(
    sdf_grad: vec2<f32>,
    base_amplitude: f32,
    world_xz: vec2<f32>,
    depth: f32,
) -> vec4<f32> {
    var disp = vec3(0.0);
    let toward_shore = -sdf_grad;
    let Q = shore.gerstner_steepness;
    let num_waves = i32(shore.gerstner_num_waves);

    // Noisy depth: simulates uneven seabed for varied breaking
    let depth_noise = (fbm3(world_xz * 0.05) - 0.5) * shore.shore_foam_distance;
    let noisy_depth = max(depth + depth_noise, 0.1);
    let envelope = shoal_and_break(noisy_depth);

    // Spatial amplitude variation — wider range for natural look
    let amp_variation = value_noise(world_xz * 0.01) * 0.8 + 0.2; // [0.2, 1.0]

    // Jacobian partial derivatives
    var dxx = 0.0;
    var dzz = 0.0;
    var dxz = 0.0;

    // 3 wave groups for natural wave sets via constructive/destructive interference
    let group_wavelength_scales = array<f32, 3>(1.0, 0.73, 1.35);
    let group_angle_offsets = array<f32, 3>(0.0, 0.15, -0.12);
    let group_amp_scales = array<f32, 3>(1.0, 0.8, 0.6);
    // Each group arrives in slow pulses — creates classic "wave set" pattern
    let set_speeds = array<f32, 3>(0.18, 0.25, 0.14);

    for (var g = 0; g < 3; g++) {
        let base_wl = shore.gerstner_wavelength * group_wavelength_scales[g];
        let base_amp_g = base_amplitude * group_amp_scales[g];
        let group_angle = group_angle_offsets[g];

        // Rotate toward_shore by group angle offset
        let cos_ga = cos(group_angle);
        let sin_ga = sin(group_angle);
        let group_toward = vec2(
            toward_shore.x * cos_ga - toward_shore.y * sin_ga,
            toward_shore.x * sin_ga + toward_shore.y * cos_ga
        );

        // Temporal wave set modulation — groups pulse at different rates
        let set_mod = 0.3 + 0.7 * max(sin(shore.time * set_speeds[g]), 0.0);

        for (var i = 0; i < num_waves; i++) {
            let fi = f32(i);
            let harmonic = fi + 1.0;
            let wavelength = base_wl / harmonic;
            let k = 2.0 * PI / wavelength;
            let w = sqrt(GRAVITY * k) * shore.gerstner_speed;
            // Separate base and shoaled amplitude so steepness grows with shoaling
            let base_amp = base_amp_g / harmonic * amp_variation;
            let amplitude = base_amp * envelope * set_mod;

            // Per-harmonic angular spread
            let perp = vec2(-group_toward.y, group_toward.x);
            let spread_angle = (fi - f32(num_waves - 1) * 0.5) * 0.12;
            let dir = normalize(group_toward * cos(spread_angle) + perp * sin(spread_angle));

            // Along-crest envelope: noise breaks up regular wave segments
            let along_crest = dot(perp, world_xz);
            let env_freq = 0.1 / harmonic;
            let crest_noise = (value_noise(world_xz * 0.04 + vec2(fi * 3.7, f32(g) * 2.1)) - 0.5) * 2.0;
            let env = max(sin(along_crest * env_freq + fi * 2.39 + f32(g) * 1.7) + crest_noise, 0.0);

            let phase = dot(dir, world_xz) * k - w * shore.time;
            let sin_p = sin(phase);
            let cos_p = cos(phase);

            // qi uses base (un-shoaled) amplitude — steepness increases with shoaling
            // Normalized per-group only so each wave train crests independently
            let qi = Q / (k * base_amp * f32(num_waves) + 0.001);
            disp.x += amplitude * env * qi * dir.x * cos_p;
            disp.y += amplitude * env * sin_p;
            disp.z += amplitude * env * qi * dir.y * cos_p;

            // Accumulate Jacobian partial derivatives
            let Ak = amplitude * env * qi * k;
            dxx += Ak * dir.x * dir.x * sin_p;
            dzz += Ak * dir.y * dir.y * sin_p;
            dxz += Ak * dir.x * dir.y * sin_p;
        }
    }

    let J = (1.0 - dxx) * (1.0 - dzz) - dxz * dxz;

    return vec4(disp, J);
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
    let toward_shore = -sdf_grad;

    // Noisy depth (must match displacement)
    let depth_noise = (fbm3(world_xz * 0.05) - 0.5) * shore.shore_foam_distance;
    let noisy_depth = max(depth + depth_noise, 0.1);
    let envelope = shoal_and_break(noisy_depth);

    // Must match displacement
    let amp_variation = value_noise(world_xz * 0.01) * 0.8 + 0.2;

    // 3 wave groups (must match displacement)
    let group_wavelength_scales = array<f32, 3>(1.0, 0.73, 1.35);
    let group_angle_offsets = array<f32, 3>(0.0, 0.15, -0.12);
    let group_amp_scales = array<f32, 3>(1.0, 0.8, 0.6);
    let set_speeds = array<f32, 3>(0.18, 0.25, 0.14);

    for (var g = 0; g < 3; g++) {
        let base_wl = shore.gerstner_wavelength * group_wavelength_scales[g];
        let base_amp_g = base_amplitude * group_amp_scales[g];
        let group_angle = group_angle_offsets[g];

        let cos_ga = cos(group_angle);
        let sin_ga = sin(group_angle);
        let group_toward = vec2(
            toward_shore.x * cos_ga - toward_shore.y * sin_ga,
            toward_shore.x * sin_ga + toward_shore.y * cos_ga
        );

        // Temporal wave set modulation (must match displacement)
        let set_mod = 0.3 + 0.7 * max(sin(shore.time * set_speeds[g]), 0.0);

        for (var i = 0; i < num_waves; i++) {
            let fi = f32(i);
            let harmonic = fi + 1.0;
            let wavelength = base_wl / harmonic;
            let k = 2.0 * PI / wavelength;
            let w = sqrt(GRAVITY * k) * shore.gerstner_speed;
            // Must match displacement: separate base and shoaled amplitude
            let base_amp = base_amp_g / harmonic * amp_variation;
            let amplitude = base_amp * envelope * set_mod;

            let perp = vec2(-group_toward.y, group_toward.x);
            let spread_angle = (fi - f32(num_waves - 1) * 0.5) * 0.12;
            let dir = normalize(group_toward * cos(spread_angle) + perp * sin(spread_angle));

            // Along-crest envelope (must match displacement)
            let along_crest = dot(perp, world_xz);
            let env_freq = 0.1 / harmonic;
            let crest_noise = (value_noise(world_xz * 0.04 + vec2(fi * 3.7, f32(g) * 2.1)) - 0.5) * 2.0;
            let env = max(sin(along_crest * env_freq + fi * 2.39 + f32(g) * 1.7) + crest_noise, 0.0);

            // Clean traveling wave — must match displacement
            let phase = dot(dir, world_xz) * k - w * shore.time;
            let dy_dphase = amplitude * env * cos(phase) * k;

            dydx += dy_dphase * dir.x;
            dydz += dy_dphase * dir.y;
        }
    }

    return normalize(vec3(-dydx, 1.0, -dydz));
}

// Per-pixel Jacobian for pixel-perfect foam in fragment shader.
// Re-evaluates all Gerstner waves but only computes Jacobian partials (no displacement).
fn shore_gerstner_jacobian(
    sdf_grad: vec2<f32>,
    base_amplitude: f32,
    world_xz: vec2<f32>,
    depth: f32,
) -> f32 {
    let toward_shore = -sdf_grad;
    let Q = shore.gerstner_steepness;
    let num_waves = i32(shore.gerstner_num_waves);

    // Noisy depth (must match displacement)
    let depth_noise = (fbm3(world_xz * 0.05) - 0.5) * shore.shore_foam_distance;
    let noisy_depth = max(depth + depth_noise, 0.1);
    let envelope = shoal_and_break(noisy_depth);

    let amp_variation = value_noise(world_xz * 0.01) * 0.8 + 0.2;

    var dxx = 0.0;
    var dzz = 0.0;
    var dxz = 0.0;

    // 3 wave groups (must match displacement)
    let group_wavelength_scales = array<f32, 3>(1.0, 0.73, 1.35);
    let group_angle_offsets = array<f32, 3>(0.0, 0.15, -0.12);
    let group_amp_scales = array<f32, 3>(1.0, 0.8, 0.6);
    let set_speeds = array<f32, 3>(0.18, 0.25, 0.14);

    for (var g = 0; g < 3; g++) {
        let base_wl = shore.gerstner_wavelength * group_wavelength_scales[g];
        let base_amp_g = base_amplitude * group_amp_scales[g];
        let group_angle = group_angle_offsets[g];

        let cos_ga = cos(group_angle);
        let sin_ga = sin(group_angle);
        let group_toward = vec2(
            toward_shore.x * cos_ga - toward_shore.y * sin_ga,
            toward_shore.x * sin_ga + toward_shore.y * cos_ga
        );

        // Temporal wave set modulation (must match displacement)
        let set_mod = 0.3 + 0.7 * max(sin(shore.time * set_speeds[g]), 0.0);

        for (var i = 0; i < num_waves; i++) {
            let fi = f32(i);
            let harmonic = fi + 1.0;
            let wavelength = base_wl / harmonic;
            let k = 2.0 * PI / wavelength;
            let w = sqrt(GRAVITY * k) * shore.gerstner_speed;
            // Must match displacement: separate base and shoaled amplitude
            let base_amp = base_amp_g / harmonic * amp_variation;
            let amplitude = base_amp * envelope * set_mod;

            let perp = vec2(-group_toward.y, group_toward.x);
            let spread_angle = (fi - f32(num_waves - 1) * 0.5) * 0.12;
            let dir = normalize(group_toward * cos(spread_angle) + perp * sin(spread_angle));

            let along_crest = dot(perp, world_xz);
            let env_freq = 0.1 / harmonic;
            let crest_noise = (value_noise(world_xz * 0.04 + vec2(fi * 3.7, f32(g) * 2.1)) - 0.5) * 2.0;
            let env = max(sin(along_crest * env_freq + fi * 2.39 + f32(g) * 1.7) + crest_noise, 0.0);

            let phase = dot(dir, world_xz) * k - w * shore.time;
            let sin_p = sin(phase);

            // Must match displacement qi
            let qi = Q / (k * base_amp * f32(num_waves) + 0.001);
            let Ak = amplitude * env * qi * k;
            dxx += Ak * dir.x * dir.x * sin_p;
            dzz += Ak * dir.y * dir.y * sin_p;
            dxz += Ak * dir.x * dir.y * sin_p;
        }
    }

    return (1.0 - dxx) * (1.0 - dzz) - dxz * dxz;
}


struct OceanVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) original_xz: vec2<f32>,  // Original world XZ before displacement (for UV calculation)
    @location(2) lod_factors: vec3<f32>,  // LOD scale factors for each cascade
    @location(3) jacobian: f32,
    @location(4) shore_blend: f32,        // 0 = pure FFT, 1 = pure Gerstner
    @location(5) sdf_distance: f32,       // World-space SDF distance for fragment foam
    @location(6) shore_jacobian: f32,     // Jacobian from Gerstner waves
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

    // Inscribed circle fade: 1.0 inside 95% of radius, fades to 0 at texture edge
    let tex_fade = sdf_texture_fade(original_xz);

    // shore_blend: 0 in open ocean, 1 near shore.
    // Combines SDF-distance blend with inscribed circle fade to guarantee
    // shore effects reach zero before the texture boundary.
    var shore_blend = 0.0;
    if (sdf_world_dist > 0.0) {
        let sdf_blend = 1.0 - smoothstep(shore.blend_end, shore.blend_start, sdf_world_dist);
        shore_blend = sdf_blend * tex_fade;
    }

    // Land (inside island): no displacement at all
    // Only apply within the inscribed circle where SDF values are trustworthy.
    var blended_displacement = total_displacement;
    var shore_wave_h = 0.0;
    if (sdf_world_dist <= 0.0 && tex_fade > 0.0) {
        blended_displacement = vec3(0.0);
    } else {
        // Transition zone: FFT fully fades to 0 at shore_blend=1.0,
        // replaced entirely by shore sine waves near coast
        if (shore_blend > 0.001) {
            let grad = analytical_sdf_gradient(original_xz);
            let depth = sdf_to_depth(sdf_world_dist);

            let shore_result = shore_wave_displacement(grad, shore.gerstner_amplitude, original_xz, depth);
            let shore_disp = shore_result.xyz;
            shore_wave_h = shore_result.w; // Jacobian from Gerstner waves
            // FFT vertical stays for surface texture; horizontal goes to 0
            // to prevent FFT choppiness fighting shore-ward Gerstner motion
            let fft_v_atten = mix(1.0, shore.shore_foam_band_freq, shore_blend);
            let fft_atten_disp = vec3(
                total_displacement.x * (1.0 - shore_blend),
                total_displacement.y * fft_v_atten,
                total_displacement.z * (1.0 - shore_blend)
            );
            blended_displacement = fft_atten_disp + shore_disp * shore_blend;
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
    out.shore_jacobian = shore_wave_h;

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

      // FFT derivatives attenuate in shore zone but retain a minimum contribution
      let fft_deriv_weight = mix(1.0, shore.shore_foam_band_freq, mesh.shore_blend);
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

    // Add FFT foam as highlights (uses foam color from params)
    ocean_color = ocean_color + params.foam_color * foam_mask * 0.8;

    // Shore foam: Jacobian-based breaking detection per pixel
    // Applied directly to ocean_color — bypasses FFT noise mask which would suppress it
    if (mesh.shore_blend > 0.001) {
        let grad = analytical_sdf_gradient(mesh.original_xz);
        let frag_depth = sdf_to_depth(analytical_sdf(mesh.original_xz));
        let J = shore_gerstner_jacobian(grad, shore.gerstner_amplitude, mesh.original_xz, frag_depth);

        // Foam where wave is steepening toward breaking (J < ~0.3)
        // J=1 is flat, J=0 is about to fold, J<0 is folded over
        let foam_threshold = 0.3;
        let break_intensity = saturate((foam_threshold - J) * params.foam_multiplier);

        // Multi-scale foam texture for organic breakup
        let foam_uv = mesh.original_xz * params.foam_tile_scale * 0.03;
        let foam_fine = textureSample(t_foam, s_ocean, foam_uv * 3.0).r;
        let foam_med = textureSample(t_foam, s_ocean, foam_uv).r;
        let foam_broad = textureSample(t_foam, s_ocean, foam_uv * 0.3).r;

        // Procedural noise for variation independent of texture tiling
        let foam_proc = fbm3(mesh.original_xz * 0.08 + shore.time * 0.05);

        // Combined: broad patches modulate medium detail with fine breakup
        let foam_pattern = foam_broad * (foam_med * 0.7 + foam_fine * 0.3) * (0.5 + foam_proc * 0.5);

        // Foam in organic patches — stronger breaking pushes more through the noise threshold
        let shore_foam = break_intensity * smoothstep(0.1, 0.35, foam_pattern + break_intensity * 0.4);

        ocean_color = ocean_color + params.foam_color * shore_foam * shore.shore_foam_intensity;
    }

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
