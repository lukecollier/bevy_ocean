// Foam persistence compute shader
// Updates foam texture with exponential decay while accumulating new foam from Jacobian
// Generates mipmap levels 1 and 2 directly from the computed foam values

@group(0) @binding(0)
var displacement_texture: texture_storage_2d_array<rgba32float, read>;

// Foam persistence mip level 0 (read-write for decay calculation)
@group(0) @binding(1)
var foam_persistence: texture_storage_2d_array<r32float, read_write>;

// Foam persistence mip level 1 (write only)
@group(0) @binding(2)
var foam_persistence_mip1: texture_storage_2d_array<r32float, write>;

// Foam persistence mip level 2 (write only)
@group(0) @binding(3)
var foam_persistence_mip2: texture_storage_2d_array<r32float, write>;

// Foam persistence mip level 3 (write only)
@group(0) @binding(4)
var foam_persistence_mip3: texture_storage_2d_array<r32float, write>;

struct Parameters {
    decay_rate: f32,  // How quickly foam fades (e.g., 0.95 = slow decay, 0.8 = fast decay)
    foam_spawn_threshold: f32,  // Jacobian threshold for spawning foam
    foam_spawn_strength: f32,   // How much foam to add when spawning
    delta_time: f32,
};

var<push_constant> params: Parameters;

fn compute_foam_at(coords: vec2<i32>, layer: u32, decay_factor: f32) -> f32 {
    let jacobian = textureLoad(displacement_texture, coords, layer).w;
    let prev_foam = textureLoad(foam_persistence, coords, layer).r;
    let jacobian_foam = saturate((-jacobian + params.foam_spawn_threshold) * params.foam_spawn_strength);
    let decayed_foam = prev_foam * decay_factor;
    return max(decayed_foam, jacobian_foam);
}

@compute @workgroup_size(16, 16, 1)
fn update_foam(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let coords = vec2<i32>(global_id.xy);
    let layer = global_id.z;
    let decay_factor = pow(params.decay_rate, params.delta_time * 60.0);  // Normalized to 60fps base
    let new_foam = compute_foam_at(coords, layer, decay_factor);

    // Write mip 0
    textureStore(foam_persistence, coords, layer, vec4<f32>(new_foam, 0.0, 0.0, 1.0));

    // === Mip 1 (8x8 threads active) ===
    if (local_id.x < 8u && local_id.y < 8u) {
        let origin = vec2<i32>(i32(workgroup_id.x * 16u), i32(workgroup_id.y * 16u));
        let local_base = vec2<i32>(i32(local_id.x) * 2, i32(local_id.y) * 2);

        var sum = 0.0;
        for (var dy = 0; dy < 2; dy = dy + 1) {
            for (var dx = 0; dx < 2; dx = dx + 1) {
                sum = sum + compute_foam_at(origin + local_base + vec2<i32>(dx, dy), layer, decay_factor);
            }
        }

        let avg_foam = sum * 0.25;
        let mip1_coords = vec2<i32>(i32(workgroup_id.x * 8u + local_id.x),
                                     i32(workgroup_id.y * 8u + local_id.y));
        textureStore(foam_persistence_mip1, mip1_coords, layer, vec4<f32>(avg_foam, 0.0, 0.0, 1.0));
    }

    // === Mip 2 (4x4 threads active) ===
    if (local_id.x < 4u && local_id.y < 4u) {
        let origin = vec2<i32>(i32(workgroup_id.x * 16u), i32(workgroup_id.y * 16u));
        let local_base = vec2<i32>(i32(local_id.x) * 4, i32(local_id.y) * 4);

        var sum = 0.0;
        for (var dy = 0; dy < 4; dy = dy + 1) {
            for (var dx = 0; dx < 4; dx = dx + 1) {
                sum = sum + compute_foam_at(origin + local_base + vec2<i32>(dx, dy), layer, decay_factor);
            }
        }

        let avg_foam = sum * 0.0625;
        let mip2_coords = vec2<i32>(i32(workgroup_id.x * 4u + local_id.x),
                                     i32(workgroup_id.y * 4u + local_id.y));
        textureStore(foam_persistence_mip2, mip2_coords, layer, vec4<f32>(avg_foam, 0.0, 0.0, 1.0));
    }

    // === Mip 3 (2x2 threads active) ===
    if (local_id.x < 2u && local_id.y < 2u) {
        let origin = vec2<i32>(i32(workgroup_id.x * 16u), i32(workgroup_id.y * 16u));
        let local_base = vec2<i32>(i32(local_id.x) * 8, i32(local_id.y) * 8);

        var sum = 0.0;
        for (var dy = 0; dy < 8; dy = dy + 1) {
            for (var dx = 0; dx < 8; dx = dx + 1) {
                sum = sum + compute_foam_at(origin + local_base + vec2<i32>(dx, dy), layer, decay_factor);
            }
        }

        let avg_foam = sum * 0.015625;
        let mip3_coords = vec2<i32>(i32(workgroup_id.x * 2u + local_id.x),
                                     i32(workgroup_id.y * 2u + local_id.y));
        textureStore(foam_persistence_mip3, mip3_coords, layer, vec4<f32>(avg_foam, 0.0, 0.0, 1.0));
    }
}
