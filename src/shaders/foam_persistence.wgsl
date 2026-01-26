// Foam persistence compute shader
// Updates foam texture with exponential decay while accumulating new foam from Jacobian
// Also generates mipmap level 1 for the foam texture using shared memory

@group(0) @binding(0)
var displacement_texture: texture_storage_2d_array<rgba32float, read>;

// Foam persistence mip level 0 (read-write for decay calculation)
@group(0) @binding(1)
var foam_persistence: texture_storage_2d_array<r32float, read_write>;

// Foam persistence mip level 1 (write only)
@group(0) @binding(2)
var foam_persistence_mip1: texture_storage_2d_array<r32float, write>;

struct Parameters {
    decay_rate: f32,  // How quickly foam fades (e.g., 0.95 = slow decay, 0.8 = fast decay)
    foam_spawn_threshold: f32,  // Jacobian threshold for spawning foam
    foam_spawn_strength: f32,   // How much foam to add when spawning
    delta_time: f32,
};

var<push_constant> params: Parameters;

// Shared memory for mipmap generation (16x16 workgroup, single channel)
var<workgroup> shared_foam: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn update_foam(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let coords = vec2<i32>(global_id.xy);
    let layer = global_id.z;
    let local_idx = local_id.y * 16u + local_id.x;

    // Read current Jacobian from displacement texture (stored in .w component)
    let jacobian = textureLoad(displacement_texture, coords, layer).w;

    // Read previous foam value
    let prev_foam = textureLoad(foam_persistence, coords, layer).r;

    // Calculate foam from current Jacobian
    // Negative Jacobian indicates wave compression/folding where foam forms
    let jacobian_foam = saturate((-jacobian + params.foam_spawn_threshold) * params.foam_spawn_strength);

    // Apply exponential decay based on delta time
    // decay_factor = decay_rate^delta_time for frame-rate independent decay
    let decay_factor = pow(params.decay_rate, params.delta_time * 60.0);  // Normalized to 60fps base
    let decayed_foam = prev_foam * decay_factor;

    // New foam is max of decayed previous foam and new jacobian-based foam
    // This ensures foam persists and doesn't immediately disappear
    let new_foam = max(decayed_foam, jacobian_foam);

    // Write mip 0
    textureStore(foam_persistence, coords, layer, vec4<f32>(new_foam, 0.0, 0.0, 1.0));

    // Store to shared memory for mipmap generation
    shared_foam[local_idx] = new_foam;
    workgroupBarrier();

    // === Mip 1 (8x8 threads active) ===
    if (local_id.x < 8u && local_id.y < 8u) {
        let base = local_id.xy * 2u;
        let idx00 = base.y * 16u + base.x;
        let idx10 = base.y * 16u + base.x + 1u;
        let idx01 = (base.y + 1u) * 16u + base.x;
        let idx11 = (base.y + 1u) * 16u + base.x + 1u;

        let avg_foam = (shared_foam[idx00] + shared_foam[idx10] +
                        shared_foam[idx01] + shared_foam[idx11]) * 0.25;

        let mip1_coords = vec2<i32>(workgroup_id.xy * 8u + local_id.xy);
        textureStore(foam_persistence_mip1, mip1_coords, layer, vec4<f32>(avg_foam, 0.0, 0.0, 1.0));
    }
}
