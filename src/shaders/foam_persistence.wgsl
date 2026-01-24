// Foam persistence compute shader
// Updates foam texture with exponential decay while accumulating new foam from Jacobian

@group(0) @binding(0)
var displacement_texture: texture_storage_2d_array<rgba32float, read>;

@group(0) @binding(1)
var foam_persistence: texture_storage_2d_array<r32float, read_write>;

struct Parameters {
    decay_rate: f32,  // How quickly foam fades (e.g., 0.95 = slow decay, 0.8 = fast decay)
    foam_spawn_threshold: f32,  // Jacobian threshold for spawning foam
    foam_spawn_strength: f32,   // How much foam to add when spawning
    delta_time: f32,
};

var<push_constant> params: Parameters;

@compute @workgroup_size(16, 16)
fn update_foam(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let coords = vec2<i32>(id.xy);
    let layer = id.z;

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

    textureStore(foam_persistence, coords, layer, vec4<f32>(new_foam, 0.0, 0.0, 1.0));
}
