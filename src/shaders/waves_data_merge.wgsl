@group(0) @binding(0)
var amp_dx_dz__dy_dxz_texture: texture_storage_2d<rgba32float, read>;

@group(0) @binding(1)
var amp_dyx_dyz__dxx_dzz_texture: texture_storage_2d<rgba32float, read>;

@group(0) @binding(2)
var out_displacement: texture_storage_2d_array<rgba32float, read_write>;

@group(0) @binding(3)
var out_derivatives: texture_storage_2d_array<rgba32float, write>;

struct Parameters {
  lambda: f32,
  delta_time: f32,
  layer: u32
};

var<push_constant> params: Parameters;

@compute @workgroup_size(16, 16)
fn merge(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let coords = vec2<i32>(id.xy);
    // let layer = id.z;
    let layer = params.layer;
    let l = params.lambda;

    let dx_dz_dy_dxz = textureLoad(amp_dx_dz__dy_dxz_texture, coords);
    let dx_dz = dx_dz_dy_dxz.xy;
    let dy_dxz = dx_dz_dy_dxz.zw;

    let dyx_dyz_dxx_dzz = textureLoad(amp_dyx_dyz__dxx_dzz_texture, coords);
    let dyx_dyz = dyx_dyz_dxx_dzz.xy;
    let dxx_dzz = dyx_dyz_dxx_dzz.zw;

    let displacement = vec3<f32>(l * dx_dz.x, dy_dxz.x, l * dx_dz.y);
    let derivatives = vec4<f32>(dyx_dyz, dxx_dzz * l);

    let jacobian = (1.0 + l * dxx_dzz.x) * (1.0 + l * dxx_dzz.y) - l * l * dy_dxz.y * dy_dxz.y;

    textureStore(
        out_displacement,
        coords,
        layer,
        vec4<f32>(displacement, jacobian),
    );

    textureStore(
        out_derivatives,
        coords,
        layer,
        derivatives,
    );
}

const PI: f32 = 3.14159265358979323846264338;
const sigma = 8.0;

@compute @workgroup_size(16, 16)
fn blur_turbulence(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let coords = vec2<i32>(id.xy);
    let layer = id.z;

    var value = 0.0;
    for (var x = -4; x <= 4; x = x + 1) {
        for (var y = -4; y <= 4; y = y + 1) {
            var ic = vec2<i32>(coords.x + x, coords.y + y);
            let g = 1.0 / sqrt(2.0 * PI * sigma * sigma) * exp(-(f32(x) * f32(x) + f32(y) * f32(y)) / (2.0 * sigma * sigma));
            value = value + g * textureLoad(out_displacement, ic, layer).w;
        }
    }

    textureStore(out_displacement, coords, layer, vec4<f32>(textureLoad(out_displacement, coords, layer).rgb, value));
}
