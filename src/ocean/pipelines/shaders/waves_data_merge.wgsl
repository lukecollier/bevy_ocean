// Input textures (arrays - one layer per cascade)
@group(0) @binding(0)
var amp_dx_dz__dy_dxz_texture: texture_storage_2d_array<rgba32float, read>;

@group(0) @binding(1)
var amp_dyx_dyz__dxx_dzz_texture: texture_storage_2d_array<rgba32float, read>;

// Output textures - mip level 0 (arrays)
@group(0) @binding(2)
var out_displacement: texture_storage_2d_array<rgba32float, read_write>;

@group(0) @binding(3)
var out_derivatives: texture_storage_2d_array<rgba32float, write>;

// Output textures - mip level 1
@group(0) @binding(4)
var out_displacement_mip1: texture_storage_2d_array<rgba32float, write>;

@group(0) @binding(5)
var out_derivatives_mip1: texture_storage_2d_array<rgba32float, write>;

// Output textures - mip level 2
@group(0) @binding(6)
var out_displacement_mip2: texture_storage_2d_array<rgba32float, write>;

@group(0) @binding(7)
var out_derivatives_mip2: texture_storage_2d_array<rgba32float, write>;

// Output textures - mip level 3
@group(0) @binding(8)
var out_displacement_mip3: texture_storage_2d_array<rgba32float, write>;

@group(0) @binding(9)
var out_derivatives_mip3: texture_storage_2d_array<rgba32float, write>;

struct Parameters {
  lambda: f32,
  delta_time: f32,
};

var<push_constant> params: Parameters;

struct SampleResult {
  displacement: vec4<f32>,
  derivatives: vec4<f32>,
};

fn sample_at(coords: vec2<i32>, layer: u32) -> SampleResult {
  let l = params.lambda;
  let dx_dz_dy_dxz = textureLoad(amp_dx_dz__dy_dxz_texture, coords, layer);
  let dx_dz = dx_dz_dy_dxz.xy;
  let dy_dxz = dx_dz_dy_dxz.zw;

  let dyx_dyz_dxx_dzz = textureLoad(amp_dyx_dyz__dxx_dzz_texture, coords, layer);
  let dyx_dyz = dyx_dyz_dxx_dzz.xy;
  let dxx_dzz = dyx_dyz_dxx_dzz.zw;

  let displacement = vec3<f32>(l * dx_dz.x, dy_dxz.x, l * dx_dz.y);
  let derivatives = vec4<f32>(dyx_dyz, dxx_dzz * l);
  let jacobian = (1.0 + l * dxx_dzz.x) * (1.0 + l * dxx_dzz.y) - l * l * dy_dxz.y * dy_dxz.y;

  return SampleResult(vec4<f32>(displacement, jacobian), derivatives);
}

@compute @workgroup_size(16, 16, 1)
fn merge(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let coords = vec2<i32>(global_id.xy);
    let layer = global_id.z;
    let sample = sample_at(coords, layer);

    // Write mip 0
    textureStore(out_displacement, coords, layer, sample.displacement);
    textureStore(out_derivatives, coords, layer, sample.derivatives);

    // === Mip 1 (8x8 threads active) ===
    if (local_id.x < 8u && local_id.y < 8u) {
        let origin = vec2<i32>(i32(workgroup_id.x * 16u), i32(workgroup_id.y * 16u));
        let local_base = vec2<i32>(i32(local_id.x) * 2, i32(local_id.y) * 2);

        let sample00 = sample_at(origin + local_base, layer);
        let sample10 = sample_at(origin + local_base + vec2<i32>(1, 0), layer);
        let sample01 = sample_at(origin + local_base + vec2<i32>(0, 1), layer);
        let sample11 = sample_at(origin + local_base + vec2<i32>(1, 1), layer);

        let avg_disp = (sample00.displacement + sample10.displacement +
                        sample01.displacement + sample11.displacement) * 0.25;
        let avg_deriv = (sample00.derivatives + sample10.derivatives +
                         sample01.derivatives + sample11.derivatives) * 0.25;

        let mip1_coords = vec2<i32>(i32(workgroup_id.x * 8u + local_id.x),
                                    i32(workgroup_id.y * 8u + local_id.y));
        textureStore(out_displacement_mip1, mip1_coords, layer, avg_disp);
        textureStore(out_derivatives_mip1, mip1_coords, layer, avg_deriv);
    }

    if (local_id.x < 4u && local_id.y < 4u) {
        let origin = vec2<i32>(i32(workgroup_id.x * 16u), i32(workgroup_id.y * 16u));
        let local_base = vec2<i32>(i32(local_id.x) * 4, i32(local_id.y) * 4);

        var sum_disp = vec4<f32>(0.0);
        var sum_deriv = vec4<f32>(0.0);

        for (var dy = 0; dy < 4; dy = dy + 1) {
            for (var dx = 0; dx < 4; dx = dx + 1) {
                let sample = sample_at(origin + local_base + vec2<i32>(dx, dy), layer);
                sum_disp = sum_disp + sample.displacement;
                sum_deriv = sum_deriv + sample.derivatives;
            }
        }

        let avg_disp_mip2 = sum_disp * 0.0625;
        let avg_deriv_mip2 = sum_deriv * 0.0625;

        let mip2_coords = vec2<i32>(i32(workgroup_id.x * 4u + local_id.x),
                                    i32(workgroup_id.y * 4u + local_id.y));
        textureStore(out_displacement_mip2, mip2_coords, layer, avg_disp_mip2);
        textureStore(out_derivatives_mip2, mip2_coords, layer, avg_deriv_mip2);
    }

    // === Mip 3 (2x2 threads active) ===
    if (local_id.x < 2u && local_id.y < 2u) {
        let origin = vec2<i32>(i32(workgroup_id.x * 16u), i32(workgroup_id.y * 16u));
        let local_base = vec2<i32>(i32(local_id.x) * 8, i32(local_id.y) * 8);

        var sum_disp = vec4<f32>(0.0);
        var sum_deriv = vec4<f32>(0.0);

        for (var dy = 0; dy < 8; dy = dy + 1) {
            for (var dx = 0; dx < 8; dx = dx + 1) {
                let sample = sample_at(origin + local_base + vec2<i32>(dx, dy), layer);
                sum_disp = sum_disp + sample.displacement;
                sum_deriv = sum_deriv + sample.derivatives;
            }
        }

        let avg_disp_mip3 = sum_disp * 0.015625;
        let avg_deriv_mip3 = sum_deriv * 0.015625;

        let mip3_coords = vec2<i32>(i32(workgroup_id.x * 2u + local_id.x),
                                    i32(workgroup_id.y * 2u + local_id.y));
        textureStore(out_displacement_mip3, mip3_coords, layer, avg_disp_mip3);
        textureStore(out_derivatives_mip3, mip3_coords, layer, avg_deriv_mip3);
    }
}

const PI: f32 = 3.14159265358979323846264338;
const sigma = 8.0;

@compute @workgroup_size(16, 16, 1)
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
