#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var t_displacement: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1)
var t_derivatives: texture_2d<f32>;

@group(#{MATERIAL_BIND_GROUP}) @binding(2)
var s_displacement: sampler;

const LENGTH_SCALE = vec3<f32>(500.0, 85.0, 10.0);

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    /* --- Model → World transform (Bevy helper) ------------------- */
    var model = mesh_functions::get_world_from_local(in.instance_index);
    var world_pos = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 1.0)
    );


    let ocean_uv_0 = world_pos.xz / LENGTH_SCALE.x;
    let ocean_uv_2 = world_pos.xz / LENGTH_SCALE.z;
    var displacement = textureSampleLevel(t_displacement, s_displacement, ocean_uv_2, 0.0) * 1000.;


    world_pos.x = world_pos.x + displacement.x;
    world_pos.y = world_pos.y + displacement.y;
    world_pos.z = world_pos.z + displacement.z;


    /* --- Fill the required output struct ------------------------- */
    var out : VertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);
    /* Pass-through you may need later */
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        in.normal, in.instance_index
    );
    out.uv = in.uv;

    return out;
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // output the color directly
    return vec4(0.0,0.0,1.0, 1.0);
}

