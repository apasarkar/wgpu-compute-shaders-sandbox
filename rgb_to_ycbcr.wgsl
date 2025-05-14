@group(0) @binding(0) var tex_rgba: texture_2d<f32>;

@group(0) @binding(1)
var tex_y: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var tex_cbcr: texture_storage_2d<rg32float, write>;

override group_size_x: u32;
override group_size_y: u32;


@compute @workgroup_size(group_size_x, group_size_y)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // read array element i.e. "pixel" value
    var px: vec4f = textureLoad(tex_rgba, gid.xy, 0);

    var y: f32 = 0.299 * px.r + 0.587 * px.g + 0.114 * px.b;

    textureStore(tex_y, gid.xy, vec4<f32>(y, 0, 0, 0));

    var cb: f32 = -0.1687 * px.r - 0.3313 * px.g + 0.5 * px.b + 128;
    var cr: f32 = 0.5 * px.r - 0.4187 * px.g - 0.0813 * px.b + 128;

    textureStore(tex_cbcr, gid.xy, vec4<f32>(cb, cr, 0, 0));
}
