@group(0) @binding(0)
var tex_rgba: texture_2d<f32>;

@group(0) @binding(1)
var tex_y: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var tex_uv: texture_storage_2d<rg32float, write>;

override group_size_x: u32;
override group_size_y: u32;

@compute @workgroup_size(group_size_x, group_size_y)
fn computeMain(@builtin(global_invocation_id) gid: vec3<u32>) {

    var x = gid.x;
    var y = gid.y;

    input_pixel = textureLoad(in_tex, vec2(x, y), 0);

    var output_y: f32 = 0.299 * input_pixel.r + 0.587 * texel.g + 0.114;

    textureStore(tex_y, vec2(x, y), output_y);
}
