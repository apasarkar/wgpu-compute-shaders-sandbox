@group(0) @binding(0) var tex_rgba: texture_2d<f32>;

@group(0) @binding(1)
var tex_y: texture_storage_2d<r32float, write>;

//@group(0) @binding(2)
//var tex_uv: texture_storage_2d<rg32float, write>;

//override group_size_x: u32;
//override group_size_y: u32;

//const

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {

    var x = gid.x;
    var y = gid.y;

    var input_pixel: vec4f = textureLoad(tex_rgba, vec2(x, y), 0);

    var output_y: f32 = 0.299 * input_pixel.r + 0.587 * input_pixel.g + 0.114 * input_pixel.b;

    textureStore(tex_y, vec2(x, y), vec4<f32>(output_y, 0, 0, 0));
}
