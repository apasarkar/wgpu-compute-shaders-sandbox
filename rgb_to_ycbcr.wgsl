@group(0) @binding(0) var tex_rgba: texture_2d<f32>;

@group(0) @binding(1)
var tex_y: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var tex_cbcr: texture_storage_2d<rg32float, write>;

@group(0) @binding(3)
var chroma_sampler: sampler;


override group_size_x: u32;
override group_size_y: u32;


@compute @workgroup_size(group_size_x, group_size_y)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let max_x = wid.x * group_size_x;
    let max_y = wid.y * group_size_y;

    // write luminance for each pixel in block
    for (var x: u32 = wid.x; x < max_x; x++) {
        for (var y: u32 = wid.y; y < max_y; y++) {
            let pos = vec2u(x, y);

            // read array element i.e. "pixel" value
            var px: vec4f = textureLoad(tex_rgba, pos, 0);

            // create luminance channel by converting to grayscale
            var L: f32 = 0.299 * px.r + 0.587 * px.g + 0.114 * px.b;

            // store luma channel
            textureStore(tex_y, pos, vec4<f32>(L, 0, 0, 0));
        }
    }

    // chroma subsampling
    for (var x: u32 = wid.x; x < max_x; x += 2) {
        for (var y: u32 = wid.y; y < max_y; y += 2) {
            // convert to normalized coords for sampler
            let coords_sample: vec2f = (vec2f(f32(x), f32(y)) + 0.5) / vec2f(textureDimensions(tex_rgba).xy);

            var px_sample: vec4f = textureSampleLevel(tex_rgba, chroma_sampler, coords_sample, 0.0);

            // create cb, cr channels
            var cb: f32 = (-0.1687 * px_sample.r - 0.3313 * px_sample.g + 0.5 * px_sample.b) + 0.5;
            var cr: f32 = (0.5 * px_sample.r - 0.4187 * px_sample.g - 0.0813 * px_sample.b) + 0.5;
            let pos_out: vec2u = vec2u(x / 2, y / 2);
            textureStore(tex_cbcr, pos_out.xy, vec4<f32>(cb, cr, 0, 0));
        }
    }
}
