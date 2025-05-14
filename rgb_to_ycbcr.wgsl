@group(0) @binding(0)
var tex_rgba: texture_2d<f32>;

// note that we use r32float and write for the output textures
@group(0) @binding(1)
var tex_y: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var tex_cbcr: texture_storage_2d<rg32float, write>;

@group(0) @binding(3)
var chroma_sampler: sampler;

// block size
override group_size_x: u32;
override group_size_y: u32;


@compute @workgroup_size(group_size_x, group_size_y)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    // wid.xy is the workgroup invocation ID
    // to get the starting (x, y) texture coordinate for a given (8, 8) block we must multiply by workgroup size
    // Example:
    //  workgroup invocation id (0, 0) becomes texture coord (0, 0)
    //  workgroup invocation id (1, 0) becomes texture coord (8, 0) (block size is 8x8)
    //  workgroup invocation id (1, 1) becomes texture coord (8, 8)
    // We can iterate through pixels within this block by just adding to this starting (x, y) position
    // upto the max position which is (start_x + group_size_x, start_y + group_size_y)

    // start and stop indices for this block
    let start = wid.xy * vec2u(group_size_x, group_size_y);
    let stop = start + vec2u(group_size_x, group_size_y);

    // write luminance for each pixel in this block
    for (var x: u32 = start.x; x < stop.x; x++) {
        for (var y: u32 = start.y; y < stop.y; y++) {
            let pos = vec2u(x, y);

            // read array element i.e. "pixel" value
            var px: vec4f = textureLoad(tex_rgba, pos, 0);

            // create luminance channel by converting to grayscale
            var L: f32 = (0.299 * px.r + 0.587 * px.g + 0.114 * px.b);

            // store luma channel
            textureStore(tex_y, pos, vec4<f32>(L, 0, 0, 0));
        }
    }

    // chroma subsampling
    for (var x: u32 = start.x; x < stop.x; x += 2) {
        for (var y: u32 = start.y; y < stop.y; y += 2) {
            // convert to normalized uv coords for sampler
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
