@group(0) @binding(0)
var<storage, read> input_lengths: array<u32>;

@group(0) @binding(1)
var<storage,read_write> out_lengths: array<u32>;

@group(0) @binding(2)
var<storage,read_write> net_length: array<u32>;


// THIS ASSUMES WE ARE DOING 8 x 8 blockwise RLE on a 512 x 512 image
const n_blocks = u32(4096);  // m_rows

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wg_id : vec3<u32>,
        @builtin(num_workgroups) n_wkgp: vec3<u32>) {

    out_lengths[0] = 0;
    for(var x: u32 = 1; x < n_blocks; x++) {
        out_lengths[x] = out_lengths[x-1] + input_lengths[x-1];
    }

    net_length[0] = out_lengths[n_blocks - 1] + input_lengths[n_blocks - 1];
}