@group(0) @binding(0)
var<storage, read> A: array<f32>;
@group(0) @binding(1)
var<storage, read> v: array<f32>;
@group(0) @binding(2)
var<storage, read_write> c: array<f32>;

@group(0) @binding(3)
var<storage, read> A_shape: array<u32>;


fn get_1d_index(row_ix: u32, col_ix: u32, n_cols: u32) -> u32 {
    // get the 1D index in the array which corresponds
    // to the passed row and column index
    return row_ix * n_cols + col_ix;
}


@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // matrix-vector multiplication
    // A ∈ R^(m x n), v ∈ R^(n), Av = C ∈ R^(m)
    // gid.y is "m index", gid.x is "n index"

    // we make these varibles because we cannot pass individual array elements to a function
    // i.e. get_1d_index(A_shape[0]) is not possible
    let m: u32 = A_shape[0];
    let n: u32 = A_shape[1];

    var sum: f32 = 0.0;

    // computes one element of c by taking the dot product of the ith row of A with v
    for (var i: u32 = 0; i < n; i++) {
        // product of row = gid.y, col = i of A and row = i of v
        sum = sum + A[get_1d_index(gid.y, i, n)] * v[i];
    }

    c[gid.y] = sum;

    return;
}
