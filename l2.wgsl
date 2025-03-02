@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)  // n_rows, n_cols
var<storage, read> shape: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;


fn get_column_indices(col_ix: u32) -> vec2<u32> {
    // returns start, stop indices for the column of `A` at `col_ix`
    // assumes array is in column-major order!!
    var out: vec2<u32>;
    // column index + n_rows
    out[0] = col_ix * shape[0];
    // start + n_rows
    out[1] = out[0] + shape[0];
    return out;
}


fn l2_norm(col_ix: u32) -> f32 {
    // returns the l2 norm for the column of `A` at `col_ix`
    var sum_squares = f32(0.0);
    var ixs: vec2<u32>;
    ixs = get_column_indices(col_ix);
    for (var i: u32 = ixs[0]; i < ixs[1]; i++) {
        sum_squares += pow(A[i], 2.0);
    }
    return sqrt(sum_squares);
}


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    // returns l2 norm of columns of A
    output[index[0]] = l2_norm(index[0]);
}
