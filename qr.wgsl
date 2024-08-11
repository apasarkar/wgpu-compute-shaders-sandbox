@group(0) @binding(0)
var<storage,read> A: array<f32>;

@group(0) @binding(1)
var<storage,read_write> Q: array<f32>;

//@group(0) @binding(2)
//var<storage,read_write> R: array<f32>;


//struct Array {
//    shape: vec2<u32>,
//    data: array<f32>,
//}

const n_rows = u32(5);  // m_rows
const n_cols = u32(5);  // n_cols

fn l2(col_ix: u32) -> f32 {
    var sum_squares = f32(0.0);
    var result: f32;
    var offset: u32 = col_ix * n_rows;

    for (var i: u32 = 0; i < n_rows; i++) {
        sum_squares += pow(A[offset + i], 2.0);
    }

    return sqrt(sum_squares);
}




@compute
@workgroup_size(1)
fn main() {
//    // store intermediates
//    var U: array<f32>;
//
//    // first column of U is the same as A
//    for (var i = 0; i < n_rows; i++) {
//        U[i] = A[i];
//    }

    // Graham-Schmidt
    for (var col: u32 = 0; col < n_cols; col++) {
        // get norm of vector in current column
        Q[col] = l2(col);
//        for (var row: u32 = 0; row < m; row++) {
//
//        }
    }
}
