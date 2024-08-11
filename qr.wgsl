@group(0) @binding(0)
var<storage, read> A: array<f32>;

//@group(0) @binding(1)
//var<storage, read_write> U: array<f32>;
@group(0) @binding(1)
var<storage,read_write> u: array<f32>;

@group(0) @binding(2)
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


fn array_dot(a: <function,array<f32, 5>>, e: <function,array<f32, 5>>) -> f32{
    result =
    for (var i: u32 = 0; i < 5; i++) {

    }
}


@compute
@workgroup_size(1)
fn main() {
    // store intermediates
    // first column of U is the same as A
    for (var i: u32 = 0; i < n_rows; i++) {
        u[i] = A[i];
    }

    // Graham-Schmidt
    for (var col: u32 = 0; col < n_cols; col++) {
        // first iteration
        if col == 0 {
            u = A[0];
        }

        // next iterations
        else {
            
        }

        // get norm of vector in current column
        u_norm = l2(col);
        var offset = col * n_rows;
        // set row of Q
        for (var row: u32 = 0; row < n_rows; row++) {
            Q[offset + row] = u[offset + row] / u_norm
        }
    }
}
