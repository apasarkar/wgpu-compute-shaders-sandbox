@group(0) @binding(0)
var<storage, read> A: array<f32>;
@group(0) @binding(1)
var<storage, read> B: array<f32>;
@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

// Storage buffers holding the shapes of matrices A, B, and C.
// Each is assumed to contain two u32 values:
//   [0]: number of rows
//   [1]: number of columns
// For Strassen’s algorithm we assume all matrices are square (N x N).
@group(0) @binding(3)
var<storage, read> A_shape: array<u32>;
@group(0) @binding(4)
var<storage, read> B_shape: array<u32>;
@group(0) @binding(5)
var<storage, read> C_shape: array<u32>;

// ----- Helper function -----
// Compute the 1D index for a (row, col) in a row‑major matrix with given width.
fn index(row: u32, col: u32, width: u32) -> u32 {
    return row * width + col;
}

// ----- The compute shader -----
// We choose a workgroup size (e.g. 16×16 threads) so that each invocation computes one element.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // For our multiplication, we assume A, B, and C are square with the same dimension.
    // Read the dimension from A_shape[0] (which should equal A_shape[1]).
    let N: u32 = A_shape[0];

    // For simplicity, if the matrix is small, use standard (naive) multiplication.
    if (N <= 64u) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0u; k < N; k = k + 1u) {
            sum = sum + A[index(gid.y, k, N)] * B[index(k, gid.x, N)];
        }
        C[index(gid.y, gid.x, N)] = sum;
        return;
    }

    // ----- One level of Strassen’s Algorithm -----
    // Partition each matrix into four N/2×N/2 submatrices:
    //   A = [ A11  A12 ]
    //       [ A21  A22 ]
    //   B = [ B11  B12 ]
    //       [ B21  B22 ]
    //
    // Let half = N/2.
    let half = N / 2u;

    // Our threads are launched over the full N×N C.
    // Each thread computes one element of one of the four quadrants.
    // Compute the “local” coordinates within a quadrant.
    let i = gid.y % half;
    let j = gid.x % half;

    // --- M1: M1 = (A11 + A22) * (B11 + B22) ---
    var M1: f32 = 0.0;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let a_val = A[index(i, k, N)] + A[index(i + half, k + half, N)];
        let b_val = B[index(k, j, N)] + B[index(k + half, j + half, N)];
        M1 = M1 + a_val * b_val;
    }

    // --- M2: M2 = (A21 + A22) * B11 ---
    var M2: f32 = 0.0;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let a_val = A[index(i + half, k, N)] + A[index(i + half, k + half, N)];
        let b_val = B[index(k, j, N)];
        M2 = M2 + a_val * b_val;
    }

    // --- M3: M3 = A11 * (B12 - B22) ---
    var M3: f32 = 0.0;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let a_val = A[index(i, k, N)];
        let b_val = B[index(k, j + half, N)] - B[index(k + half, j + half, N)];
        M3 = M3 + a_val * b_val;
    }

    // --- M4: M4 = A22 * (B21 - B11) ---
    var M4: f32 = 0.0;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let a_val = A[index(i + half, k + half, N)];
        let b_val = B[index(k + half, j, N)] - B[index(k, j, N)];
        M4 = M4 + a_val * b_val;
    }

    // --- M5: M5 = (A11 + A12) * B22 ---
    var M5: f32 = 0.0;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let a_val = A[index(i, k, N)] + A[index(i, k + half, N)];
        let b_val = B[index(k + half, j + half, N)];
        M5 = M5 + a_val * b_val;
    }

    // --- M6: M6 = (A21 - A11) * (B11 + B12) ---
    var M6: f32 = 0.0;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let a_val = A[index(i + half, k, N)] - A[index(i, k, N)];
        let b_val = B[index(k, j, N)] + B[index(k, j + half, N)];
        M6 = M6 + a_val * b_val;
    }

    // --- M7: M7 = (A12 - A22) * (B21 + B22) ---
    var M7: f32 = 0.0;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let a_val = A[index(i, k + half, N)] - A[index(i + half, k + half, N)];
        let b_val = B[index(k + half, j, N)] + B[index(k + half, j + half, N)];
        M7 = M7 + a_val * b_val;
    }

    // --- Combine the M's to get the final quadrant of C ---
    // Strassen recombination formulas:
    //   C11 = M1 + M4 - M5 + M7
    //   C12 = M3 + M5
    //   C21 = M2 + M4
    //   C22 = M1 - M2 + M3 + M6
    var result: f32 = 0.0;
    if (gid.y < half && gid.x < half) {
        // Top-left quadrant: C11.
        result = M1 + M4 - M5 + M7;
    } else if (gid.y < half && gid.x >= half) {
        // Top-right quadrant: C12.
        result = M3 + M5;
    } else if (gid.y >= half && gid.x < half) {
        // Bottom-left quadrant: C21.
        result = M2 + M4;
    } else {
        // Bottom-right quadrant: C22.
        result = M1 - M2 + M3 + M6;
    }

    // Write the computed value to matrix C.
    C[index(gid.y, gid.x, N)] = result;
}
