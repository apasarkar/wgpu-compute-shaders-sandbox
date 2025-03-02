"""
Simple compute example that performs basic matrix-vector multiplication.

Uses linear arrays in storage buffers to represent matrices of arbitrary size since
the wgsl standard library only supports matrix multiplication upto 4x4 matrices.
"""

import numpy as np
from wgpu.utils.compute import compute_with_buffers

# define matrix and vector dimensions
m, n = 13, 20

A_shape = (m, n)
v_shape = (n, 1)

# create random matrix and vector
A = np.random.rand(A_shape[0] * A_shape[1]).astype(np.float32).reshape(A_shape, order="C")
v = np.random.rand(v_shape[0]).astype(np.float32).reshape(v_shape, order="C")

# define bindings
bindings = {
    0: A,
    1: v,
    3: np.array(A_shape, dtype=np.uint32),
}

# run shader
out = compute_with_buffers(
    input_arrays=bindings,
    output_arrays={2: (np.prod((m, 1)), "f")},
    shader=open(f"./matvec.wgsl").read(),
    n=(n, m, 1)  # n cols across "x dimension", m rows across "y dimension"
)

# get output
c = np.frombuffer(out[2], dtype=np.float32).reshape((m, 1))

# check that results are the same as numpy, we can expect 7 decimal precision
print(f"relative error: \n{np.linalg.norm(A @ v - c, ord='fro') / np.linalg.norm(A @ v, ord='fro')}\n")
