"""
Simple compute example that performs basic matrix multiplication
"""

import numpy as np
from wgpu.utils.compute import compute_with_buffers

# define matrix shapes
m, k, n = 100, 200, 100

# doesn't work, bug in shader
# m, k, n = 200, 200, 100

A_shape = (m, k)
B_shape = (k, n)

# create random matrices
A = np.random.rand(A_shape[0] * A_shape[1]).astype(np.float32).reshape(A_shape, order="C")
B = np.random.rand(B_shape[0] * B_shape[1]).astype(np.float32).reshape(B_shape, order="C")

# define bindings
bindings = {
    0: A,
    1: B,
    3: np.array(A_shape, dtype=np.uint32),
    4: np.array(B_shape, dtype=np.uint32),
}

# run shader
out = compute_with_buffers(
    input_arrays=bindings,
    output_arrays={2: (np.prod((m, n)), "f")},
    shader=open(f"./matmul_simple.wgsl").read(),
    n=(m, n, 1)
)

C = np.frombuffer(out[2], dtype=np.float32).reshape((m, n))

print(np.allclose(A @ B, C))
print(np.linalg.norm(A @ B - C, ord="fro") / np.linalg.norm(A @ B, ord="fro"))
print(A @ B - C)
print(C)
