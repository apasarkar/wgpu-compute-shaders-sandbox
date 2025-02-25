import numpy as np

import wgpu
from wgpu.utils.compute import compute_with_buffers


dtype_mapping = {
    np.int8: "b",
    np.int16: "h",
    np.int32: "i",
    np.float16: "e",
    np.float32: "f",
}


def get_column_indices(col_ix) -> tuple[int, int]:
    start = col_ix * a.shape[0]
    return start, start + a.shape[0]

m, k, n = 16, 16, 16

a_shape = (m, k)
b_shape = (k, n)
c_shape = (m, n)

a = np.arange(0, a_shape[0] * a_shape[1]).astype(np.float32)
b = np.arange(0, b_shape[0] * b_shape[1]).astype(np.float32)

a = a.reshape(a_shape, order="C")
b = b.reshape(b_shape, order="C")

bindings = {
    0: a,
    1: b,
    3: np.array(a_shape),
    4: np.array(b_shape),
    5: np.array(c_shape)
}


out = compute_with_buffers(
    input_arrays=bindings,
    output_arrays={2: (np.prod(c_shape), "f")},
    shader=open(f"./matmul_strassen.wgsl").read(),
    workgroups=(int(c_shape[0] / 16), int(c_shape[1] / 16), 1),
    n=16
)

shader_out = np.frombuffer(out[2], dtype=np.float32).reshape(c_shape)

print(np.allclose(a @ b, shader_out))

print(np.linalg.norm(a @ b - shader_out, ord="fro"))

print(a @ b - shader_out)
