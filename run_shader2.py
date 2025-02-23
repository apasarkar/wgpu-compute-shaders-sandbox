import numpy as np

import wgpu
from wgpu.utils.compute import compute_with_buffers

a = np.arange(12).reshape((3, 4), order="C")


def read_shader(name):
    return open(f"./{name}.wgsl").read()


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

a.T.ravel()[slice(*get_column_indices(3))]

a_shape = (15, 15)
a = np.random.rand(a_shape[0] * a_shape[1]).astype(np.float32)

a.reshape(a_shape, order="C")


out = compute_with_buffers(
    input_arrays={0: a, 1: np.asarray(a_shape, dtype=np.uint32)},
    # output_arrays={2: (*out_shape, dtype_mapping[np.float32]), 1: (a.shape[1], "f")},
    output_arrays={2: (a_shape[1], "f")},
    shader=read_shader("l2"),
    n=a_shape[1]
)

shader_out = np.frombuffer(out[2], dtype=np.float32)

np.linalg.norm(a.reshape(a_shape), ord=2, axis=1)
print(np.allclose(shader_out, np.linalg.norm(a.reshape(a_shape), ord=2, axis=1)))
