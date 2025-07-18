import numpy as np


block_size = 8

I, J = np.meshgrid(np.arange(block_size), np.arange(block_size), indexing="ij")

# fig = fpl.Figure(shape=(block_size, block_size), size=(1200, 1200), controller_ids="sync")


# def generate_basis():
zigzag = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [2, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [2, 1],
        [3, 0],
        [4, 0],
        [3, 1],
        [2, 2],
        [1, 3],
        [0, 4],
        [0, 5],
        [1, 4],
        [2, 3],
        [3, 2],
        [4, 1],
        [5, 0],
        [6, 0],
        [5, 1],
        [4, 2],
        [3, 3],
        [2, 4],
        [1, 5],
        [0, 6],
        [0, 7],
        [1, 6],
        [2, 5],
        [3, 4],
        [4, 3],
        [5, 2],
        [6, 1],
        [7, 0],
        [7, 1],
        [6, 2],
        [5, 3],
        [4, 4],
        [3, 5],
        [2, 6],
        [1, 7],
        [2, 7],
        [3, 6],
        [4, 5],
        [5, 4],
        [6, 3],
        [7, 2],
        [7, 3],
        [6, 4],
        [5, 5],
        [4, 6],
        [3, 7],
        [4, 7],
        [5, 6],
        [6, 5],
        [7, 4],
        [7, 5],
        [6, 6],
        [5, 7],
        [6, 7],
        [7, 6],
        [7, 7],
    ]
)

dct_basis = np.zeros((block_size**2, block_size, block_size), dtype=np.float32)

for i, (u, v) in enumerate(zigzag):
    c = (
        (2 / block_size)
        * np.cos((2 * I + 1) * u * np.pi / (2 * block_size))
        * np.cos((2 * J + 1) * v * np.pi / (2 * block_size))
    )
    c /= np.linalg.norm(c)

    dct_basis[i] = c
