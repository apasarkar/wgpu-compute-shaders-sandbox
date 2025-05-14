import numpy as np
import wgpu
import imageio.v3 as iio
import fastplotlib as fpl


def size_from_array(data, dim):
    # copied from pygfx
    # Check if shape matches dimension
    shape = data.shape

    if len(shape) not in (dim, dim + 1):
        raise ValueError(
            f"Can't map shape {shape} on {dim}D tex. Maybe also specify size?"
        )
    # Determine size based on dim and shape
    if dim == 1:
        return shape[0], 1, 1
    elif dim == 2:
        return shape[1], shape[0], 1
    else:  # dim == 3:
        return shape[2], shape[1], shape[0]


# get example image, convert to rgba
image = iio.imread("imageio:astronaut.png")
image_rgba = np.zeros((*image.shape[:-1], 4), dtype=np.uint8)
image_rgba[..., :-1] = image
image_rgba[..., -1] = 255

# wgpu texture size is (width, height) instead of (rows, cols) for whatever reason
rgba_size = size_from_array(image_rgba, dim=2)

# output
y_size = size_from_array(image_rgba[:, :, 0], dim=2)
uv_size = size_from_array(image_rgba[::2, ::2, 0], dim=2)

# create device
device: wgpu.GPUDevice = wgpu.utils.get_default_device()

# create texture
texture_rgb = device.create_texture(
    label="rgba",
    size=rgba_size,
    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    format=wgpu.TextureFormat.rgba8unorm,
    mip_level_count=1,
    sample_count=1,
)

device.queue.write_texture(
    {
        "texture": texture_rgb,
        "mip_level": 0,
        "origin": (0, 0, 0),
    },
    image_rgba,
    {
        "offset": 0,
        "bytes_per_row": image.shape[0] * 4,
    },
    rgba_size
)

texture_y = device.create_texture(
    label="y",
    size=y_size,
    # use as storage texture since we do not need to sample it
    usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.STORAGE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    # NOTE: we cannot use r8unorm for storage textures!!
    format=wgpu.TextureFormat.r32float,
    mip_level_count=1,
    sample_count=1,
)

texture_uv = device.create_texture(
    label="uv",
    size=uv_size,
    # use as storage texture since we do not need to sample it
    usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.STORAGE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    # NOTE: we cannot use r8unorm for storage textures!!
    format=wgpu.TextureFormat.rg32float,
    mip_level_count=1,
    sample_count=1,
)

with open("./rgb_to_ycbcr.wgsl", "r") as f:
    shader_src = f.read()

shader_module = device.create_shader_module(code=shader_src)

pipeline: wgpu.GPUComputePipeline = device.create_compute_pipeline(
    layout="auto",
    compute={"module": shader_module, "entry_point": "main"}
)

# binding_layouts = [
#     {
#         "binding": 0,
#         "visibility": wgpu.ShaderStage.COMPUTE,
#         "",
#     },
#     {
#         "binding": 1,
#         "visibility": wgpu.ShaderStage.COMPUTE,
#         "buffer": {
#             "type": wgpu.BufferBindingType.storage,
#         },
#     },
# ]

bindings = [
    {
        "binding": 0,
        "resource": texture_rgb.create_view()
    },
    {
        "binding": 1,
        "resource": texture_y.create_view()
    },
    # {
    #     "binding": 2,
    #     "resource": texture_uv.create_view()
    # }
]

layout = pipeline.get_bind_group_layout(0)
bind_group = device.create_bind_group(layout=layout, entries=bindings)

workgroups = np.ceil(np.asarray(image.shape[:2]) / 16).astype(int)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(*workgroups, 1)  # x y z
compute_pass.end()
device.queue.submit([command_encoder.finish()])

y_buffer = device.queue.read_texture(
    source={
        "texture": texture_y,
        "origin": (0, 0, 0),
        "mip_level": 0,
    },
    data_layout={
        "offset": 0,
        "bytes_per_row": image.shape[1] * 4,
        # "rows_per_image": image.shape[0],
    },
    size=size_from_array(image[:, :, 0], dim=2)
).cast("f")

Y = np.frombuffer(y_buffer, dtype=np.float32).reshape(image.shape[:2])

iw = fpl.ImageWidget(Y, cmap="gray")
iw.show()

fpl.loop.run()
