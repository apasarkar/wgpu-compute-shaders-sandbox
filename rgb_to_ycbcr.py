import numpy as np
import wgpu
import imageio.v3 as iio

# get example image, convert to rgba
image = iio.imread("imageio:astronaut.png")
image_rgba = np.zeros((*image.shape[:-1], 4), dtype=np.uint8)
image_rgba[..., :-1] = image
image_rgba[..., -1] = 255

# wgpu texture size is (width, height) instead of (rows, cols) for whatever reason
rgba_size = (image.shape[1], image.shape[0], 4)

# output
y_size = (image.shape[1], image.shape[0], 1)
uv_size = (*(np.asarray(image.shape[:-1]) / 2).astype(int), 2)

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
        "bytes_per_row": image_rgba.strides[0],
    },
    rgba_size
)

texture_y = device.create_texture(
    label="y",
    size=y_size,
    # use as storage texture since we do not need to sample it
    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.STORAGE_BINDING,
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
    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.STORAGE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    # NOTE: we cannot use r8unorm for storage textures!!
    format=wgpu.TextureFormat.rg32float,
    mip_level_count=1,
    sample_count=1,
)

sampler = device.create_sampler()

with open("./rgb_to_ycbcr.wgsl", "r") as f:
    shader_src = f.read()

pipeline: wgpu.GPUComputePipeline = device.create_compute_pipeline(
    layout="auto",
    compute={"module": shader_src, "entry_point": "main"}
)

# bing_group_layout = [
#     {
#         "binding": 0,
#         "visibility": wgpu.ShaderStage.COMPUTE,
#         "buffer"
#     }
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
{
        "binding": 2,
        "resource": texture_uv.create_view()
    }
]

layout = pipeline.get_bind_group_layout(0)
bind_group = device.create_bind_group(layout=layout, entries=bindings)
