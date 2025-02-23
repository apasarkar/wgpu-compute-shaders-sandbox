"""
Example compute shader that does ... nothing but copy a value from one
buffer into another.
"""

import wgpu
from wgpu.utils.compute import compute_with_buffers  # Convenience function
import numpy as np

# %% Shader and data

shader_source = open("./qr.wgsl").read()

# %% The long version using the wgpu API

# %% Create device
# Create device and shader object
device = wgpu.utils.get_default_device()

# Show all available adapters
adapters = wgpu.gpu.enumerate_adapters()
for a in adapters:
    print(a.summary)


# %%
cshader = device.create_shader_module(code=shader_source)

# Create buffer objects, input buffer is mapped.
data = np.random.rand(5, 5).astype(np.float32)
buffer1 = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
buffer2 = device.create_buffer_with_data(data=np.zeros(data.shape, dtype=np.float32), usage=wgpu.BufferUsage.STORAGE)


buffer3 = device.create_buffer(
    size=data.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
)

# Setup layout and bindings
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.read_only_storage,
        },
    },
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]
bindings = [
    {
        "binding": 0,
        "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
    },
    {
        "binding": 2,
        "resource": {"buffer": buffer3, "offset": 0, "size": buffer3.size},
    },
]

# Put everything together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create and run the pipeline
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(1, 1, 1)  # x y z
compute_pass.end()
device.queue.submit([command_encoder.finish()])

# Read result
# result = buffer2.read_data().cast("i")
out = device.queue.read_buffer(buffer2).cast("i")
result = out.tolist()
print(result)
assert result == list(range(20))
