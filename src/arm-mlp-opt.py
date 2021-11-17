import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime

batch_size = 1
num_class = 10
image_shape = (1, 28, 28)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.mlp.get_workload(
    num_classes=10, batch_size=batch_size, image_shape=image_shape
)

#Here mod is the function used to perform the mlp operation
print(mod.astext(show_meta_data=False))

#params is the value of the parameters passed to the function in mlp fn in mod above.
print(params)

opt_level = 3
#target = tvm.target.arm_cpu()
config_str = "llvm -keys=arm_cpu,cpu -device=arm_cpu -mcpu=a64fx -model=a64fx -mtriple=aarch64-unknown-linux-gnu"
target = tvm.target.Target(config_str)
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

#Print Optimized graph as json
print(lib.get_json())

ctx = tvm.cpu()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_runtime.GraphModule(lib["default"](ctx))
# set input and parameters
module.set_input("data", data)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

# Print first 10 elements of output
print(out.flatten()[0:10])

path_lib = "/localscratch/seswar3/arm_mlp_opt.tar"
lib.export_library(path_lib)

loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(data)

module = graph_runtime.GraphModule(loaded_lib["default"](ctx))
module.run(data=input_data)
out_deploy = module.get_output(0).asnumpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])

# check whether the output from deployed module is consistent with original one
tvm.testing.assert_allclose(out_deploy, out, atol=1e-3)