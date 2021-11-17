
import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
from tvm import topi

from opt_dense import opt_dense,opt_dense_schedule

# data_shape = (batch_size,) + image_shape
#     data = relay.var("data", shape=data_shape, dtype=dtype)
#     data = relay.nn.batch_flatten(data)
#     fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=128)
#     fc1 = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=-1)
#     act1 = relay.nn.relu(fc1)
#     fc2 = relay.nn.dense(act1, relay.var("fc2_weight"), units=64)
#     fc2 = relay.nn.bias_add(fc2, relay.var("fc2_bias"), axis=-1)
#     act2 = relay.nn.relu(fc2)
#     fc3 = relay.nn.dense(act2, relay.var("fc3_weight"), units=num_classes)
#     fc3 = relay.nn.bias_add(fc3, relay.var("fc3_bias"), axis=-1)
#     mlp = relay.nn.softmax(data=fc3)
#     args = relay.analysis.free_vars(mlp)

A = te.placeholder((128, 1, 28, 28))
W1= te.placeholder((128,28*28))
b1= te.placeholder((128,))

W2= te.placeholder((64,128))
b2= te.placeholder((64,))

W3= te.placeholder((10,64))
b3= te.placeholder((10,))

A1 = topi.nn.flatten(A)
A2 , B2 = opt_dense(A1,W1,b1)
A2r = topi.nn.relu(A2)
A3 , B3 =  opt_dense(A2r,W2,b2)
A3r = topi.nn.relu(A3)
A4 , B4 = opt_dense(A3r,W3,b3)
A5 = topi.nn.softmax(A4)

s = te.create_schedule(A5.op)
print(tvm.lower(s,[A,W1,b1,W2,b2,W3,b3,A5], simple_mode=True))

target = "llvm"
ctx = tvm.context(target, 0)
func = tvm.build(s, [A,W1,b1,W2,b2,W3,b3,A5], target=target)

dtype = "float32"
a = tvm.nd.array(numpy.random.rand(128, 1, 28, 28).astype(dtype), ctx)
w = tvm.nd.array(numpy.random.rand(128,28*28).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(128,).astype(dtype), ctx)
w2 = tvm.nd.array(numpy.random.rand(64,128).astype(dtype), ctx)
b2 = tvm.nd.array(numpy.random.rand(64,).astype(dtype), ctx)
w3 = tvm.nd.array(numpy.random.rand(10,64).astype(dtype), ctx)
b3 = tvm.nd.array(numpy.random.rand(10,).astype(dtype), ctx)
c = tvm.nd.array(numpy.zeros((128,10), dtype=dtype), ctx)
func(a,w,b,w2,b2,w3,b3,c)

#Numpy

import numpy as np
a11 = a.asnumpy().reshape(a.asnumpy().shape[0],-1)
a21 = np.matmul(a11,w.asnumpy().T) + b.asnumpy()
a21 = np.maximum(0, a21)
a31 = np.matmul(a21,w2.asnumpy().T) + b2.asnumpy()
a31 = np.maximum(0, a31)
a41 = np.matmul(a31,w3.asnumpy().T) + b3.asnumpy()
s2 = np.max(a41, axis=1)
s2 = s2[:, np.newaxis] # (z.shape[0], 1)
e_x = np.exp(a41 - s2) #  improve the numerical stability
# see https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
div = np.sum(e_x, axis=1)
div = div[:, np.newaxis] 
a51 = e_x / div
 
tvm.testing.assert_allclose(c.asnumpy(), a51, rtol=1e-5)