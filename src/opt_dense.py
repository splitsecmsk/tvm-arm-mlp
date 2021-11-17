from tvm import te
from tvm.topi.nn import tag
import tvm

def opt_dense(data, weight, bias=None, out_dtype=None,bnx=32,bny=32):
    """The default implementation of dense in topi.
    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]
    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]
    bias : tvm.te.Tensor, optional
        1-D with shape [out_dim]
    out_dtype : str
        The output type. This is used for mixed precision.
    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1

    if out_dtype is None:
        out_dtype = data.dtype

    batch, in_dim = data.shape
    out_dim, _ = weight.shape

    (M,N) = (batch,out_dim)
    K = in_dim
    B = weight
    A = data

    k = te.reduce_axis((0, in_dim), name="k")
    # matmul = te.compute(
    #     (batch, out_dim),
    #     lambda i, j: te.sum(data[i, k].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=k),
    #     name="T_dense",
    #     tag="dense",
    # )

    packedB = te.compute(( tvm.tir.indexdiv(N,bny), K, bny), lambda x, y, z: B[y, x * bny + z], name="packedB")
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(A[x, k] * packedB[ tvm.tir.indexdiv( y , bny), k, tvm.tir.indexmod(y, bny)], axis=k),
        name="C",
        tag="dense",
    )

    # func = tvm.build(s, [A, B, C], target=target, name="mmult")

    if bias is not None:
        C = te.compute(
            (batch, out_dim),
            lambda i, j: C[i, j] + bias[j].astype(out_dtype),
            tag=tag.BROADCAST,
        )
    return (C,packedB)

def opt_dense_schedule(s,C,packedB,bnx=32,bny=32):

    CC = s.cache_write(C, "global")

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bnx, bny)

    s[CC].compute_at(s[C], yo)

    xc, yc = s[CC].op.axis

    (k,) = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)

    # parallel
    s[C].parallel(xo)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)

    return s