import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.weight = weight.astype(mx.float32)
        self.eps = eps
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        self.odtype = x.dtype
        x = x.astype(mx.float32)
        return (x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps) * self.weight).astype(self.odtype)