import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = query.shape[-1]
    scale = scale if scale is not None else 1.0 / (d_k ** 0.5)
    scores = mx.matmul(query, mx.swapaxes(key, -2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    attn = mx.matmul(mx.softmax(scores, axis=-1), value)
    return attn


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.scale = 1.0 / (self.d_k ** 0.5)
        assert wq.shape == (hidden_size, self.d_k * self.num_heads)
        assert wk.shape == (hidden_size, self.d_k * self.num_heads)
        assert wv.shape == (hidden_size, self.d_k * self.num_heads)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo


    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        project_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.d_k)
            .transpose(0, 2, 1, 3)
        )
        project_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.d_k)
            .transpose(0, 2, 1, 3)
        )
        project_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.d_k)
            .transpose(0, 2, 1, 3)
        )
        attn = scaled_dot_product_attention_simple(
            project_q,
            project_k, 
            project_v, 
            scale=self.scale, 
            mask=mask
        )
        attn = (
            attn.transpose(0, 2, 1, 3)
            .reshape(N, L, self.num_heads * self.d_k)
        )
        return linear(attn, self.wo)

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
