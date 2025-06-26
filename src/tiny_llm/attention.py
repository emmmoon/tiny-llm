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
    scale = scale if scale is not None else mx.rsqrt(d_k)
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
        self.scale = mx.rsqrt(self.d_k)
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
    # mask = mx.triu(mx.ones((L, S)), k=(S - L))
    # mask = mx.where(mask, mx.array(-mx.inf), mx.array(0)).astype(dtype)
    # return mask
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask

def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    #query: B, Hq, L, D
    H_q, L, d_k = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0
    expect_shape = query.shape
    n_repeats = H_q // H
    scale = scale if scale is not None else mx.rsqrt(d_k)
    query = query.reshape(-1, H, n_repeats, L, d_k)
    key = key.reshape(-1, H, 1, S, d_k)
    value = value.reshape(-1, H, 1, S, d_k)
    scores = mx.matmul(query, mx.swapaxes(key, -2, -1)) * scale
    if mask is not None:
        if mask == "causal":
            mask =  causal_mask(L, S, query.dtype)
            scores = scores + mask
        else:
            mask = mask.reshape(-1, H, n_repeats, L, S)
            scores = scores + mask
    attn = mx.matmul(mx.softmax(scores, axis=-1), value)
    attn = attn.reshape(expect_shape)
    return attn

def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
