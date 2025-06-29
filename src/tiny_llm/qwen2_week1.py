import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        #q B, L, H_q, D
        q = (
            linear(x, self.wq, self.bq)
            .reshape(B, L, self.num_heads, self.head_dim)
        )
        k = (
            linear(x, self.wk, self.bk)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
        )
        v = (
            linear(x, self.wv, self.bv)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
        )
        q = self.rope(q, slice(offset, offset + L))
        k = self.rope(k, slice(offset, offset + L))
        q = q.transpose(0, 2, 1, 3)  # B, H_q, L, D
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        attn_output = scaled_dot_product_attention_grouped(
            q,
            k,
            v,
            self.scale,
            mask,
        ).astype(x.dtype)
        attn_output = attn_output.transpose(0, 2, 1, 3)# B, L, H_q, D
        attn_output = (
            attn_output.reshape(B, L, self.num_heads * self.head_dim)
        )
        return linear(attn_output, self.wo)
    
    
class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate_proj = linear(x, self.w_gate)
        up_proj = linear(x, self.w_up)
        return linear(silu(gate_proj) * up_proj, self.w_down)

class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
