import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
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
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        #q B, L, H_q, D
        projection_q = (
            linear(x, self.wq, self.bq)
            .reshape(B, L, self.num_heads, self.head_dim)
        )
        projection_k = (
            linear(x, self.wk, self.bk)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
        )
        projection_v = (
            linear(x, self.wv, self.bv)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
        )
        projection_q = self.rope(projection_q, slice(offsets, offsets + L))
        projection_k = self.rope(projection_k, slice(offsets, offsets + L))
        projection_q = projection_q.transpose(0, 2, 1, 3)  # B, H_q, L, D
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        projection_k, projection_v, _, mask = cache.update_and_fetch(
            projection_k, projection_v, mask_length=L, mask=mask)
        attn_output = scaled_dot_product_attention_grouped(
            projection_q,
            projection_k,
            projection_v,
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
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
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
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq,
            wk,
            wv,
            wo,
            bq,
            bk,
            bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.mlp = Qwen2MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down,
        )
        self.input_layernorm = RMSNorm(
            hidden_size,
            w_input_layernorm,
            eps=rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            w_post_attention_layernorm,
            eps=rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.intermediate_size = mlx_model.args.intermediate_size
        self.num_attention_heads = mlx_model.args.num_attention_heads
        self.num_kv_heads = mlx_model.args.num_key_value_heads
        self.rms_norm_eps = mlx_model.args.rms_norm_eps
        self.rope_theta = mlx_model.args.rope_theta
        self.tie_word_embeddings = mlx_model.args.tie_word_embeddings
        self.vocab_size = mlx_model.args.vocab_size
        self.precision = mx.float16
        self.embeding = Embedding(
            mlx_model.args.vocab_size,
            self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(self.precision),
        )
        self.inner_layers = []
        for i in range(mlx_model.args.num_hidden_layers):
            layer = Qwen2TransformerBlock(
                num_attention_heads=self.num_attention_heads,
                num_kv_heads=self.num_kv_heads,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                rms_norm_eps=self.rms_norm_eps,
                wq=dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj).astype(self.precision),
                wk=dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj).astype(self.precision),
                wv=dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj).astype(self.precision),
                wo=dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj).astype(self.precision),
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(self.precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(self.precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(self.precision),
                w_gate=dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj).astype(self.precision),
                w_up=dequantize_linear(mlx_model.model.layers[i].mlp.up_proj).astype(self.precision),
                w_down=dequantize_linear(mlx_model.model.layers[i].mlp.down_proj).astype(self.precision),
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight.astype(self.precision),
                w_post_attention_layernorm=mlx_model.model.layers[i].post_attention_layernorm.weight.astype(self.precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=self.rope_theta,
            )
            self.inner_layers.append(layer)
        
        self.norm = RMSNorm(
            self.hidden_size,
            mlx_model.model.norm.weight.astype(self.precision),
            eps=self.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        h = self.embeding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.inner_layers[layer](
                h, offset ,cache[layer], mask="causal" if h.shape[1] > 1 else None)
        h = self.norm(h)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        else:
            return self.embeding.as_linear(h)
