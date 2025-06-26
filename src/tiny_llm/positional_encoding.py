import math
import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        half_dims = dims // 2
        position = mx.arange(0, seq_len)
        freqs = mx.exp(
            -mx.arange(0.0, half_dims) * (math.log(base) / half_dims)
        )
        freqs = mx.outer(position, freqs)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)
        self.base = base
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # doesn't change shape
        N, L, H, D = x.shape
        if offset is not None:
            if isinstance(offset, slice):
                assert offset.stop - offset.start == L, f"offset must be of length {L}"
        
        cos_basis = (
            self.cos_freqs[:L, :] if offset is None else self.cos_freqs[offset, :]
        )
        sin_basis = (
            self.sin_freqs[:L, :] if offset is None else self.sin_freqs[offset, :]
        )
        if self.traditional:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        else:
            x1 = x[..., : self.dims // 2]
            x2 = x[..., self.dims // 2 :]
        
        cos_basis = cos_basis.reshape(1, L, 1, self.dims // 2)
        sin_basis = sin_basis.reshape(1, L, 1, self.dims // 2)
        real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)
        if self.traditional:
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, L, H, D)
        else:
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(N, L, H, D)
        
        return y.astype(x.dtype)



        