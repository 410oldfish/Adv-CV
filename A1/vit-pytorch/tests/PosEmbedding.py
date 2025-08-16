import math
import torch
from torch import nn
from typing import Optional
from rotary_embedding_torch import RotaryEmbedding

class BasePositionalEmbedding(nn.Module):
    """Base class that returns x + position[:, :seq_len, :]."""
    def __init__(self, dim: int, num_tokens: int):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.is_rotary = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos[:, : x.size(1), :]
        return x + pos


class FixedSinusoidalPositionalEmbedding(BasePositionalEmbedding):
    """
    Classic Transformer sinusoidal embeddings, precomputed and stored as a buffer.
    """
    def __init__(self, dim: int, num_tokens: int, base: float = 10000.0):
        super().__init__(dim, num_tokens)

        pe = torch.zeros(num_tokens, dim, dtype=torch.float32)  # [L, D]
        position = torch.arange(0, num_tokens, dtype=torch.float32).unsqueeze(1)  # [L, 1]

        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(base) / dim)
        )  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # If dim is odd, the last column (cos branch) remains zeros (fine for most uses)
        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer("pos", pe, persistent=False)


class LearnablePositionalEmbedding(BasePositionalEmbedding):
    """
    Learnable positional embeddings (default in many ViT implementations).
    """
    def __init__(self, dim: int, num_tokens: int, init_std: float = 0.02):
        super().__init__(dim, num_tokens)
        pos = torch.zeros(1, num_tokens, dim)  # [1, L, D]
        nn.init.trunc_normal_(pos, std=init_std)
        self.pos = nn.Parameter(pos)  # learnable

class RotaryTorchPositional(nn.Module):
    """
    Wrapper around lucidrains/rotary-embedding-torch for clean integration.
    - forward(x) is a no-op (RoPE is multiplicative, applied to q/k).
    - call apply_rotary(q, k) inside attention after head-splitting.
    """
    def __init__(
        self,
        *,
        dim_head: int,
        rope_dim: Optional[int] = None,      # defaults to dim_head // 2
        rope_base: float = 10000.0,
        use_xpos: bool = False,              # XPos is for AR decoders; keep False for ViT
        interpolate_factor: float = 1.0
    ):
        super().__init__()

        self.is_rotary = True
        self._use_xpos = bool(use_xpos)

        if rope_dim is None:
            rope_dim = (dim_head // 2)
        assert rope_dim % 2 == 0, "rope_dim must be even"

        self._rotary = RotaryEmbedding(
            dim=rope_dim,
            base=rope_base,
            use_xpos=self._use_xpos,
            interpolate_factor=interpolate_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # additive no-op

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: [B, H, L, Dh]
        Uses rotate_queries_or_keys() or rotate_queries_and_keys() per README.
        """
        if self._use_xpos:
            # applies both queries and keys with proper scaling for XPos
            q, k = self._rotary.rotate_queries_and_keys(q, k)
            return q, k
        # vanilla RoPE: rotate q and k separately
        q = self._rotary.rotate_queries_or_keys(q)
        k = self._rotary.rotate_queries_or_keys(k)
        return q, k

def build_positional_embedding(
    kind: str,
    dim: int,
    num_tokens: int,
    *,
    # rope-specific args
    dim_head: Optional[int] = None,
    rope_dim: Optional[int] = None,
    rope_base: float = 10000.0,
    interpolate_factor: float = 1.0,
) -> nn.Module:
    """
    kind: "sinusoidal" | "learnable" | "rope"
    dim: token dim (additive only)
    num_tokens: sequence length incl. CLS for additive
    For rope: pass dim_head (head dimension) and (optionally) rope_dim.
    """
    kind = kind.lower()
    if kind in ("sine"):
        return FixedSinusoidalPositionalEmbedding(dim=dim, num_tokens=num_tokens)
    if kind in ("learnable"):
        return LearnablePositionalEmbedding(dim=dim, num_tokens=num_tokens)
    if kind in ("rotary"):
        if dim_head is None:
            # fallback to half of token dim if head dim not given
            dim_head = dim
        return RotaryTorchPositional(
            dim_head=dim_head,
            rope_dim=rope_dim,
            rope_base=rope_base,
            interpolate_factor=interpolate_factor,
        )
    raise ValueError(f"Unknown positional embedding kind: {kind}")