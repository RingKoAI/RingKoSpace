"""Model definitions: single-line liquid-gated linear recurrence.

Each block owns ONE recurrent state h in [B, D]. No FFN or explicit position encoding;
a learned dv*x input bypass remains.
Byte LM over a 260 vocab (4 specials + 256 UTF-8 bytes), tied head.

Two modes:
  evolve=False  z is a pure content candidate        -> h = a*h + (1-a)*z
  evolve=True   the candidate linearly evolves the old state too:
                  h_t = [keep + (1-keep)*a_evo] * h_{t-1}
                        + (1-keep)*(1-a_evo) * z_t
                effective keep stays in (0,1), so the affine prefix scan remains
                closed-form. This is the "CfC as step-length controller over a
                single SSM line" variant (see docs/DESIGN-evolve.md).
"""

from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._scan import scan_first_order
from .stream import StreamState, causal_convolution, forward_stream

PAD, BOS, EOS, UNK = 0, 1, 2, 3
VOCAB = 260  # 4 special + 256 bytes


class LiquidSSMBlock(nn.Module):
    """One block: single recurrent state h in [B,D], liquid-gated update.

    Update (per-channel first-order recurrence, fully differentiable):
        z_t     = SiLU( conv( Wz x ) )                 content candidate
        Delta   = softplus( Wd x + bd )                input step size
        a_t     = exp( -Delta * exp(A_log) )           per-channel keep gate
        h_t     = a_t * h_{t-1} + (1 - a_t) * z_t      single-line update

    Parallel: affine prefix within chunks along the sequence axis (history NOT
    detached, window-level gradients).  Read: y = Wr * LN(h) * scale + dv*x.
    """

    def __init__(self, dim: int, conv_k: int = 4, evolve: bool = False, scan_floor: float = 1e-2) -> None:
        super().__init__()
        if dim <= 0 or conv_k <= 0:
            raise ValueError("dim and conv_k must be positive")
        if not 0 <= scan_floor <= 1:
            raise ValueError("scan_floor must be in [0,1]")
        self.scan_floor = scan_floor
        self.dim = dim
        self.evolve = evolve
        self.norm = nn.LayerNorm(dim)
        self.wz = nn.Linear(dim, dim, bias=False)
        self.wd = nn.Linear(dim, dim, bias=False)
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=conv_k, padding=conv_k - 1, groups=dim, bias=True
        )
        self.a_log = nn.Parameter(torch.zeros(dim))
        self.d_bias = nn.Parameter(torch.full((dim,), -3.5))
        if evolve:
            # evolution diagonal a_evo in (0,1): how much of the OLD state is
            # linearly carried into the content candidate.
            self.wa = nn.Linear(dim, dim, bias=False)
            nn.init.normal_(self.wa.weight, std=0.02)
        self.wr = nn.Linear(dim, dim, bias=False)
        self.scale = nn.Parameter(torch.full((dim,), 0.1))
        self.dv = nn.Parameter(torch.full((dim,), 0.5))  # Mamba D·x current-token skip
        nn.init.normal_(self.wr.weight, std=0.02)
        nn.init.normal_(self.wz.weight, std=0.02)

    def forward(self, x: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        y, carry, _ = self._forward(x, state)
        return y, carry

    def _forward(self, x: Tensor, state: Tensor | None, history: Tensor | None = None,
                 streaming: bool = False) -> tuple[Tensor, Tensor, Tensor | None]:
        # x: [B,T,D]; state: [B,D] or None
        b, t, d = x.shape
        hn = self.norm(x)
        hc, next_history = causal_convolution(self.conv, hn, history, streaming)

        z = F.silu(self.wz(hc))  # [B,T,D] content
        delta = F.softplus(self.wd(hc) + self.d_bias.view(1, 1, d))  # >0
        keep = torch.exp(-delta * torch.exp(self.a_log).view(1, 1, d))  # (0,1]
        if self.evolve:
            a_evo = torch.sigmoid(self.wa(hc))
            keep_eff = keep + (1.0 - keep) * a_evo
            inc_eff = (1.0 - keep) * (1.0 - a_evo) * z
        else:
            keep_eff = keep
            inc_eff = (1.0 - keep) * z

        h_seq, carry = scan_first_order(keep_eff.clamp_min(self.scan_floor), inc_eff, state)

        # read
        y = self.wr(self.norm(h_seq)) * self.scale.view(1, 1, d) + self.dv.view(
            1, 1, d
        ) * x
        return y, carry, next_history


class RingKoSSM(nn.Module):
    def __init__(self, dim: int, layers: int, conv_k: int = 4, evolve: bool = False, scan_floor: float = 1e-2) -> None:
        super().__init__()
        if dim <= 0 or conv_k <= 0:
            raise ValueError("dim and conv_k must be positive")
        if not 0 <= scan_floor <= 1:
            raise ValueError("scan_floor must be in [0,1]")
        self.scan_floor = scan_floor
        self.dim = dim
        self.evolve = evolve
        if layers <= 0:
            raise ValueError("layers must be positive")
        self._stream_owner = object()
        self.embedding = nn.Embedding(VOCAB, dim, padding_idx=PAD)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.blocks = nn.ModuleList([LiquidSSMBlock(dim, conv_k, evolve, scan_floor) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB, bias=False)
        self.head.weight = self.embedding.weight
        self.layers = layers

    def forward_stream(self, x: Tensor, state: StreamState | None = None) -> tuple[Tensor, StreamState]:
        """Continue with complete recurrent + convolution history, or reset with None."""
        return forward_stream(self, x, state)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor, states: list[Tensor] | None = None) -> tuple[Tensor, list[Tensor]]:
        """Window forward; states seed only the recurrence, not convolution.

        For contiguous chunks use forward_stream and its complete StreamState.
        This legacy API is retained for checkpoint-era callers.
        """
        if not isinstance(x, Tensor) or x.ndim != 2 or any(n == 0 for n in x.shape):
            raise ValueError("tokens must be nonempty [B,T]")
        if states is not None and (not isinstance(states, (list, tuple)) or len(states) != self.layers):
            raise ValueError("states must contain one recurrent tensor per layer; use forward_stream for streaming")
        b, t = x.shape
        h = self.embedding(x)
        next_states: list[Tensor] = []
        for i, blk in enumerate(self.blocks):
            st = states[i] if states is not None else None
            y, st_new = blk(h, st)
            next_states.append(st_new)
            h = y
        self._latent = h  # final block output (pre head), for read-side add-ons
        logits = self.head(self.final_norm(h))
        return logits, next_states
