"""Numerical equivalence: chunked log-space scan == sequential recurrence.

The scan is the trust core of the model (parallel training with window-level
gradients). If it drifts from the sequential definition the whole "state"
story is void. We compare the READ output (which is what the loss sees) of the
vectorized forward against a plain per-token loop, for both evolve modes and
for sequence lengths that straddle the internal chunk boundary (chunk=32).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ringkospace.model import LiquidSSMBlock

D = 48


def _reference(blk: LiquidSSMBlock, x: torch.Tensor) -> torch.Tensor:
    b, t, d = x.shape
    hn = blk.norm(x)
    hc = F.silu(blk.conv(hn.transpose(1, 2)))[..., :t].transpose(1, 2)
    z = F.silu(blk.wz(hc))
    delta = F.softplus(blk.wd(hc) + blk.d_bias.view(1, 1, d))
    keep = torch.exp(-delta * torch.exp(blk.a_log).view(1, 1, d))
    if blk.evolve:
        a_evo = torch.sigmoid(blk.wa(hc))
        ke = keep + (1.0 - keep) * a_evo
        ie = (1.0 - keep) * (1.0 - a_evo) * z
    else:
        ke, ie = keep, (1.0 - keep) * z
    h = torch.zeros(b, d, dtype=torch.float64)
    outs = []
    for s in range(t):
        h = ke[:, s].double() * h + ie[:, s].double()
        outs.append(h.float())
    hs = torch.stack(outs, dim=1)
    return blk.wr(blk.norm(hs)) * blk.scale.view(1, 1, d) + blk.dv.view(1, 1, d) * x


def _make(evolve: bool) -> LiquidSSMBlock:
    torch.manual_seed(0)
    blk = LiquidSSMBlock(D, conv_k=4, evolve=evolve).eval()
    with torch.no_grad():
        for p in blk.parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.3)
    return blk


def _assert_eq(evolve: bool, t: int) -> None:
    blk = _make(evolve)
    torch.manual_seed(1)
    x = torch.randn(2, t, D)
    with torch.no_grad():
        y_fwd, _ = blk(x, None)
        y_ref = _reference(blk, x)
    err = (y_fwd - y_ref).abs().max().item()
    assert err < 2e-3, f"evolve={evolve} t={t} max|err|={err:.2e}"


def test_scan_matches_sequential_short() -> None:
    _assert_eq(False, 16)
    _assert_eq(True, 16)


def test_scan_matches_sequential_across_chunk() -> None:
    _assert_eq(False, 100)  # spans 4 chunks incl. padding tail
    _assert_eq(True, 100)


def test_scan_matches_sequential_exact_chunk() -> None:
    _assert_eq(False, 64)
    _assert_eq(True, 64)


def test_evolve_keep_eff_bounded() -> None:
    blk = _make(True)
    torch.manual_seed(2)
    x = torch.randn(1, 8, D)
    hn = blk.norm(x)
    hc = F.silu(blk.conv(hn.transpose(1, 2)))[..., :8].transpose(1, 2)
    keep = torch.exp(
        -F.softplus(blk.wd(hc) + blk.d_bias.view(1, 1, D)) * torch.exp(blk.a_log).view(1, 1, D)
    )
    a_evo = torch.sigmoid(blk.wa(hc))
    ke = keep + (1.0 - keep) * a_evo
    assert bool((ke > 0).all()) and bool((ke < 1).all())
