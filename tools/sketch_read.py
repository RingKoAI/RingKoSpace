"""Differentiable read-enhancement Sketch memory (gated external index).

Scope: faithful-enough torch build of the archived SketchIndex semantics so
gradients reach the stored entries (value projections are kept on the graph by
storing graphed tensors in python lists; write routing stays discrete/pass-
through). Read happens BEFORE the current token is written (past-only), so a
recall must come from an earlier stored fact — no trivial self-copy.

  r_t   = softmax( q_t · stored_k / T ) · stored_v        (per hash table, sum)
  λ_t   = σ(Wg · LN(latent_t))                              (read gate, one/tok)
  add_t = λ_t · Wout(r_t)                                   (added to logits)

Bank: NT tables × NB buckets × SL slots; bucket = sign bits of q·H (fixed H).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

PAD = 0


class SketchRead(nn.Module):
    def __init__(self, dim: int, kd: int = 64, nt: int = 2, nb_bits: int = 6,
                 slots: int = 2, temperature: float = 0.1) -> None:
        super().__init__()
        self.kd, self.nt, self.nb, self.slots, self.temp = kd, nt, 1 << nb_bits, slots, temperature
        self.wq = nn.Linear(dim, kd, bias=False)
        self.wk = nn.Linear(dim, kd, bias=False)
        self.wv = nn.Linear(dim, kd, bias=False)
        self.wout = nn.Linear(kd, 260, bias=False)
        self.gate = nn.Linear(dim, 1)
        h = torch.randn(nt, kd, nb_bits) * 0.4
        self.register_buffer("H", h)
        nn.init.normal_(self.wout.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)

    def _bucket(self, q: Tensor) -> Tensor:
        # q: [B, kd] -> [B, nt]
        bits = (torch.einsum("bd,ndk->bnk", q, self.H) > 0).float()
        w = (2 ** torch.arange(self.H.shape[-1], device=q.device)).view(1, 1, -1)
        return (bits * w).sum(-1).long()

    def forward(self, latent: Tensor) -> Tensor:
        B, T, D = latent.shape
        q = self.wq(latent)          # [B,T,kd]
        k = self.wk(latent)
        v = self.wv(latent)
        # per-batch per-table per-bucket slot store (graphed tensor refs)
        banks = [[[] for _ in range(self.nt * self.nb)] for _ in range(B)]
        adds: list[Tensor] = []
        lam = torch.sigmoid(self.gate(latent))  # [B,T,1]
        for t in range(T):
            qt, kt, vt, lt = q[:, t], k[:, t], v[:, t], lam[:, t]  # [B,kd] / [B,1]
            bc = self._bucket(qt)  # [B,nt]
            r = torch.zeros(B, self.kd, device=latent.device, dtype=latent.dtype)
            for b in range(B):
                for n in range(self.nt):
                    bk = int(bc[b, n])
                    entries = banks[b][n * self.nb + bk]
                    if not entries:
                        continue
                    ks = torch.stack([e[0] for e in entries])   # [L,kd]
                    vs = torch.stack([e[1] for e in entries])   # [L,kd]
                    sim = torch.einsum("d,ld->l", qt[b], ks) / self.temp
                    w = torch.softmax(sim, dim=0)
                    r[b] = r[b] + torch.einsum("l,ld->d", w, vs)
            # write current AFTER read
            for b in range(B):
                for n in range(self.nt):
                    bk = int(bc[b, n])
                    bucket = banks[b][n * self.nb + bk]
                    bucket.append((kt[b], vt[b]))
                    if len(bucket) > self.slots:
                        bucket.pop(0)
            adds.append(r)
        r = torch.stack(adds, dim=1)             # [B,T,kd]
        add = lam * self.wout(r)                 # [B,T,260]
        return add
