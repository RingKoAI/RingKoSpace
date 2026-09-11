"""Division-free affine prefix scan; exact zero gates are supported."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

SCAN_CHUNK = 32


def validate_scan(keep: Tensor, inc: Tensor, state: Tensor | None) -> None:
    if not isinstance(keep, Tensor) or not isinstance(inc, Tensor) or keep.ndim != 3 or keep.shape != inc.shape or any(n == 0 for n in keep.shape):
        raise ValueError("keep and inc must have identical nonempty [B,T,D] shapes")
    if keep.device != inc.device or not keep.is_floating_point() or not inc.is_floating_point():
        raise ValueError("scan inputs must be floating tensors on the same device")
    if state is not None and (not isinstance(state, Tensor) or state.shape != (keep.shape[0], keep.shape[2])
                              or state.device != keep.device or not state.is_floating_point()):
        raise ValueError("state must be floating [B,D] on the input device")


def scan_first_order(keep: Tensor, inc: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Compute h[t]=keep[t]*h[t-1]+inc[t], without a hidden gate floor.

    Parallel prefix within 32-token chunks, sequential carries across chunks.
    Float64 intermediates limit roundoff; public outputs retain historical fp32.
    History is differentiable, including the returned final carry.
    """
    validate_scan(keep, inc, state)
    b, t, d = keep.shape
    if t == 1:
        initial = state.double() if state is not None else torch.zeros_like(keep[:, 0], dtype=torch.float64)
        value = inc[:, 0].double() + keep[:, 0].double() * initial
        value = value.float()
        return value.unsqueeze(1), value.contiguous()
    padding = (-t) % SCAN_CHUNK
    a = F.pad(keep.double(), (0, 0, 0, padding), value=1).reshape(b, -1, SCAN_CHUNK, d)
    v = F.pad(inc.double(), (0, 0, 0, padding)).reshape(b, -1, SCAN_CHUNK, d)
    offset = 1
    while offset < SCAN_CHUNK:
        v = torch.cat((v[:, :, :offset], v[:, :, offset:] + a[:, :, offset:] * v[:, :, :-offset]), dim=2)
        a = torch.cat((a[:, :, :offset], a[:, :, offset:] * a[:, :, :-offset]), dim=2)
        offset *= 2
    carry = state.double() if state is not None else torch.zeros(b, d, device=keep.device, dtype=torch.float64)
    chunks = []
    for i in range(a.shape[1]):
        values = v[:, i] + a[:, i] * carry.unsqueeze(1)
        chunks.append(values)
        carry = values[:, -1]
    return torch.cat(chunks, dim=1)[:, :t].float(), carry.float().contiguous()
