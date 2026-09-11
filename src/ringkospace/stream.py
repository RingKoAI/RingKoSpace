"""Explicit per-request streaming state; callers must treat state tensors as read-only."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LayerState:
    recurrent: Tensor
    convolution: Tensor

    def detach(self) -> LayerState:
        """Cut history only when the caller explicitly chooses truncated BPTT."""
        return LayerState(self.recurrent.detach(), self.convolution.detach())


@dataclass(frozen=True)
class StreamState:
    layers: tuple[LayerState, ...]
    owner: object

    def detach(self) -> StreamState:
        return StreamState(tuple(layer.detach() for layer in self.layers), self.owner)


def causal_convolution(conv: nn.Conv1d, normalized: Tensor, history: Tensor | None,
                       streaming: bool) -> tuple[Tensor, Tensor | None]:
    x = normalized.transpose(1, 2)
    if not streaming:
        return F.silu(conv(x)[..., :x.shape[-1]]).transpose(1, 2), None
    width = conv.kernel_size[0] - 1
    if history is None:
        history = x.new_zeros(x.shape[0], x.shape[1], width)
    if not isinstance(history, Tensor) or history.shape != (x.shape[0], x.shape[1], width) or history.device != x.device or history.dtype != x.dtype:
        raise ValueError("convolution history shape/device/dtype does not match this stream")
    joined = torch.cat((history, x), dim=-1)
    out = F.conv1d(joined, conv.weight, conv.bias, groups=conv.groups)
    # Clone the bounded history so it does not retain the whole prompt storage.
    tail = joined[..., -width:].clone() if width else joined[..., :0].clone()
    return F.silu(out).transpose(1, 2), tail


def forward_stream(model: nn.Module, tokens: Tensor, state: StreamState | None = None):
    """Request-local state ownership; model diagnostics remain last-call snapshots.

    Pass None to reset. Do not reuse a state after changing model weights, dtype
    or device. Concurrent training/mutation of a model is not supported.
    """
    if not isinstance(tokens, Tensor) or tokens.ndim != 2 or any(n == 0 for n in tokens.shape) or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("tokens must be nonempty integer [B,T]")
    if tokens.device != model.embedding.weight.device:
        raise ValueError("tokens and model must be on the same device")
    if state is not None:
        if not isinstance(state, StreamState) or state.owner is not model._stream_owner:
            raise ValueError("stream state belongs to another model")
        if len(state.layers) != len(model.blocks):
            raise ValueError("stream state layer count mismatch")
    h = model.embedding(tokens)
    result = []
    for i, block in enumerate(model.blocks):
        previous = None if state is None else state.layers[i]
        if previous is not None:
            width = model.dim * (2 if getattr(block, "mass", False) else 1)
            if not isinstance(previous, LayerState) or not isinstance(previous.recurrent, Tensor) or previous.recurrent.shape != (tokens.shape[0], width):
                raise ValueError("recurrent state shape mismatch")
        h, carry, history = block._forward(h, None if previous is None else previous.recurrent,
                                           None if previous is None else previous.convolution, True)
        result.append(LayerState(carry, history))
    model._latent = h
    return model.head(model.final_norm(h)), StreamState(tuple(result), model._stream_owner)
