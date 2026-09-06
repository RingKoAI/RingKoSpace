"""Data helpers: byte encoding/decoding and streaming sources.

Token scheme: 260 vocab = 4 specials (PAD/BOS/EOS/UNK) + 256 UTF-8 bytes
(BYTE_OFFSET=4).  UTF-8 bytes are length-coded by 0xFF -> continuation bytes
are NOT distinguished from lead bytes; the model must learn byte structure.

Two sources:
  jsonl    developer-readable text docs, encoded on the fly (load_tokens)
  bin      offline-mined uint16 stream (see tools/mine_bin or DATA.md), read
           with mmap in train order, doc-shuffled, EOS-separated.

A mined corpus for an epoch of *N* tokens should be produced once and then
streamed sequentially (stateful models want one continuous, stable order;
document-level shuffle is done once at mining time, not at train time).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

PAD, BOS, EOS, UNK = 0, 1, 2, 3
BYTE_OFFSET = 4


def encode_fast(text: str) -> np.ndarray:
    """Encode UTF-8 text to a np array of vocab ids (int16)."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int16) + BYTE_OFFSET


def decode(ids) -> str:
    """Decode vocab ids back to a UTF-8 string (byte ids only)."""
    bs = bytes(max(0, int(t) - BYTE_OFFSET) for t in ids if BYTE_OFFSET <= int(t) < 260)
    return bs.decode("utf-8", errors="replace")


def decode_char_boundary(bs: bytes) -> bytes:
    """Trim raw utf-8 bytes to the last full character boundary (for prompts)."""
    return bs.decode("utf-8", "ignore").encode("utf-8")


def load_tokens(root: Path, max_tokens: int) -> list[int]:
    """Stream-encode jsonl text docs up to max_tokens (EOS-separated).

    Each doc is capped at 2000 chars so a single giant doc cannot dominate a
    window budget.  Malformed lines are skipped (external data is untrusted).
    """
    tokens: list[int] = []
    for path in sorted(Path(root).glob("*.jsonl")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text")
                if isinstance(text, str) and text.strip():
                    tokens.extend(encode_fast(text[:2000]).tolist())
                    tokens.append(EOS)
                if len(tokens) >= max_tokens:
                    return tokens
    return tokens


def load_bin(bin_path: str | Path, max_tokens: int, eval_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Memmap a mined uint16 stream; return (train, eval) 1-D tensors."""
    mm = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
    n = int(min(len(mm), max_tokens + eval_tokens))
    arr = np.asarray(mm[:n]).astype(np.int64)
    data = torch.from_numpy(arr)
    n_train = max(n - eval_tokens, 0)
    return data[:n_train], data[n_train:]
