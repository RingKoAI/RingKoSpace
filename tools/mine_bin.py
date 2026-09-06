#!/usr/bin/env python3
"""mine_bin: document-level shuffled uint16 corpus miner.

Stateful byte models want ONE continuous, stable training stream. Random
shuffling every batch destroys cross-doc state meaning; shuffling on the fly
is what the mined stream is for. Two passes:

  pass 1  tokenize every jsonl doc (UTF-8 bytes + BYTE_OFFSET, EOS-terminated),
          append to a seekable raw temp file, record doc (start, length).
  pass 2  visit docs in a fixed-seed shuffled order and stream-append each
          doc's bytes into the final .bin.

Output is consumed sequentially by `ringkospace-train --bin`.

Example:
    python tools/mine_bin.py --input-root ./data/skypile \
        --out corpus.bin --max-docs 500000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

BYTE_OFFSET = 4  # vocab id of byte 0x00
EOS = 2

MAX_DOC_CHARS = 200_000


def encode_fast(text: str) -> np.ndarray:
    """UTF-8 bytes -> int16 vocab ids (byte value + BYTE_OFFSET)."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int16) + BYTE_OFFSET


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="dir of *.jsonl docs")
    ap.add_argument("--out", required=True, help="output .bin path")
    ap.add_argument("--max-docs", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".raw", dir=os.path.dirname(os.path.abspath(args.out)) or ".")
    doc_idx: list[tuple[int, int]] = []  # (offset, length) in uint16 units

    with os.fdopen(tmp_fd, "wb") as raw:
        offset = 0
        n_docs = 0
        for path in sorted(Path(args.input_root).glob("*.jsonl")):
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if n_docs >= args.max_docs:
                        break
                    if not line.strip():
                        continue
                    try:
                        text = (json.loads(line).get("text") or "").strip()
                    except json.JSONDecodeError:
                        continue
                    if not text:
                        continue
                    toks = np.concatenate([encode_fast(text[:MAX_DOC_CHARS]), np.array([EOS], dtype=np.int16)])
                    toks.astype("<u2").tofile(raw)
                    doc_idx.append((offset, int(toks.size)))
                    offset += int(toks.size)
                    n_docs += 1
            if n_docs >= args.max_docs:
                break
    print(f"[pass1] docs={n_docs} tokens={offset}", flush=True)

    order = rng.permutation(len(doc_idx))
    raw = open(tmp_path, "rb")
    try:
        with open(args.out, "wb") as out:
            written = 0
            for pos in order:
                off, ln = doc_idx[pos]
                raw.seek(off * 2)
                chunk = raw.read(ln * 2)
                out.write(chunk)
                written += ln
    finally:
        raw.close()
        os.unlink(tmp_path)
    print(f"[pass2] wrote {args.out} tokens={written}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
