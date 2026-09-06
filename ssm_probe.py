"""Legacy thin entry point (RingKoSpace probe era).

Model code now lives in the installed package `src/ringkospace/`. This file
keeps the historical invocation working on the local workspace:

    python ssm_probe.py --steps 1800 --dim 512 --layers 8 --seq 256 --batch 16

It defaults --data-root to the local skypile jsonl dir when none is given.
Open-source users: `pip install -e .` then `ringkospace-train ...`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # legacy dev convenience

from ringkospace.data import *  # noqa: F401,F403  (encode_fast, decode, load_tokens, ...)
from ringkospace.data import PAD, BOS, EOS, UNK  # noqa: F401
from ringkospace.model import *  # noqa: F401,F403  (RingKoSSM, LiquidSSMBlock, VOCAB)
from ringkospace.model import PAD as _PAD  # noqa

DATA_ROOT = "/storage/Projects/RingKoAI/RingKo/datasets/skypile/data"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    has_src = any(a in ("--bin", "--data-root", "--help", "-h") for a in argv)
    if not has_src:
        argv += ["--data-root", DATA_ROOT]
    from ringkospace.train import main as train_main
    train_main(argv)


if __name__ == "__main__":
    main()
