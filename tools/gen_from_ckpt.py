"""Load a RingKoSpace checkpoint and generate (windowed autoregressive).

Window mode re-runs the causal window from zero each token (no carried state):
still reflects learned language quality; cross-window memory is intentionally
absent (see docs/SUMMARY-20260906.md, exposure-bias section).

Run (after `pip install -e .` or with PYTHONPATH=src):
    python tools/gen_from_ckpt.py --ckpt ckpt/step-15000.pt --data-root ./data/skypile
"""
from __future__ import annotations
from pathlib import Path as _P
import sys as _s
_s.path.insert(0, str(_P(__file__).resolve().parent.parent / "src"))



import argparse

import torch

from ringkospace.data import decode
from ringkospace.generate import first_prompt, generate
from ringkospace.model import RingKoSSM

DEV = "cuda"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True, help="skypile-style jsonl dir for the prompt")
    ap.add_argument("--tokens", type=int, default=240)
    ap.add_argument("--maxlen", type=int, default=700)
    ap.add_argument("--mode", choices=["stream", "window"], default="stream")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    print(f"[load] step={ck['step']} ema={ck.get('ema_loss')} cfg dim={cfg['dim']} L={cfg['layers']} evolve={cfg.get('evolve', 0)}")
    model = RingKoSSM(cfg["dim"], cfg["layers"], evolve=bool(cfg.get("evolve", 0)), conv_k=cfg.get("conv_k", 4), scan_floor=cfg.get("scan_floor", 1e-2)).to(DEV).eval()
    model.load_state_dict(ck["state_dict"])

    pr = first_prompt(args.data_root)
    print(">>> " + decode(pr[:200]))
    for temp, topk, rp in ((0.8, 40, 1.3), (0.01, 1, 1.0)):
        _, text = generate(model, pr, args.tokens, temp, topk, args.seed,
                           args.maxlen, rep_pen=rp, mode=args.mode, device=DEV)
        print(f"\n--- {args.mode} temp={temp} topk={topk} ---")
        print(text)


if __name__ == "__main__":
    main()
