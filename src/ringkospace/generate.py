"""Autoregressive generation from a RingKoSpace checkpoint.

Two modes (see docs/SUMMARY-20260906.md, exposure-bias section):
  window  re-run the causal window (prompt + generated so far, capped at
          --maxlen) from zero at each token. Stable continuation, but it does
          NOT exercise a carried state (v0 was trained window-wise).
  stream  prime recurrent states AND causal convolution histories, then carry
          both forward one byte at a time. This matches a growing full prefix
          up to numerical roundoff. Training still uses independent windows;
          longer-than-training-context quality is not guaranteed.

UTF-8 note: never truncate a prompt inside a multi-byte char; trim to the last
clean char boundary first (decode_char_boundary).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import model as M
from .data import BYTE_OFFSET, decode, decode_char_boundary, encode_fast


def first_prompt(root: str, maxb: int = 260) -> list[int]:
    """First non-trivial jsonl doc encoded to byte ids (char-boundary trimmed)."""
    for path in sorted(Path(root).glob("*.jsonl")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    txt = (json.loads(line).get("text") or "").strip()
                except json.JSONDecodeError:
                    continue
                if len(txt) > 40:
                    raw = txt.encode("utf-8")[:maxb]
                    clean = decode_char_boundary(raw)
                    return encode_fast(clean.decode("utf-8")).tolist()
    raise SystemExit("no prompt found in --data-root")


@torch.inference_mode()
def generate(model: M.RingKoSSM, prompt: list[int], n: int, temperature: float,
             top_k: int, seed: int, maxlen: int, rep_pen: float = 1.0,
             mode: str = "stream", device: str = "cuda", precision: str = "bf16") -> tuple[list[int], str]:
    if not prompt or maxlen < 2 or n < 0 or temperature <= 0 or rep_pen <= 0 or top_k < 0 or mode not in ("stream", "window") or precision not in ("fp32", "bf16"):
        raise ValueError("invalid generation parameters or empty prompt")
    g = torch.Generator(device=device).manual_seed(seed)
    seq = prompt[: maxlen - 1]
    done: list[int] = []
    enc = lambda toks: torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16"):
        lg, states = model.forward_stream(enc(seq)) if mode == "stream" else model(enc(seq))
        logits = lg[0, -1].float()
        for _ in range(n):
            if mode == "window":
                lg, _ = model(enc(seq[:maxlen]))
                logits = lg[0, -1].float()
            if rep_pen > 1.0:
                for t in set(done):
                    if logits[t] > 0:
                        logits[t] = logits[t] / rep_pen
                    else:
                        logits[t] = logits[t] * rep_pen
            logits = logits / temperature
            if top_k and top_k < logits.shape[-1]:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            p = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(p, 1, generator=g).item())
            done.append(nxt)
            if mode == "stream":
                lg, states = model.forward_stream(enc([nxt]), states)
                logits = lg[0, -1].float()
            else:
                seq = (seq + [nxt])[-maxlen:]
    raw = bytes(max(0, t - BYTE_OFFSET) for t in done if BYTE_OFFSET <= t < 260)
    return done, decode_char_boundary(raw).decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", type=str, default=None, help="raw text prompt (else first --data-root doc)")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--tokens", type=int, default=240)
    ap.add_argument("--maxlen", type=int, default=700)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--mode", choices=["stream", "window"], default="stream")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--rep-pen", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    print(f"[load] step={ck['step']} ema={ck.get('ema_loss')} cfg dim={cfg['dim']} L={cfg['layers']} evolve={cfg.get('evolve', 0)}")
    model = M.RingKoSSM(cfg["dim"], cfg["layers"], evolve=bool(cfg.get("evolve", 0)),
                        conv_k=cfg.get("conv_k", 4), scan_floor=cfg.get("scan_floor", 1e-2)).to("cuda").eval()
    model.load_state_dict(ck["state_dict"])

    if args.prompt is not None:
        pr = encode_fast(args.prompt).tolist()
    else:
        if args.data_root is None:
            raise SystemExit("need --prompt or --data-root")
        pr = first_prompt(args.data_root)
    print(">>> " + decode(pr[:200]))

    torch.set_grad_enabled(False)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        xb = torch.tensor(pr, dtype=torch.long, device="cuda").unsqueeze(0)
        lg, _ = model(xb)
    top = lg[0, -1].float().topk(12)
    print("[diag] prompt末尾 top-12:")
    for tid, pv in zip(top.indices.tolist(), top.values.tolist()):
        ch = bytes([max(0, tid - BYTE_OFFSET)]).decode("utf-8", "ignore") if BYTE_OFFSET <= tid < 260 else "<?>"
        print(f"   id={tid:4d} p={pv:.3f} byte=0x{tid - BYTE_OFFSET:02x} '{ch}'")

    gen, text = generate(model, pr, args.tokens, args.temp, args.topk, args.seed,
                         args.maxlen, rep_pen=args.rep_pen, mode=args.mode, precision=args.precision)
    print(f"\n--- {args.mode} temp={args.temp} topk={args.topk} ---")
    print(text)


if __name__ == "__main__":
    main()
