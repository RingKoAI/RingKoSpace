"""Decode-method sweep on an existing RingKoSpace checkpoint (windowed AR).

Each method gets the same clean prompt and token budget; outputs printed so
the exposure-bias/repetition behavior across sampling algorithms is visible.
"""
from __future__ import annotations
from pathlib import Path as _P
import sys as _s
_s.path.insert(0, str(_P(__file__).resolve().parent.parent / "src"))

import argparse, torch
import torch.nn.functional as F
from ringkospace.model import RingKoSSM
from ringkospace.data import BYTE_OFFSET, decode, encode_fast
from ringkospace.generate import first_prompt

BOS = 1


@torch.inference_mode()
def gen(model, prompt, n, maxlen, method, top_k=0, p=0.0, temp=1.0,
        rep_pen=1.0, no_repeat=0, contrastive_k=0, contrastive_alpha=0.6,
        seed=0, embed=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq = prompt[:maxlen - 1]
    done: list[int] = []
    enc = lambda toks: torch.tensor(toks, dtype=torch.long, device="cuda").unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(n):
            lg, _ = model(enc(seq[:maxlen]))
            logits = lg[0, -1].float()
            if rep_pen > 1.0:
                for t in set(done):
                    logits[t] = logits[t] / rep_pen if logits[t] > 0 else logits[t] * rep_pen
            if no_repeat > 1:
                hist = seq + done
                if len(hist) >= no_repeat:
                    grams = set()
                    for i in range(len(hist) - no_repeat + 1):
                        grams.add(tuple(hist[i:i + no_repeat]))
                    base = hist[-(no_repeat - 1):] if len(hist) >= no_repeat - 1 else []
                    for tid in range(260):
                        if tuple(base + [tid]) in grams:
                            logits[tid] = float("-inf")
            if method == "greedy":
                nxt = int(torch.argmax(logits).item())
            else:
                logits = logits / temp
                if method == "contrastive":
                    k = contrastive_k or top_k
                    v, idx = torch.topk(logits, k)
                    probs = F.softmax(v, dim=-1)
                    cand = F.embedding(idx, embed)          # [k, D]
                    hist = torch.tensor(hist if False else (seq + done)[-maxlen:],
                                        device="cuda", dtype=torch.long)
                    past = F.embedding(hist, embed)        # [h, D]
                    sim = F.normalize(cand, dim=-1) @ F.normalize(past, dim=-1).T
                    maxsim, _ = sim.max(dim=1)
                    logp = probs.log()
                    score = (1 - contrastive_alpha) * logp - contrastive_alpha * maxsim
                    nxt = int(idx[score.argmax()].item())
                else:
                    if top_k > 0:
                        v, idx = torch.topk(logits, min(top_k, logits.shape[-1]))
                        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, v)
                    probs = torch.softmax(logits, dim=-1)
                    if p > 0 and method != "topk":
                        sp, si = torch.sort(probs, descending=True)
                        cum = torch.cumsum(sp, dim=0)
                        mask = cum - sp > p
                        sp[mask] = 0.0
                        probs = torch.zeros_like(probs).scatter(-1, si, sp)
                        if probs.sum() <= 0:
                            probs = F.softmax(logits, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=g).item())
            done.append(nxt)
            seq = seq + [nxt]
            if len(seq) > maxlen:
                seq = seq[-maxlen:]
    return done, decode(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/step-15999.pt")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--tokens", type=int, default=260)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-text", type=str, default=None)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ck['config']
    print(f"[load] step={ck['step']} cfg dim={cfg['dim']} L={cfg['layers']} evolve={cfg.get('evolve', 0)}")
    model = RingKoSSM(cfg['dim'], cfg['layers'], evolve=bool(cfg.get('evolve', 0)), conv_k=cfg.get('conv_k', 4), scan_floor=cfg.get('scan_floor', 1e-2)).to("cuda").eval()
    model.load_state_dict(ck['state_dict'])
    embed = model.embedding.weight.detach()

    if args.prompt_text:
        prompt = encode_fast(args.prompt_text).tolist()
    else:
        if args.data_root is None:
            raise SystemExit("need --prompt-text or --data-root")
        prompt = first_prompt(args.data_root)
    print("\n>>> " + decode(prompt[:160]) + "\n")

    battery = [
        dict(name="greedy", method="greedy"),
        dict(name="topk t.8 k40", method="topk", temp=0.8, top_k=40),
        dict(name="topp t.8 p0.92", method="topp", temp=0.8, p=0.92),
        dict(name="topp + rep1.2", method="topp", temp=0.9, p=0.92, rep_pen=1.2),
        dict(name="topp + rep1.2 + no4gram", method="topp", temp=0.9, p=0.92, rep_pen=1.2, no_repeat=4),
        dict(name="contrastive k16 a.6", method="contrastive", contrastive_k=16, contrastive_alpha=0.6),
    ]
    for cfg in battery:
        tok, text = gen(model, prompt, args.tokens, args.maxlen,
                        embed=embed, seed=args.seed, **{k: v for k, v in cfg.items() if k != "name"})
        print(f"\n===== {cfg['name']} =====")
        print(text)


if __name__ == "__main__":
    main()
