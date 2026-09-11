"""Training entry point for RingKoSpace.

Streams a byte stream (jsonl docs or a mined uint16 bin), trains the
single-line liquid-gated recurrence with plain next-byte cross-entropy.
No manual lr schedule beyond warmup+cosine decay; loss *shape* (smooth
descent, no plateaus) is the accepted design signal, not the absolute number.

Flags mirror the probe lineage; every mechanism is an independent, swappable
switch so ablations stay honest (--evolve, --ema, --no-residual is structural:
the model has none).

Example (same-budget probe run, ~5 min on an RTX 4060):
    python -m ringkospace.train --steps 1800 --dim 512 --layers 8 \
        --seq 256 --batch 16 --max-tokens 9000000 --evolve 0
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

from . import model as M
from .data import load_bin, load_tokens

PAD = M.PAD


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RingKoSpace byte-LM training")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--scan-floor", type=float, default=0.0, help="explicit keep floor; 0 is exact recurrence")
    ap.add_argument("--evolve", type=int, default=0,
                    help="single-line fused evolution: candidate carries a_evo*h")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=6_000_000)
    ap.add_argument("--bin", type=str, default=None)
    ap.add_argument("--data-root", type=str, default=None,
                    help="dir of skypile-style jsonl docs when --bin is unset")
    ap.add_argument("--save-dir", type=str, default=None)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-tokens", type=int, default=262_144)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ema", type=int, default=0)
    ap.add_argument("--ema-w", type=float, default=0.3)
    ap.add_argument("--ema-warmup", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=300)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    dev = "cuda"

    if args.bin:
        train, ev = load_bin(args.bin, args.max_tokens, args.eval_tokens)
        print(f"[data] bin={args.bin} train={len(train)} eval={len(ev)}", flush=True)
    else:
        if args.data_root is None:
            raise SystemExit("set --data-root (jsonl dir) or --bin")
        data = torch.tensor(load_tokens(args.data_root, args.max_tokens + args.eval_tokens), dtype=torch.long)
        train, ev = data[: args.max_tokens], data[args.max_tokens:]
        print(f"[data] jsonl train={len(train)} eval={len(ev)}", flush=True)

    evo = bool(args.evolve)
    model = M.RingKoSSM(args.dim, args.layers, evolve=evo, scan_floor=args.scan_floor).to(dev).train()
    teacher = None
    if args.ema:
        teacher = M.RingKoSSM(args.dim, args.layers, evolve=evo, scan_floor=args.scan_floor).to(dev)
        teacher.load_state_dict(model.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)
    print(f"[model] params={model.count_parameters():,} ema={bool(args.ema)} evolve={evo}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * 0.1 + 0.5 * (args.lr - args.lr * 0.1) * (1.0 + math.cos(math.pi * min(p, 1.0)))

    per_step = args.batch * args.seq
    eval_t = ev[: (ev.numel() // (args.batch * 256)) * (args.batch * 256)].view(-1, 256).to(dev)

    def run_eval() -> float:
        model.eval()
        tot = cnt = 0.0
        with torch.no_grad():
            for xb in eval_t.split(args.batch):
                tgt = torch.roll(xb, -1, dims=1); tgt[:, -1] = PAD
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = model(xb)
                tot += F.cross_entropy(lg.float().reshape(-1, M.VOCAB), tgt.reshape(-1), ignore_index=PAD).item() * len(xb)
                cnt += len(xb)
        model.train()
        return tot / max(cnt, 1.0)

    def carry_ce(stream: torch.Tensor, use_carry: bool, win: int) -> tuple[list[float], float]:
        """CE on one long stream, window by window: fresh-from-zero windows vs
        windows fed the previous window's final state (cross-window memory)."""
        model.eval()
        n = stream.numel()
        nw = n // win
        losses: list[float] = []
        states: list[torch.Tensor] | None = None
        tot = cnt = 0.0
        with torch.no_grad():
            for w in range(nw):
                xw = stream[w * win:(w + 1) * win].to(dev).view(1, win)
                tgt = torch.roll(xw, -1, dims=1); tgt[:, -1] = PAD
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, st = model(xw, states if use_carry else None)
                lv = F.cross_entropy(lg.float().reshape(-1, M.VOCAB), tgt.reshape(-1), ignore_index=PAD).item()
                losses.append(lv)
                tot += lv * win
                cnt += win
                states = st if use_carry else None
        model.train()
        return losses, tot / max(cnt, 1.0)

    evf = ev  # carry probe uses held-out stream
    s1 = ev[: 8 * 256]
    s2 = ev[8 * 256: 16 * 256]
    have_streams = ev.numel() >= 16 * 256

    start = time.perf_counter()
    ema_loss: float | None = None
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        base = (step * per_step) % len(train)
        idx = torch.arange(base, base + per_step) % len(train)
        xb = train[idx].view(args.batch, args.seq).to(dev)
        tgt = torch.roll(xb, -1, dims=1); tgt[:, -1] = PAD
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(xb)
        ce = F.cross_entropy(lg.float().reshape(-1, M.VOCAB), tgt.reshape(-1), ignore_index=PAD)
        loss = ce
        if teacher is not None and step >= 200:
            with torch.no_grad():
                tlg, _ = teacher(xb)
                soft = torch.softmax(tlg.float(), dim=-1)
            kld = F.kl_div(F.log_softmax(lg.float(), dim=-1), soft, reduction="batchmean")
            w_eff = args.ema_w * min(1.0, (step - 200) / max(args.ema_warmup, 1))
            loss = ce + w_eff * kld
        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if teacher is not None:
            with torch.no_grad():
                m = 0.99 if step < args.ema_warmup else 0.995
                for p, pt in zip(model.parameters(), teacher.parameters()):
                    pt.mul_(m).add_(p, alpha=1 - m)
        if step % 20 == 0 or step == args.steps - 1:
            tok_s = per_step * (step + 1) / max(time.perf_counter() - start, 1e-9)
            lr = opt.param_groups[0]["lr"]
            print(f"[step {step:4d}] lr={lr:.1e} loss={loss.item():.4f} ema={ema_loss:.4f} grad={gnorm.item():.2f} tok/s={tok_s:.0f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps - 1:
            print(f"[eval {step}] eval_loss={run_eval():.4f}", flush=True)
        if args.save_dir and (step > 0) and (step % args.save_every == 0 or step == args.steps - 1):
            os.makedirs(args.save_dir, exist_ok=True)
            tmp = f"{args.save_dir}/step-{step}.pt.tmp"
            torch.save({"step": step, "ema_loss": ema_loss, "config": vars(args),
                        "state_dict": model.state_dict()}, tmp)
            os.replace(tmp, f"{args.save_dir}/step-{step}.pt")
    print(f"[done] final_eval={run_eval():.4f} elapsed={time.perf_counter()-start:.0f}s", flush=True)

    if have_streams:
        print("\n[carry] cross-window memory probe (window=256, two 2048-token streams):", flush=True)
        for name, stream in (("stream1", s1), ("stream2", s2)):
            lf, mf = carry_ce(stream, use_carry=False, win=256)
            lc, mc = carry_ce(stream, use_carry=True, win=256)
            dl = [c - f for c, f in zip(lc, lf)]
            print(f"  {name}: from-zero mean={mf:.4f} | carry mean={mc:.4f} | delta={mc - mf:+.4f}")
            print(f"    per-window carry-zero delta: " + " ".join(f"{x:+.3f}" for x in dl), flush=True)
    else:
        print("\n[carry] skipped (held-out eval stream < 16 windows)", flush=True)


if __name__ == "__main__":
    main()
