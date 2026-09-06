"""RingKoSpace recall-pressure probe. Synthetic key-value memory protocol.

Stream bytes contain three ASCII sub-patterns:
    M E M <id3> : <content4>    store
    <random letter/digit noise> 
    Q <id3> ?                    ask
    <content4>                   supervised answer

Plain next-byte CE over the stream is the memory-pressure gradient natural
text refuses to give. Metric: top-1 accuracy at the 4 content bytes, bucketed
by (fact marker -> query) distance.

Variants: A = gated linear recurrence (ringkospace.model.RingKoSSM). C adds an
EMA teacher (weights) + a latent predictor trained with a JEPA-style target
(predict future-step logits/latent of the no-grad EMA teacher). B/D (Sketch)
staged later. This protocol measures DATABASE recall and is kept only as a
baseline; the target "reuse-shaping" memory task is a roadmap item.
"""
from __future__ import annotations
from pathlib import Path as _P
import sys as _s
_s.path.insert(0, str(_P(__file__).resolve().parent.parent / "src"))

import argparse, math, random, time
import torch, torch.nn.functional as F
from torch import Tensor, nn

PAD, BOS = 0, 1
ID_SET = "0123456789"
CT_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NOISE_A, NOISE_B = 4 + 97, 4 + 123   # ascii lowercase a..z, disjoint from markers
DIST_CHOICES = (64, 128, 192, 256)
MAX_FACTS = 3


def enc(s: str):
    return [b + 4 for b in s.encode()]


def make_sample(seq: int, rng: random.Random):
    """Return (token_ids [seq], facts [(fpos, qpos, fid, fc, dist)])."""
    out: list[int] = [BOS]
    facts: list[tuple[int, int, str, str, int]] = []
    placed = 0
    while placed < MAX_FACTS and len(out) + 9 + 4 + max(DIST_CHOICES) + 10 < seq:
        dist = rng.choice(DIST_CHOICES)
        fid = "".join(rng.choice(ID_SET) for _ in range(3))
        fc = "".join(rng.choice(CT_SET) for _ in range(4))
        fpos = len(out)
        out += enc("MEM") + enc(fid) + enc(":") + enc(fc)
        # noise until we are `dist` tokens after the fact marker 'M'
        while len(out) - fpos < dist:
            out.append(rng.randrange(NOISE_A, NOISE_B))
        qpos = len(out)
        out += enc("Q") + enc(fid) + enc("?")
        out += enc(fc)
        facts.append((fpos, qpos, fid, fc, qpos - fpos))
        placed += 1
    while len(out) < seq:
        out.append(rng.randrange(NOISE_A, NOISE_B))
    return torch.tensor(out[:seq], dtype=torch.long), facts


def recall_accuracy(run, model, device, seq, buckets, n_seqs=4, seed=0):
    model.eval()
    acc = {b: [0, 0] for b in buckets}
    with torch.no_grad():
        for i in range(n_seqs):
            x, facts = make_sample(seq, random.Random(seed * 100 + i))
            lg, _ = model(x.to(device).unsqueeze(0))
            lg = lg[0]
            for (fpos, qpos, fid, fc, dist) in facts:
                bucket = min(buckets, key=lambda z: abs(z - dist))
                cstart = qpos + len("Q") + 3 + len("?")  # = content start
                pred = lg[cstart:cstart + 4].argmax(-1)
                truth = torch.tensor(enc(fc), device=device)
                hits = (pred == truth).sum().item()
                acc[bucket][0] += hits
                acc[bucket][1] += 4
    model.train()
    return {b: (h / m if m else float("nan")) for b, (h, m) in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "C"], default="A")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--eval-seqs", type=int, default=8)
    ap.add_argument("--jepa-w", type=float, default=1.0)
    ap.add_argument("--use-sketch", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    dev = "cuda"
    from ringkospace.model import RingKoSSM

    model = RingKoSSM(args.dim, args.layers).to(dev).train()
    sketch = None
    if args.use_sketch:
        from sketch_read import SketchRead
        sketch = SketchRead(args.dim).to(dev).train()
    def run_net(x):
        logits, st = model(x)
        if sketch is not None:
            logits = logits + sketch(model._latent)
        return logits, st
    params = list(model.parameters())
    if sketch is not None:
        params = params + list(sketch.parameters())
    predictor = None
    teacher = None

    if args.variant == "C":
        teacher = RingKoSSM(args.dim, args.layers).to(dev)
        teacher.load_state_dict(model.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)
        predictor = nn.Sequential(nn.Linear(260, args.dim), nn.LayerNorm(args.dim),
                                  nn.GELU(), nn.Linear(args.dim, 260)).to(dev)
        params = params + list(predictor.parameters())

    print(f"[model] variant={args.variant} params={sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / max(args.warmup, 1)
        p = (s - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * 0.1 + 0.5 * (args.lr - args.lr * 0.1) * (1 + math.cos(math.pi * min(p, 1.0)))

    buckets = sorted(set(DIST_CHOICES))
    start = time.perf_counter()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        rng = random.Random(args.seed * 7919 + step * 101)
        xb = torch.stack([make_sample(args.seq, rng)[0] for _ in range(args.batch)]).to(dev)
        tgt = torch.roll(xb, -1, dims=1)
        tgt[:, -1] = PAD
        logits, _ = run_net(xb)
        ce = F.cross_entropy(logits.reshape(-1, 260), tgt.reshape(-1), ignore_index=PAD)

        loss = ce
        if args.variant == "C":
            with torch.no_grad():
                tlg, _ = teacher(xb)
                tt = torch.softmax(tlg.detach() / 0.8, dim=-1)
            # student-predictor regresses its own logits toward the (slow) EMA
            # teacher's distribution at the SAME positions (predictor as smoother)
            pred = predictor(logits.float())
            jl = F.kl_div(F.log_softmax(pred / 0.8, dim=-1), tt, reduction="batchmean")
            loss = ce + args.jepa_w * jl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if args.variant == "C":
            with torch.no_grad():
                for p, pt in zip(model.parameters(), teacher.parameters()):
                    pt.mul_(args.ema).add_(p, alpha=1 - args.ema)

        if step % 50 == 0 or step == args.steps - 1:
            acc = recall_accuracy(run_net, model, dev, args.seq, buckets, n_seqs=args.eval_seqs)
            s = " ".join(f"{d}:{acc[d]:.2f}" for d in buckets)
            print(f"[step {step:4d}] loss={loss.item():.4f} ce={ce.item():.4f}"
                  f" recall({s}) {time.perf_counter()-start:.0f}s", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
