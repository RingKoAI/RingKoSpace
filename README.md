# RingKoSpace

Single-line **liquid-gated linear recurrence** byte language model — residual-free,
FFN-free, one recurrent state per block. Working codename: **Tibo**.

> **Terminology nail (source-level, read this first).**
> The model is *not* "CfC". Code carries zero CfC-specific pieces (state-cascaded
> backbone, explicit `timespan` Δt input, ff1/ff2 two-candidate interpolation —
> none present in `src/ringkospace/model.py`). Its recurrence, chunked f64
> log-space scan and decay all descend from the **RNCM PURE DUAL SSM branch**;
> the parameterization (`conv1d + SiLU + A_log/Δ + D·x`) is the Mamba recipe.
> See `docs/DESIGN-v0.md` §1/§8 and the evolution note in `docs/DESIGN-evolve.md`.

```text
z_t     = SiLU( Wz · conv(x) )                    # content candidate  (conv = short memory)
Δ_t     = softplus( Wd · x + bd )                 # input-dependent step
a_t     = exp( −Δ_t · exp(A_log) )                # per-channel keep gate (liquid)
h_t     = a_t · h_{t-1} + (1 − a_t) · z_t         # ONE state, window-level gradients
y_t     = Wr · LN(h_t) · scale + dv ⊙ x           # read + current-byte D·x skip
```

* No residual, no FFN, no attention. History is **not** detached: the recurrence
  is parallelized with a chunked f64 log-space scan (window-level BPTT).
* State is an activation, not parameters — *memory is the shadow of a learning
  rule*, there is no external memory store.

## Why it exists

`field/volume` (matrix-state memory, EMA decay + softmax features) plateaued at
**eval 2.958**, never breaking byte unigram entropy — a single-step key pinned it
to 1-gram lookback. RingKoSpace keeps the same *shape* of recurrent update but
fixes the three pathologies: learned per-channel input-dependent `a_t` (not a
constant 0.99), a nonlinear conv content candidate (not a softmax outer product),
and full history gradients (not `detach`-every-step).

## Results (all reproducible, same-domain baselines)

| run | config | tokens | final eval | vs byte 2-gram |
|---|---|---|---|---|
| v0 probe (RTX 4060) | dim512/L8 | 7.4M | **2.004** | breaks 2.031 |
| A dim512/6.5M | dim512/L8/seq512 | 15M | **1.795** | breaks |
| **A 68M** (RTX 5090) | dim1280/L14/seq768 | 98M / 42 min | **1.2782** | 1-gram 4.065 / 2-gram 2.834 → eats past 2-gram; BPB 1.844 ≈ 4-gram entropy |

Cross-window memory (`carry` probe, window=256): dim256 ≈ 0 → 6.5M **−0.009** →
68M **−0.019/−0.036** — cross-window state use grows with scale.

### Judged-out mechanisms (kept as honest negatives)

* EMA slow-target (same-position softmax distillation): eval 3.53 vs A 1.795 → **no**.
* Sketch LSH read-index: recall ≈ noise → **no** ("database memory", against the
  no-warehouse philosophy). Report: `docs/sketch-report.md`.
* Random-binding recall protocol: A/B/C/D all ≈ noise → protocol measures
  *database* recall, not the target "reuse-shaping" memory.

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# jsonl text docs (skypile-style: {"text": "..."}) with an EOS per doc:
ringkospace-train --steps 1800 --dim 512 --layers 8 \
    --seq 256 --batch 16 --max-tokens 9000000 --data-root ./data/skypile

# or a mined uint16 stream (doc-shuffled once, streamed sequentially):
python tools/mine_bin.py --input-root ./data/skypile --out corpus.bin
ringkospace-train --steps 16000 --dim 1280 --layers 14 --seq 768 \
    --batch 8 --max-tokens 100000000 --bin corpus.bin --eval-every 1000

# generate (window vs stream) from a checkpoint:
ringkospace-generate --ckpt ckpt/step-15000.pt --data-root ./data/skypile --tokens 240
```

Each training run ends with an automatic **carry probe** on two held-out 2048-token
streams — the reported `delta` (carry − from-zero) is the cross-window memory signal.

## Checkpoints & evaluation scope (read before trusting the numbers)

No pretrained weights live in this repo (ckpt/ is gitignored; 68M is ~277MB/step
and released separately if ever). Every number above was produced under a
**narrow, single-database regime** — be aware of the gap between probe results
and real multi-source behaviour:

- The 68M run (final eval 1.2782, carry −0.019/−0.036, BPB 1.844) trained on the
  **first ~300M tokens of one mined corpus stream** (dominantly government/news
  style, ~37% title-like lines). It is a *single-distribution* result, not a
  multi-source one. On real mixed corpora (web/wiki/code/math, other domains)
  loss, generation quality and cross-window memory **can diverge** from these
  figures. Do not cite 1.2782 as multi-source performance.
- Held-out eval = same-domain byte CE vs same-domain n-gram baselines (1-gram
  4.065 / 2-gram 2.834 on that eval segment). Numbers are only comparable within
  this eval-set convention.
- Generation was assessed **windowed** (model never trained for single-step
  stream roll-out), so outputs carry exposure bias — see docs/SUMMARY-20260906.md.
- Reported negatives (EMA distillation, Sketch, --evolve at short budget) are
  equally single-regime; they were rejected as main-line on that evidence, not on
  a universal proof.

Next-gen training (four-source weighted interleave + streaming-state targets) is
exactly what addresses the single-db gap — see Roadmap.

## Repository layout

```text
src/ringkospace/       model.py · data.py · train.py · generate.py
tools/                 mine_bin.py, probe/diagnostic scripts
tests/                 scan↔sequential numerical equivalence, encode/decode
docs/                  DESIGN-v0.md (equations, fixes, term nail), SUMMARY, sketch-report
```

## Roadmap (in order, each behind a switch + same-budget A/B)

1. `--evolve` single-line fusion: the content candidate linearly evolves the old
   state too (`h = [keep+(1−keep)·a_evo]·h + (1−keep)(1−a_evo)·z`, still closed-form
   scan). Ablation ladder g=1 / g=const / g(x) / g(x,h,Δt). **Status: A/B running.**
2. Streaming-state training (carry injection probability + cross-window targets) —
   cures the exposure-bias of window-only training (v0 "closes its eyes" when asked
   to roll out one byte at a time).
3. Four-source weighted interleave (skypile/wiki/code/math) + 128M run; BPB +
   relative n-gram as the reporting currency.
4. Reuse-shaping memory task (only the used survives) replaces the database-style
   recall protocol.

## Design discipline (from the field/volume post-mortem)

* No residual shortcut: it became an unigram scoring crutch under byte-CE.
* Every mechanism is an independent switch; nothing is adopted without a same-budget,
  same-domain A/B that improves the *curve shape* (smooth descent), not just the headline.
* No claim of beating anything without the same eval set and domain n-gram baselines.
* External input is untrusted: jsonl decode errors and half-UTF-8 are handled, docs are
  length-capped, prompts are trimmed at character boundaries.

## License

MIT — Copyright (c) 2026 辉夜铃Ring (KaguyaRing) && 玲可AI (RingKoAI).
Parameterization ideas inspired by Mamba (Gu & Dao, Apache-2.0); all code in
this repo is independently implemented — see `NOTICE`. Checkpoints (68M) are
released separately, not in this tree.
