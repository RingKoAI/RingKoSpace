# Streaming and checkpoint semantics

Use `model.forward_stream(tokens, state=None)` for a contiguous stream. The returned state contains each layer's recurrence and causal-convolution history. Pass it explicitly to the next chunk; pass None for a new request. Call `state.detach()` only at intentional truncated-BPTT boundaries. Do not mutate state tensors or reuse state after modifying model weights, device or dtype. State is request-local and bound to one model instance, not a portable memory-file format.

The legacy `model(tokens, states)` interface only seeds recurrence; it does not carry convolution history. `_latent` and `_gates` are last-call diagnostics, not concurrent request-local outputs. Concurrent weight mutation/training is unsupported.

NoPE remains: causal convolution and recurrence preserve order without explicit positional embeddings. The existing learned dv*x bypass remains.

`scan_floor` is checkpoint semantics, independent of execution backend. New training runs explicitly save 0 (exact recurrence). Legacy RCMS defaults are 0.001 for Torch and 0 for Triton; legacy RingKoSpace defaults to 0.01. Load this setting from checkpoint config; do not silently migrate a trained model's recurrence. RCMS resume inherits it and rejects a conflicting override.

Torch uses division-free affine prefix operations within 32-token chunks with sequential chunk carries. Triton is parallel across batch/channels and sequential across time within each chunk; it supports first-order backward including the final carry and nonzero initial state. Neither implementation claims fully parallel streaming time steps or improved training sample efficiency.

The main generation entry offers `--mode stream` and `--precision fp32|bf16` (existing BF16 default preserved). FP32 regression matches full growing-prefix inference within numerical tolerance. BF16 can change logits and greedy choices with chunk shape; use FP32 with TF32 disabled for strict comparisons. A sliding finite window and a growing stream intentionally differ after window eviction. Training still uses independent windows.
