"""RingKoSpace: single-line liquid-gated linear recurrence byte language model.

Lineage: RNCM PURE DUAL (CfC + SSM dual-stream validation) ->
field/volume (matrix-state, plateau 2.96, discarded) ->
RingKoSpace (residual-free gated linear recurrence).

Architecture in one equation (per channel, per block, one state h):

    z     = SiLU(Wz · conv(x))                    # content candidate
    Delta = softplus(Wd · conv(x) + bd)           # input-dependent step
    a     = exp(-Delta · exp(A_log))              # per-channel keep gate
    h_t   = a · h_{t-1} + (1 - a) · z_t           # single-state recurrence
    y_t   = Wr · LN(h_t) · scale + dv ⊙ x          # read + current-byte D·x

No residual, no FFN. History is NOT detached: the recurrence is parallelized
with a chunked f64 log-space scan (window-level gradients, Mamba-style).

Terminology nail (source-level, do not call this "CfC"):
the CfC-specific pieces (state-cascaded backbone, explicit timespan Delta-t,
ff1/ff2 two-candidate interpolation) are NOT present in the code. The decay
and chunked-scan skeleton are inherited from the RNCM SSM branch; the
parameterization (conv1d + SiLU + A_log/Delta + D·x) is the Mamba recipe.
"""

from .model import LiquidSSMBlock, PAD, RingKoSSM, VOCAB

__version__ = "0.1.0"

__all__ = ["LiquidSSMBlock", "PAD", "RingKoSSM", "VOCAB", "__version__"]
