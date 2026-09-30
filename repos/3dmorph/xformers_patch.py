"""Blackwell(sm_120)向け: xformersがHopper専用のFA3を選ばないよう、計算方式を固定する"""
import os
import xformers.ops as xops
from xformers.ops import fmha

_OPS = {
    "cutlass": fmha.MemoryEfficientAttentionCutlassOp,
    "flash":   fmha.MemoryEfficientAttentionFlashAttentionOp,
}
_op = _OPS[os.environ.get("XFORMERS_OP", "cutlass")]
_orig = xops.memory_efficient_attention

def _patched(*args, **kwargs):
    kwargs.setdefault("op", _op)
    return _orig(*args, **kwargs)

xops.memory_efficient_attention = _patched
