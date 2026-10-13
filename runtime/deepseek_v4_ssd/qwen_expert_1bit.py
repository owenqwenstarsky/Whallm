"""In-memory affine 1-bit copies of Qwen routed experts.

Installed expert files stay MXFP4. After a blob is read into a slot, the cache
keeps this smaller copy and releases the MXFP4 buffer. Products have no 1-bit
kernel, so each one restores one matrix, multiplies, and drops that copy before
the next matrix. A group stores its minimum as a BF16 bias, the gap to its
maximum as a BF16 scale, and one bit per weight. Equal values store zeros and a
zero scale. Output tokens can differ from the MXFP4 expert.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .expert_cache import QwenBatchedExperts, QwenExpertWeights

GROUP_SIZE = 64
MXFP4_GROUP_SIZE = 32
MXFP4_BITS = 4
# Bit 0 of each byte is the first weight in that byte.
_BIT_PLACES = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class OneBitQwenExpert:
    gate_up: mx.array
    gate_up_scales: mx.array
    gate_up_biases: mx.array
    down: mx.array
    down_scales: mx.array
    down_biases: mx.array


def pack_affine_1bit(weight: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Pack one matrix into bits, BF16 scales and BF16 biases.

    A weight above the group's midpoint is 1. The midpoint and every smaller
    value are 0. A constant group, including a negative one, stays all zeros.
    """
    rows, cols = weight.shape
    if cols % GROUP_SIZE:
        raise ValueError(f"1-bit expert weights need an input size multiple of {GROUP_SIZE}")
    groups = cols // GROUP_SIZE
    values = weight.astype(mx.float32).reshape(rows, groups, GROUP_SIZE)
    lo = mx.min(values, axis=-1)
    hi = mx.max(values, axis=-1)
    midpoint = lo[..., None] + (hi - lo)[..., None] * 0.5
    bits = (values > midpoint).astype(mx.int32).reshape(rows, cols // 8, 8)
    places = mx.array(_BIT_PLACES, dtype=mx.int32)
    packed = mx.sum(bits * places, axis=-1).astype(mx.uint8)
    return packed, (hi - lo).astype(mx.bfloat16), lo.astype(mx.bfloat16)


def unpack_affine_1bit(packed: mx.array, scales: mx.array, biases: mx.array, dtype) -> mx.array:
    """Restore ``q * scale + bias`` for packed affine 1-bit weights."""
    rows, packed_cols = packed.shape
    cols = packed_cols * 8
    groups = cols // GROUP_SIZE
    places = mx.array(_BIT_PLACES, dtype=mx.int32)
    expanded = packed.astype(mx.int32)[..., None]
    q = mx.remainder(mx.floor_divide(expanded, places), 2)
    q = q.reshape(rows, groups, GROUP_SIZE).astype(dtype)
    scale = scales.astype(dtype).reshape(rows, groups, 1)
    bias = biases.astype(dtype).reshape(rows, groups, 1)
    return (q * scale + bias).reshape(rows, cols)


def quantize_qwen_expert(weights: QwenExpertWeights) -> OneBitQwenExpert:
    """Dequantize one resident MXFP4 expert and keep the affine 1-bit copy."""
    gate = pack_affine_1bit(_dequantize_mxfp4(weights.gate_up, weights.gate_up_scales))
    down = pack_affine_1bit(_dequantize_mxfp4(weights.down, weights.down_scales))
    packed = OneBitQwenExpert(*gate, *down)
    mx.eval(packed.gate_up, packed.gate_up_scales, packed.gate_up_biases,
            packed.down, packed.down_scales, packed.down_biases)
    # Drop the dequantized MXFP4 copy before the caller releases the slot.
    mx.clear_cache()
    return packed


def expert_from_batched(batched: QwenBatchedExperts, expert: int) -> QwenExpertWeights:
    """One expert's MXFP4 views inside a whole-layer staging buffer."""
    return QwenExpertWeights(
        gate_up=batched.gate_up[expert],
        gate_up_scales=batched.gate_up_scales[expert],
        down=batched.down[expert],
        down_scales=batched.down_scales[expert],
    )


def apply_one_bit_expert(value: mx.array, weights: OneBitQwenExpert) -> mx.array:
    """SwiGLU expert. Each restored matrix is released before the next one."""
    projected = _matmul(value, weights.gate_up, weights.gate_up_scales, weights.gate_up_biases)
    gate, up = mx.split(projected, 2, axis=-1)
    return _matmul(nn.silu(gate) * up, weights.down, weights.down_scales, weights.down_biases)


def _dequantize_mxfp4(weight: mx.array, scales: mx.array) -> mx.array:
    return mx.dequantize(weight, scales, group_size=MXFP4_GROUP_SIZE, bits=MXFP4_BITS, mode="mxfp4")


def _matmul(value: mx.array, packed: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    output = value @ unpack_affine_1bit(packed, scales, biases, value.dtype).T
    mx.eval(output)
    # The restored matrix is only an input to this product.
    mx.clear_cache()
    return output
