"""In-memory affine 2-bit copies of part of each Qwen routed expert.

Installed expert files stay MXFP4. After a blob is read into a slot, the cache
keeps this smaller copy and releases the MXFP4 buffer. The configured percentage
selects a prefix of each matrix, in groups of 64 input columns, rounded to the
nearest group. A share above 0% quantizes at least one group. The remaining
columns stay MXFP4. Products have no 2-bit kernel, so each one restores the
quantized prefix, multiplies, and drops that copy before the MXFP4 tail. A group
stores its minimum as a BF16 bias, the gap to its maximum as a BF16 scale, and
two bits per weight. Equal values store zeros and a zero scale. Output tokens
can differ from the MXFP4 expert.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .expert_cache import QwenBatchedExperts, QwenExpertWeights
from .qwen_flash_config import EXPERT_QUANT_GROUP, quantized_expert_columns

GROUP_SIZE = EXPERT_QUANT_GROUP
MXFP4_GROUP_SIZE = 32
MXFP4_BITS = 4
# Two-bit place values. The low bits of each byte are the first weight.
_CODE_PLACES = (1, 4, 16, 64)


@dataclass(frozen=True)
class TwoBitQwenExpert:
    gate_up: mx.array | None
    gate_up_scales: mx.array | None
    gate_up_biases: mx.array | None
    gate_up_columns: int
    gate_up_mx: mx.array | None
    gate_up_mx_scales: mx.array | None
    down: mx.array | None
    down_scales: mx.array | None
    down_biases: mx.array | None
    down_columns: int
    down_mx: mx.array | None
    down_mx_scales: mx.array | None


def pack_affine_2bit(weight: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Pack one matrix into 2-bit codes, BF16 scales and BF16 biases.

    Four levels span the group's minimum and maximum. A constant group, including
    a negative one, stays all zeros with a zero scale.
    """
    rows, cols = weight.shape
    if cols % GROUP_SIZE:
        raise ValueError(f"2-bit expert weights need an input size multiple of {GROUP_SIZE}")
    groups = cols // GROUP_SIZE
    values = weight.astype(mx.float32).reshape(rows, groups, GROUP_SIZE)
    lo = mx.min(values, axis=-1)
    hi = mx.max(values, axis=-1)
    span = hi - lo
    # Exact levels 0, 1, 2 and 3. A zero span is replaced after the division.
    ratio = (values - lo[..., None]) * 3 / mx.maximum(span, mx.array(1, dtype=mx.float32))[..., None]
    codes = mx.clip(mx.round(ratio), 0, 3)
    codes = mx.where(span[..., None] == 0, mx.zeros_like(codes), codes).astype(mx.int32)
    codes = codes.reshape(rows, cols // 4, 4)
    places = mx.array(_CODE_PLACES, dtype=mx.int32)
    packed = mx.sum(codes * places, axis=-1).astype(mx.uint8)
    return packed, span.astype(mx.bfloat16), lo.astype(mx.bfloat16)


def unpack_affine_2bit(packed: mx.array, scales: mx.array, biases: mx.array, dtype) -> mx.array:
    """Restore ``bias + (code / 3) * scale``. Codes 0 and 3 are the group ends."""
    rows, packed_cols = packed.shape
    cols = packed_cols * 4
    groups = cols // GROUP_SIZE
    places = mx.array(_CODE_PLACES, dtype=mx.int32)
    expanded = packed.astype(mx.int32)[..., None]
    codes = mx.remainder(mx.floor_divide(expanded, places), 4)
    codes = codes.reshape(rows, groups, GROUP_SIZE).astype(mx.float32)
    scale = scales.astype(mx.float32).reshape(rows, groups, 1)
    bias = biases.astype(mx.float32).reshape(rows, groups, 1)
    # Dividing the code by 3 makes code 3 an exact 1, so the maximum is preserved.
    restored = bias + codes / mx.array(3, dtype=mx.float32) * scale
    return restored.astype(dtype).reshape(rows, cols)


def quantize_qwen_expert(weights: QwenExpertWeights, percent: int) -> TwoBitQwenExpert | QwenExpertWeights:
    """Dequantize one resident MXFP4 expert and keep the mixed 2-bit copy.

    Returns the original weights when the share rounds to no columns, so the
    caller can keep the MXFP4 slot.
    """
    gate = _quantize_matrix(weights.gate_up, weights.gate_up_scales, percent)
    down = _quantize_matrix(weights.down, weights.down_scales, percent)
    if gate is None and down is None:
        return weights
    if gate is None:
        gate = (None, None, None, 0, _retain(weights.gate_up), _retain(weights.gate_up_scales))
    if down is None:
        down = (None, None, None, 0, _retain(weights.down), _retain(weights.down_scales))
    packed = TwoBitQwenExpert(*gate, *down)
    live = [array for array in (
        packed.gate_up, packed.gate_up_scales, packed.gate_up_biases,
        packed.gate_up_mx, packed.gate_up_mx_scales,
        packed.down, packed.down_scales, packed.down_biases,
        packed.down_mx, packed.down_mx_scales,
    ) if array is not None]
    mx.eval(*live)
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


def apply_two_bit_expert(value: mx.array, weights: TwoBitQwenExpert) -> mx.array:
    """SwiGLU expert. Each restored prefix is released before the next matrix."""
    import mlx.nn as nn
    projected = _project(
        value, weights.gate_up, weights.gate_up_scales, weights.gate_up_biases,
        weights.gate_up_columns, weights.gate_up_mx, weights.gate_up_mx_scales)
    gate, up = mx.split(projected, 2, axis=-1)
    return _project(
        nn.silu(gate) * up, weights.down, weights.down_scales, weights.down_biases,
        weights.down_columns, weights.down_mx, weights.down_mx_scales)


def _dequantize_mxfp4(weight: mx.array, scales: mx.array) -> mx.array:
    return mx.dequantize(weight, scales, group_size=MXFP4_GROUP_SIZE, bits=MXFP4_BITS, mode="mxfp4")


def _quantize_matrix(weight: mx.array, scales: mx.array, percent: int):
    dense = _dequantize_mxfp4(weight, scales)
    columns = int(dense.shape[1])
    quantized = quantized_expert_columns(columns, percent)
    if quantized == 0:
        return None
    packed, q_scales, q_biases = pack_affine_2bit(dense[:, :quantized])
    if quantized == columns:
        return packed, q_scales, q_biases, quantized, None, None
    tail, tail_scales = _slice_mxfp4(weight, scales, quantized, columns)
    return packed, q_scales, q_biases, quantized, tail, tail_scales


def _slice_mxfp4(weight: mx.array, scales: mx.array, start: int, columns: int):
    packed_columns = int(weight.shape[-1])
    if packed_columns * 8 == columns:
        values_per_unit = 8
    elif packed_columns * 2 == columns:
        values_per_unit = 2
    else:
        raise ValueError("MXFP4 expert weight does not match its column count")
    if int(scales.shape[-1]) * MXFP4_GROUP_SIZE != columns:
        raise ValueError("MXFP4 expert scales do not match its column count")
    return (
        _retain(weight[:, start // values_per_unit:]),
        _retain(scales[:, start // MXFP4_GROUP_SIZE:]),
    )


def _retain(array: mx.array) -> mx.array:
    """Copy a slot slice so it survives when the MXFP4 slot is released."""
    retained = mx.array(np.array(array, copy=True))
    mx.eval(retained)
    return retained


def _project(value, packed, scales, biases, columns, mx_weight, mx_scales):
    if packed is None:
        return _qmm(value, mx_weight, mx_scales)
    quantized = _matmul(value[..., :columns], packed, scales, biases)
    if mx_weight is None:
        return quantized
    return quantized + _qmm(value[..., columns:], mx_weight, mx_scales)


def _qmm(value: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.quantized_matmul(
        value, weight, scales, transpose=True, group_size=MXFP4_GROUP_SIZE,
        bits=MXFP4_BITS, mode="mxfp4")


def _matmul(value: mx.array, packed: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    output = value @ unpack_affine_2bit(packed, scales, biases, value.dtype).T
    mx.eval(output)
    # The restored prefix is only an input to this product.
    mx.clear_cache()
    return output
