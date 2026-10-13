"""Opt-in resident-weight formats for the Qwen engine.

Every BF16 projection with at least 256 output rows, ``lm_head`` and the MTP
layer included, can be repacked after load. The routers (``mlp.gate``), the
embedding, smaller gates, convolutions and norms stay BF16. Experts stay in
their installed packed format.

8-bit stores affine groups of 64 and is the App default. Products of up to 16
rows use MLX's 8-bit kernel. Larger products multiply by a temporary BF16 copy,
because that kernel is the slower one at that size.

1-bit stores the same projections as affine groups of 64: each group keeps its
minimum as a BF16 bias, the gap to its maximum as a BF16 scale, and one bit per
weight (0 at the minimum, 1 at the maximum). A group whose values are equal
stores zeros and a zero scale. There is no 1-bit kernel, so each product
restores at most ``ONE_BIT_OUTPUT_CHUNK`` output rows to BF16 and releases that
copy before the next chunk. Output tokens can differ from the BF16 model.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

BITS = 8
GROUP_SIZE = 64
MINIMUM_ROWS = 256
DECODE_ROWS = 16
# One restored slice of a 1-bit projection. MemoryPlanning uses this same count.
ONE_BIT_OUTPUT_CHUNK = 4096
_BIT_PLACES = (1, 2, 4, 8, 16, 32, 64, 128)


class CommonQuantizedLinear(nn.QuantizedLinear):
    """8-bit weights; products of more than 16 rows use a temporary BF16 copy."""

    def __call__(self, x: mx.array) -> mx.array:
        if x.size // x.shape[-1] <= DECODE_ROWS:
            return super().__call__(x)
        weight = mx.dequantize(self["weight"], self["scales"], self.get("biases"),
                               group_size=self.group_size, bits=self.bits, mode=self.mode)
        if "bias" in self:
            return mx.addmm(self["bias"], x, weight.T)
        return x @ weight.T


class OneBitLinear(nn.Module):
    """Affine 1-bit weights with per-group BF16 scale and bias."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array, bias=None):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases
        if bias is not None:
            self.bias = bias
        self.bits = 1
        self.group_size = GROUP_SIZE
        self.mode = "affine"

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "OneBitLinear":
        weight = linear.weight
        if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE != 0:
            raise ValueError(
                "1-bit resident weights need a matrix whose input size is a multiple of "
                f"{GROUP_SIZE}"
            )
        packed, scales, biases = pack_affine_1bit(weight)
        bias = linear["bias"] if "bias" in linear else None
        return cls(packed, scales, biases, bias)

    def __call__(self, x: mx.array) -> mx.array:
        rows = self.scales.shape[0]
        bias = self["bias"] if "bias" in self else None
        if rows <= ONE_BIT_OUTPUT_CHUNK:
            return _apply(x, unpack_affine_1bit(
                self.weight, self.scales, self.biases, x.dtype), bias)
        parts = []
        for start in range(0, rows, ONE_BIT_OUTPUT_CHUNK):
            end = min(start + ONE_BIT_OUTPUT_CHUNK, rows)
            part = _apply(
                x,
                unpack_affine_1bit(self.weight[start:end], self.scales[start:end],
                                   self.biases[start:end], x.dtype),
                None if bias is None else bias[start:end],
            )
            # Drop this slice before the next one is built. The concatenated
            # result is the real output; the restored weights are not retained.
            mx.eval(part)
            parts.append(part)
        return mx.concatenate(parts, axis=-1)


def pack_affine_1bit(weight: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Pack one matrix into bits, BF16 scales and BF16 biases.

    Bit 0 of each byte is the first weight in that byte. A weight above the
    group's midpoint is 1. The midpoint and every smaller value are 0.
    """
    rows, cols = weight.shape
    groups = cols // GROUP_SIZE
    values = weight.astype(mx.float32).reshape(rows, groups, GROUP_SIZE)
    lo = mx.min(values, axis=-1)
    hi = mx.max(values, axis=-1)
    # Midpoint stays at lo when the group is constant, including negative values.
    midpoint = lo[..., None] + (hi - lo)[..., None] * 0.5
    bits = (values > midpoint).astype(mx.int32)
    bits = bits.reshape(rows, cols // 8, 8)
    places = mx.array(_BIT_PLACES, dtype=mx.int32)
    packed = mx.sum(bits * places, axis=-1).astype(mx.uint8)
    return packed, (hi - lo).astype(mx.bfloat16), lo.astype(mx.bfloat16)


def unpack_affine_1bit(packed: mx.array, scales: mx.array, biases: mx.array,
                       dtype) -> mx.array:
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


def quantize_common_tensors(model: nn.Module) -> int:
    """Replace the chosen BF16 projections with 8-bit weights.

    One projection at a time: its BF16 weight is released as soon as the 8-bit
    form replaces it, so loading never holds a second full copy.
    """
    return _repack(model, lambda linear: CommonQuantizedLinear.from_linear(
        linear, group_size=GROUP_SIZE, bits=BITS))


def quantize_1bit_common_tensors(model: nn.Module) -> int:
    """Replace the chosen BF16 projections with 1-bit weights.

    One projection at a time, same as the 8-bit repack, so loading never holds
    a second full copy.
    """
    return _repack(model, OneBitLinear.from_linear)


def _resident_projection_paths(model: nn.Module) -> list[str]:
    return [path for path, module in model.named_modules()
            if isinstance(module, nn.Linear) and module.weight.shape[0] >= MINIMUM_ROWS
            and not path.endswith("mlp.gate")]


def _repack(model: nn.Module, convert) -> int:
    paths = _resident_projection_paths(model)
    for path in paths:
        parent, key = _parent_and_key(model, path)
        try:
            packed = convert(parent[key])
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        mx.eval(packed.parameters())
        parent[key] = packed
    mx.clear_cache()
    return len(paths)


def _parent_and_key(model: nn.Module, path: str):
    *names, name = path.split(".")
    parent = model
    for part in names:
        parent = parent[int(part)] if part.isdigit() else parent[part]
    return parent, int(name) if name.isdigit() else name


def _apply(x: mx.array, weight: mx.array, bias) -> mx.array:
    if bias is not None:
        return mx.addmm(bias, x, weight.T)
    return x @ weight.T
