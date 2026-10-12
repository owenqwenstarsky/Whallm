"""Opt-in row deduplication and FP8 lookup-table decoding for Qwen.

Produces the same BF16 values as the baseline: each FP8 byte is decoded and
scaled in NumPy, rounded to float32 and cast to BF16 by MLX. The 256 possible
results are converted once and rows are decoded on the GPU by table lookup.
This is not a speculative SSD prefetcher.
"""
import numpy as np
import mlx.core as mx


def fp8_table(weight_scale):
    values = np.arange(256, dtype=np.uint8)
    exponent = (values >> 3) & 0x0F
    mantissa = values & 0x07
    sign = np.where(values & 0x80, -1.0, 1.0)
    normal = np.ldexp(1.0 + mantissa.astype(np.float32) / 8.0,
                      exponent.astype(np.int16) - 7)
    subnormal = np.ldexp(mantissa.astype(np.float32), -9)
    decoded = sign * np.where(exponent == 0, subnormal, normal)
    decoded[(exponent == 15) & (mantissa == 7)] = np.nan
    return decoded * weight_scale


def device_table(table):
    """BF16 value of every FP8 byte, converted as a decoded row would be."""
    values = mx.array(table).astype(mx.bfloat16)
    mx.eval(values)
    return values


def decode(packed, values):
    """BF16 rows for the FP8 bytes in ``packed``; ``values`` is ``device_table``."""
    # E4M3 encodes NaN only as 0x7F and 0xFF. Checking the bytes avoids
    # decoding every row to float64 on the host first.
    if np.any((packed & 0x7F) == 0x7F):
        raise ValueError("Qwen N-gram store contains NaN")
    return mx.take(values, mx.array(packed))


def lookup_rows(rows, row_ids, values):
    if row_ids.size == 0:
        return mx.zeros((*row_ids.shape, rows.shape[1]), dtype=mx.bfloat16)
    unique, inverse = np.unique(row_ids.reshape(-1), return_inverse=True)
    # Advanced indexing already owns a copy; avoid a second host allocation.
    decoded = decode(rows[unique], values)
    restored = mx.take(decoded, mx.array(inverse.astype(np.int32)), axis=0)
    return restored.reshape(*row_ids.shape, rows.shape[1])
