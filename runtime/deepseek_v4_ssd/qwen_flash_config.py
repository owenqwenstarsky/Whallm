"""Validation and shared CLI switches for opt-in Qwen Flash experiments."""
from __future__ import annotations

import argparse


DEFAULTS = {
    "qwen_prefill_read_experts": 1,
    "qwen_prefill_seed_experts": 0,
    "qwen_prefill_routed_max_tokens": 512,
    "qwen_packed_gdn_prefill": False,
    "qwen_sorted_expert_prefill": False,
    "qwen_expert_wave_slots": 0,
    "qwen_ngram_io": "mmap",
    "qwen_ngram_cache_bytes": 0,
    "qwen_qsa_query_chunk": 16,
    "qwen_qsa_dense_within_budget": False,
    "qwen_qsa_skip_complete_gather": True,
    "qwen_qsa_dense_threshold": 0,
    "qwen_qsa_masked_prefill": False,
    "qwen_qsa_selected_kernel": False,
    "qwen_8bit_common_tensors": False,
    "qwen_2bit_experts": False,
    "qwen_2bit_expert_percent": 50,
}

# Affine 2-bit groups and MXFP4 groups share this column alignment.
EXPERT_QUANT_GROUP = 64
_MXFP4_GROUP = 32


def validate_flash_config(config) -> None:
    packed = getattr(config, "qwen_packed_gdn_prefill", False)
    if type(packed) is not bool:
        raise ValueError("qwen_packed_gdn_prefill must be a boolean")
    if packed and getattr(config, "mtp_enabled", False):
        raise ValueError("Disable MTP before enabling Packed GDN prefill")
    sorted_prefill = getattr(config, "qwen_sorted_expert_prefill", False)
    if type(sorted_prefill) is not bool:
        raise ValueError("qwen_sorted_expert_prefill must be a boolean")
    if sorted_prefill and not getattr(config, "qwen_grouped_experts", True):
        raise ValueError("qwen_sorted_expert_prefill requires qwen_grouped_experts")
    for name, minimum, maximum in (("qwen_prefill_read_experts", 1, 32),
                                   ("qwen_prefill_seed_experts", 0, 128)):
        value = getattr(config, name, DEFAULTS[name])
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    if (getattr(config, "qwen_prefill_read_experts", 1) != 1
            or getattr(config, "qwen_prefill_seed_experts", 0)):
        if (not getattr(config, "layer_major_prefill", True)
                or not getattr(config, "batched_expert_prefill", True)
                or getattr(config, "qwen_expert_wave_slots", 0)):
            raise ValueError("Qwen prefill read/seed experiments require whole-layer prefill without waves")
    # Applies only where whole-layer Prefill runs; other Prefill paths ignore it.
    routed = getattr(config, "qwen_prefill_routed_max_tokens", 512)
    if type(routed) is not int or not 0 <= routed <= 262_144:
        raise ValueError("qwen_prefill_routed_max_tokens must be an integer from 0 through 262144")
    for name, maximum in (("qwen_expert_wave_slots", 512),
                          ("qwen_ngram_cache_bytes", 512 * 1024**2)):
        value = getattr(config, name, DEFAULTS[name])
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 0 through {maximum}")
    chunk = getattr(config, "qwen_qsa_query_chunk", 4)
    if type(chunk) is not int or not 1 <= chunk <= 128:
        raise ValueError("qwen_qsa_query_chunk must be an integer from 1 through 128")
    if type(getattr(config, "qwen_qsa_dense_within_budget", False)) is not bool:
        raise ValueError("qwen_qsa_dense_within_budget must be a boolean")
    if type(getattr(config, "qwen_qsa_skip_complete_gather", False)) is not bool:
        raise ValueError("qwen_qsa_skip_complete_gather must be a boolean")
    if type(getattr(config, "qwen_qsa_masked_prefill", False)) is not bool:
        raise ValueError("qwen_qsa_masked_prefill must be a boolean")
    if type(getattr(config, "qwen_qsa_selected_kernel", False)) is not bool:
        raise ValueError("qwen_qsa_selected_kernel must be a boolean")
    common = getattr(config, "qwen_8bit_common_tensors", False)
    if type(common) is not bool:
        raise ValueError("qwen_8bit_common_tensors must be a boolean")
    two_bit = getattr(config, "qwen_2bit_experts", False)
    if type(two_bit) is not bool:
        raise ValueError("qwen_2bit_experts must be a boolean")
    percent = getattr(config, "qwen_2bit_expert_percent", 50)
    if type(percent) is not int or not 0 <= percent <= 100:
        raise ValueError("qwen_2bit_expert_percent must be an integer from 0 through 100")
    if two_bit and percent > 0 and getattr(config, "staged_expert_streaming", False):
        raise ValueError("qwen_2bit_experts does not support staged expert streaming")
    # Qwen ANE Prefill compiles the BF16 q_proj weights it finds at load.
    ane = getattr(config, "ane_prefill", False) and getattr(config, "ane_prefill_ratio", 0) > 0
    if common and ane:
        raise ValueError("qwen_8bit_common_tensors does not support ANE Prefill")
    threshold = getattr(config, "qwen_qsa_dense_threshold", 0)
    if type(threshold) is not int or not 0 <= threshold <= 262_144:
        raise ValueError("qwen_qsa_dense_threshold must be an integer from 0 through 262144")
    backend = getattr(config, "qwen_ngram_io", "mmap")
    if not isinstance(backend, str) or backend not in ("mmap", "pread"):
        raise ValueError("qwen_ngram_io must be mmap or pread")
    if getattr(config, "qwen_ngram_cache_bytes", 0) and backend != "pread":
        raise ValueError("qwen_ngram_cache_bytes requires qwen_ngram_io=pread")
    if (getattr(config, "qwen_expert_wave_slots", 0)
            and getattr(config, "qwen_next_layer_prefetch", False)):
        raise ValueError("qwen_expert_wave_slots cannot use whole-layer next-layer prefetch")


def add_flash_arguments(parser: argparse.ArgumentParser) -> None:
    _add_qsa_arguments(parser)
    parser.add_argument("--qwen-packed-gdn-prefill", action=argparse.BooleanOptionalAction,
                        default=False, help="experimental M5 Swift text prefill; requires MTP off; speed gains not established")
    parser.add_argument("--qwen-sorted-expert-prefill", action=argparse.BooleanOptionalAction,
                        default=False, help="experimental sorted gather_qmm for grouped Qwen prefill; "
                        "output is not bit-identical")
    parser.add_argument("--qwen-prefill-read-experts", type=int, default=1,
                        help="1..32 consecutive experts per prefill read; 1 keeps old I/O")
    parser.add_argument("--qwen-prefill-seed-experts", type=int, default=0,
                        help="0..128 hot prompt-tail experts per layer retained in existing decode slots")
    parser.add_argument("--qwen-prefill-routed-max-tokens", type=int, default=512,
                        help="whole-layer prefill reads only routed experts up to this many prompt "
                        "tokens; 0 reads every expert; same output")
    parser.add_argument("--qwen-expert-wave-slots", type=int, default=0,
                        help="experimental exact expert waves; 1..512 experts per wave, 0 disables")
    parser.add_argument("--qwen-ngram-io", choices=("mmap", "pread"), default="mmap",
                        help="experimental positioned N-gram reads instead of mmap")
    parser.add_argument("--qwen-ngram-cache-bytes", type=int, default=0,
                        help="packed N-gram row LRU payload limit; requires pread; max 512 MiB")
    parser.add_argument("--qwen-8bit-common-tensors", action=argparse.BooleanOptionalAction,
                        default=False, help="store Qwen projections, lm_head and MTP included, as "
                        "affine 8-bit; output is not bit-identical")
    parser.add_argument("--qwen-2bit-experts", action=argparse.BooleanOptionalAction,
                        default=False, help="store part of each routed Qwen expert in memory as affine 2-bit; "
                        "installed expert files stay MXFP4; output is not bit-identical")
    parser.add_argument("--qwen-2bit-expert-percent", type=int, default=50,
                        help="0..100 percent of each expert matrix stored in 2 bits when "
                        "--qwen-2bit-experts is set; 50 is the default; 0 keeps MXFP4")


def _add_qsa_arguments(parser):
    parser.add_argument("--qwen-qsa-query-chunk", type=int, default=16,
                        help="QSA queries per chunk, 1..128; bounded estimated workspace")
    parser.add_argument("--qwen-qsa-dense-within-budget", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="QSA dense SDPA when the context fits the indexer budget; selection is unchanged")
    parser.add_argument("--qwen-qsa-skip-complete-gather", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="skip the identity per-query gather when QSA selects every visible key")
    parser.add_argument("--qwen-qsa-dense-threshold", type=int, default=0,
                        help="0 keeps the indexer budget; otherwise dense SDPA up to this key length")
    parser.add_argument("--qwen-qsa-masked-prefill", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="experimental fused SDPA with the unchanged QSA selection as a mask for "
                        "prompt chunks of 16+ tokens; output is not bit-identical")
    parser.add_argument("--qwen-qsa-selected-kernel", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="experimental Metal kernel that reads only the selected QSA key blocks for "
                        "prompt chunks of 16+ tokens (from 16,384 keys when masked prefill is on); "
                        "FP32 softmax, output is not bit-identical")


def flash_arguments(arguments: argparse.Namespace) -> dict:
    return {name: getattr(arguments, name) for name in DEFAULTS}


def resident_two_bit(config) -> tuple[bool, int]:
    """Whether resident experts are compacted, and the share that identifies them.

    A disabled setting or a 0% share keeps MXFP4, including its prompt cache.
    """
    enabled = getattr(config, "qwen_2bit_experts", False) is True
    percent = getattr(config, "qwen_2bit_expert_percent", DEFAULTS["qwen_2bit_expert_percent"])
    if type(percent) is not int or not 0 <= percent <= 100 or not enabled or percent <= 0:
        return False, 0
    return True, percent


def expert_uses_two_bit(expert_id: int, expert_count: int, percent: int) -> bool:
    """Whether this routed expert is one of the experts stored in 2 bits.

    The share is spread across expert ids. 1% of 512 experts is 5 experts.
    The others stay in the original format. 0% selects none and 100% selects
    every expert.
    """
    if type(percent) is not int or not 0 <= percent <= 100:
        raise ValueError("qwen_2bit_expert_percent must be an integer from 0 through 100")
    if type(expert_id) is not int or type(expert_count) is not int or expert_count <= 0:
        raise ValueError("expert id and expert count must be integers")
    if percent <= 0 or not 0 <= expert_id < expert_count:
        return False
    if percent >= 100:
        return True
    quota = expert_count * percent // 100
    if quota <= 0:
        return False
    return (expert_id * quota) % expert_count < quota


def quantized_expert_columns(column_count: int, percent: int) -> int:
    """Input columns stored in 2 bits.

    Columns are whole groups of 64, rounded to the nearest group. A share above
    0% keeps at least one group. 0% keeps the original matrix. 100% quantizes
    every column. The counted columns start at the first input column.
    """
    if type(percent) is not int or not 0 <= percent <= 100:
        raise ValueError("qwen_2bit_expert_percent must be an integer from 0 through 100")
    if type(column_count) is not int or column_count < 0 or column_count % EXPERT_QUANT_GROUP:
        raise ValueError(
            f"2-bit expert weights need an input size multiple of {EXPERT_QUANT_GROUP}")
    if percent <= 0 or column_count == 0:
        return 0
    groups = column_count // EXPERT_QUANT_GROUP
    if percent >= 100:
        return column_count
    chosen = (groups * percent + 50) // 100
    return min(groups, max(1, chosen)) * EXPERT_QUANT_GROUP


def mxfp4_matrix_bytes(rows: int, columns: int) -> int:
    """Packed MXFP4 weights plus one scale byte per group of 32."""
    return rows * columns // 2 + rows * columns // _MXFP4_GROUP


def two_bit_matrix_bytes(rows: int, columns: int, quantized_columns: int) -> int:
    """2-bit prefix plus the MXFP4 columns that stay at the original precision."""
    if not 0 <= quantized_columns <= columns or quantized_columns % EXPERT_QUANT_GROUP:
        raise ValueError("quantized columns must be a group-aligned prefix of the matrix")
    rest = columns - quantized_columns
    quantized = 0 if quantized_columns == 0 else (
        rows * quantized_columns // 4 + rows * (quantized_columns // EXPERT_QUANT_GROUP) * 4)
    kept = 0 if rest == 0 else mxfp4_matrix_bytes(rows, rest)
    return quantized + kept
