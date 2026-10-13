"""Opt-in 2-bit copies of part of each Qwen routed expert. Synthetic weights; no checkpoint."""
from __future__ import annotations

import argparse
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from deepseek_v4_ssd.expert_cache import QwenExpertWeights
from deepseek_v4_ssd.generation import _prompt_cache_contract
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelCatalogError, _parse_runtime, validate_runtime_config
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.qwen_expert_2bit import (
    TwoBitQwenExpert,
    apply_two_bit_expert,
    pack_affine_2bit,
    quantize_qwen_expert,
    unpack_affine_2bit,
)
from deepseek_v4_ssd.qwen_flash_config import (
    DEFAULTS,
    add_flash_arguments,
    flash_arguments,
    mxfp4_matrix_bytes,
    quantized_expert_columns,
    two_bit_matrix_bytes,
)


class TwoBitExpertTests(unittest.TestCase):
    def test_four_levels_and_constant_groups_round_trip(self):
        rows, cols = 3, 128
        weight = mx.zeros((rows, cols), dtype=mx.float32)
        weight[0, :4] = mx.array([0, 1, 2, 3])
        weight[1] = -2
        weight[2, :64] = mx.arange(64)
        weight[2, 64:] = 7
        source = weight.astype(mx.bfloat16)
        packed, scales, biases = pack_affine_2bit(source)
        self.assertEqual(packed.shape, (rows, cols // 4))
        self.assertEqual(scales.shape, (rows, cols // 64))
        # Codes 0, 1, 2, 3 occupy the 1, 4, 16 and 64 places.
        self.assertEqual(int(packed[0, 0].item()), 1 * 4 + 2 * 16 + 3 * 64)
        self.assertEqual(int(packed[1, 0].item()), 0)
        self.assertEqual(float(scales[1, 0].item()), 0)
        restored = unpack_affine_2bit(packed, scales, biases, mx.bfloat16)
        self.assertTrue(mx.array_equal(restored[0, :4], source[0, :4]).item())
        self.assertTrue(mx.array_equal(restored[1], source[1]).item())
        ends = mx.stack([biases[2, 0], biases[2, 0] + scales[2, 0]]).astype(mx.bfloat16)
        group = restored[2, :64]
        self.assertTrue(mx.array_equal(group[0], ends[0]).item())
        self.assertTrue(mx.array_equal(group[-1], ends[1]).item())
        self.assertTrue(mx.array_equal(restored[2, 64:], source[2, 64:]).item())

    def test_share_keeps_an_mxfp4_tail_and_the_product_uses_both(self):
        mx.random.seed(7)
        # gate_up is [2 * intermediate, hidden]; down is [hidden, intermediate].
        gate = mx.random.normal((256, 128)).astype(mx.bfloat16)
        down = mx.random.normal((32, 128)).astype(mx.bfloat16)
        gate_weight, gate_scales, *_ = mx.quantize(gate, group_size=32, bits=4, mode="mxfp4")
        down_weight, down_scales, *_ = mx.quantize(down, group_size=32, bits=4, mode="mxfp4")
        weights = QwenExpertWeights(gate_weight, gate_scales, down_weight, down_scales)
        self.assertIs(quantize_qwen_expert(weights, 0), weights)
        packed = quantize_qwen_expert(weights, 50)
        self.assertIsInstance(packed, TwoBitQwenExpert)
        self.assertEqual(packed.gate_up_columns, 64)
        self.assertEqual(packed.down_columns, 64)
        dense_gate = mx.dequantize(gate_weight, gate_scales, group_size=32, bits=4, mode="mxfp4")
        dense_down = mx.dequantize(down_weight, down_scales, group_size=32, bits=4, mode="mxfp4")
        prefix, prefix_scales, prefix_biases = pack_affine_2bit(dense_gate[:, :64])
        self.assertTrue(mx.array_equal(packed.gate_up, prefix).item())
        self.assertTrue(mx.array_equal(packed.gate_up_scales, prefix_scales).item())
        self.assertTrue(mx.array_equal(packed.gate_up_biases, prefix_biases).item())
        tail = mx.dequantize(packed.gate_up_mx, packed.gate_up_mx_scales,
                             group_size=32, bits=4, mode="mxfp4")
        self.assertTrue(mx.allclose(tail, dense_gate[:, 64:]).item())
        expected_tail = dense_gate[:, 64:].astype(mx.float32)
        mx.eval(expected_tail)
        stored = sum(array.nbytes for array in (
            packed.gate_up, packed.gate_up_scales, packed.gate_up_biases, packed.gate_up_mx,
            packed.gate_up_mx_scales, packed.down, packed.down_scales, packed.down_biases,
            packed.down_mx, packed.down_mx_scales))
        expected = (two_bit_matrix_bytes(256, 128, 64) + two_bit_matrix_bytes(32, 128, 64))
        self.assertEqual(stored, expected)
        # The tail must not be a view of the slot that quantization releases.
        gate_weight[:] = mx.zeros_like(gate_weight)
        tail_after = mx.dequantize(packed.gate_up_mx, packed.gate_up_mx_scales,
                                   group_size=32, bits=4, mode="mxfp4")
        self.assertTrue(mx.allclose(tail_after.astype(mx.float32), expected_tail).item())
        value = mx.random.normal((3, 128)).astype(mx.bfloat16)
        got = apply_two_bit_expert(value, packed)
        gate_prefix = unpack_affine_2bit(
            packed.gate_up, packed.gate_up_scales, packed.gate_up_biases, value.dtype)
        projected = (value[:, :64] @ gate_prefix.T
                     + value[:, 64:] @ dense_gate[:, 64:].astype(value.dtype).T)
        left, right = mx.split(projected, 2, axis=-1)
        hidden = nn.silu(left) * right
        down_prefix = unpack_affine_2bit(
            packed.down, packed.down_scales, packed.down_biases, hidden.dtype)
        expected = (hidden[:, :64] @ down_prefix.T
                    + hidden[:, 64:] @ dense_down[:, 64:].astype(hidden.dtype).T)
        self.assertTrue(mx.allclose(got, expected, rtol=0, atol=0).item())

    def test_full_share_is_ten_seventeenths_of_mxfp4(self):
        gate = mx.arange(256 * 128).reshape(256, 128).astype(mx.bfloat16)
        down = mx.arange(32 * 128).reshape(32, 128).astype(mx.bfloat16)
        gate_weight, gate_scales, *_ = mx.quantize(gate, group_size=32, bits=4, mode="mxfp4")
        down_weight, down_scales, *_ = mx.quantize(down, group_size=32, bits=4, mode="mxfp4")
        weights = QwenExpertWeights(gate_weight, gate_scales, down_weight, down_scales)
        packed = quantize_qwen_expert(weights, 100)
        self.assertIsNone(packed.gate_up_mx)
        self.assertIsNone(packed.down_mx)
        self.assertEqual(packed.gate_up_columns, 128)
        self.assertEqual(packed.down_columns, 128)
        mxfp4_bytes = (mxfp4_matrix_bytes(256, 128) + mxfp4_matrix_bytes(32, 128))
        stored = sum(array.nbytes for array in (
            packed.gate_up, packed.gate_up_scales, packed.gate_up_biases,
            packed.down, packed.down_scales, packed.down_biases))
        self.assertEqual(stored * 17, mxfp4_bytes * 10)
        value = mx.ones((2, 128), dtype=mx.bfloat16)
        got = apply_two_bit_expert(value, packed)
        self.assertEqual(tuple(got.shape), (2, 32))

    def test_setting_is_off_by_default_and_defaults_the_share_to_half(self):
        self.assertFalse(RuntimeConfig().qwen_2bit_experts)
        self.assertEqual(RuntimeConfig().qwen_2bit_expert_percent, 50)
        config = RuntimeConfig(qwen_2bit_experts=True, qwen_2bit_expert_percent=50,
                               qwen_8bit_common_tensors=True, ane_prefill=True, ane_prefill_ratio=0.5)
        validate_runtime_config(config)
        for kind in ("qwen3.8-flash-next", "swift1.5-qwen3.8-flash-next"):
            get_support(kind).validate_config(config)
        for kind in ("deepseek-v4", "deepseek-v4.1"):
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(config)
        with self.assertRaises(ValueError):
            validate_runtime_config(replace(config, qwen_2bit_experts=1))
        with self.assertRaises(ValueError):
            validate_runtime_config(replace(config, qwen_2bit_expert_percent=101))
        with self.assertRaises(ValueError):
            validate_runtime_config(replace(config, qwen_2bit_expert_percent=True))
        with self.assertRaises(ValueError):
            validate_runtime_config(replace(config, staged_expert_streaming=True))
        validate_runtime_config(replace(config, qwen_2bit_experts=True, qwen_2bit_expert_percent=0,
                                        staged_expert_streaming=True))
        parser = argparse.ArgumentParser()
        add_flash_arguments(parser)
        self.assertIs(flash_arguments(parser.parse_args([]))["qwen_2bit_experts"], False)
        self.assertEqual(flash_arguments(parser.parse_args([]))["qwen_2bit_expert_percent"], 50)
        enabled = flash_arguments(parser.parse_args(
            ["--qwen-2bit-experts", "--qwen-2bit-expert-percent", "25"]))
        self.assertIs(enabled["qwen_2bit_experts"], True)
        self.assertEqual(enabled["qwen_2bit_expert_percent"], 25)

    def test_prompt_caches_separate_each_share_from_common_tensors(self):
        installed = SimpleNamespace(root="/missing", revision="r", model_id="m")
        off = _prompt_cache_contract(installed, RuntimeConfig())
        eight = _prompt_cache_contract(installed, RuntimeConfig(qwen_8bit_common_tensors=True))
        half = _prompt_cache_contract(installed, RuntimeConfig(qwen_2bit_experts=True))
        full = _prompt_cache_contract(installed, RuntimeConfig(
            qwen_2bit_experts=True, qwen_2bit_expert_percent=100))
        zero = _prompt_cache_contract(installed, RuntimeConfig(
            qwen_2bit_experts=True, qwen_2bit_expert_percent=0))
        both = _prompt_cache_contract(installed, RuntimeConfig(
            qwen_8bit_common_tensors=True, qwen_2bit_experts=True, qwen_2bit_expert_percent=50))
        self.assertNotIn("qwenExpertWeights", off)
        self.assertNotIn("qwenExpertWeights", eight)
        self.assertNotIn("qwenExpertWeights", zero)
        self.assertEqual(half["qwenExpertWeights"], "affine2-group64-p50-v1")
        self.assertEqual(full["qwenExpertWeights"], "affine2-group64-p100-v1")
        self.assertEqual(both["qwenCommonTensors"], "affine8-group64-v1")
        self.assertEqual(both["qwenExpertWeights"], "affine2-group64-p50-v1")
        self.assertEqual(quantized_expert_columns(2560, 1), 64)
        self.assertEqual(quantized_expert_columns(2560, 50), 1280)

    def test_removed_one_bit_flag_is_not_the_two_bit_share(self):
        raw = asdict(RuntimeConfig())
        for name in DEFAULTS:
            raw.pop(name)
        absent = dict(raw)
        absent["qwen_1bit_common_tensors"] = False
        absent["qwen_1bit_experts"] = False
        parsed = _parse_runtime(absent, "runtime", "qwen3.8-flash-next")
        self.assertFalse(parsed.qwen_2bit_experts)
        self.assertEqual(parsed.qwen_2bit_expert_percent, 50)
        raw["qwen_1bit_experts"] = True
        with self.assertRaises(ModelCatalogError):
            _parse_runtime(raw, "runtime", "qwen3.8-flash-next")
        raw["qwen_1bit_experts"] = False
        raw["qwen_1bit_common_tensors"] = True
        with self.assertRaises(ModelCatalogError):
            _parse_runtime(raw, "runtime", "qwen3.8-flash-next")


if __name__ == "__main__":
    unittest.main()
