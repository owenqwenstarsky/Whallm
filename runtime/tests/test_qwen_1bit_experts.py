"""Opt-in 1-bit copies of Qwen routed experts. Synthetic weights; no checkpoint."""
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
from deepseek_v4_ssd.qwen_expert_1bit import (
    apply_one_bit_expert,
    pack_affine_1bit,
    quantize_qwen_expert,
    unpack_affine_1bit,
)
from deepseek_v4_ssd.qwen_flash_config import DEFAULTS, add_flash_arguments, flash_arguments


class OneBitExpertTests(unittest.TestCase):
    def test_midpoint_bits_and_constant_groups_round_trip(self):
        rows, cols = 3, 128
        weight = mx.zeros((rows, cols), dtype=mx.float32)
        weight[0, ::2] = 1
        weight[0, 1::2] = 5
        weight[1] = -2
        weight[2, :64] = mx.arange(64)
        weight[2, 64:] = 7
        source = weight.astype(mx.bfloat16)
        packed, scales, biases = pack_affine_1bit(source)
        self.assertEqual(packed.shape, (rows, cols // 8))
        self.assertEqual(scales.shape, (rows, cols // 64))
        # Odd positions in the first byte are the 2, 8, 32 and 128 places.
        # A negative constant stays zero.
        self.assertEqual(int(packed[0, 0].item()), 2 + 8 + 32 + 128)
        self.assertEqual(int(packed[1, 0].item()), 0)
        restored = unpack_affine_1bit(packed, scales, biases, mx.bfloat16)
        self.assertTrue(mx.array_equal(restored[0], source[0]).item())
        self.assertTrue(mx.array_equal(restored[1], source[1]).item())
        ends = mx.stack([biases[2, 0], biases[2, 0] + scales[2, 0]]).astype(mx.bfloat16)
        group = restored[2, :64]
        self.assertTrue(mx.all((group == ends[0]) | (group == ends[1])).item())
        self.assertTrue(mx.array_equal(restored[2, 64:], source[2, 64:]).item())

    def test_resident_mxfp4_becomes_one_bit_and_the_product_uses_that_copy(self):
        mx.random.seed(7)
        gate = mx.random.normal((16, 128)).astype(mx.bfloat16)
        down = mx.random.normal((8, 64)).astype(mx.bfloat16)
        gate_weight, gate_scales, *_ = mx.quantize(gate, group_size=32, bits=4, mode="mxfp4")
        down_weight, down_scales, *_ = mx.quantize(down, group_size=32, bits=4, mode="mxfp4")
        weights = QwenExpertWeights(gate_weight, gate_scales, down_weight, down_scales)
        packed = quantize_qwen_expert(weights)
        mxfp4_bytes = (gate_weight.nbytes + gate_scales.nbytes
                       + down_weight.nbytes + down_scales.nbytes)
        one_bit_bytes = sum(array.nbytes for array in (
            packed.gate_up, packed.gate_up_scales, packed.gate_up_biases,
            packed.down, packed.down_scales, packed.down_biases))
        self.assertEqual(one_bit_bytes * 17, mxfp4_bytes * 6)
        dense_gate = mx.dequantize(gate_weight, gate_scales, group_size=32, bits=4, mode="mxfp4")
        expected_gate, expected_scales, expected_biases = pack_affine_1bit(dense_gate)
        self.assertTrue(mx.array_equal(packed.gate_up, expected_gate).item())
        self.assertTrue(mx.array_equal(packed.gate_up_scales, expected_scales).item())
        self.assertTrue(mx.array_equal(packed.gate_up_biases, expected_biases).item())
        value = mx.random.normal((3, 128)).astype(mx.bfloat16)
        got = apply_one_bit_expert(value, packed)
        restored = unpack_affine_1bit(packed.gate_up, packed.gate_up_scales,
                                      packed.gate_up_biases, value.dtype)
        projected = value @ restored.T
        left, right = mx.split(projected, 2, axis=-1)
        hidden = nn.silu(left) * right
        down = unpack_affine_1bit(packed.down, packed.down_scales, packed.down_biases, value.dtype)
        self.assertTrue(mx.allclose(got, hidden @ down.T, rtol=0, atol=0).item())

    def test_setting_is_off_by_default_and_does_not_touch_other_weights(self):
        self.assertFalse(RuntimeConfig().qwen_1bit_experts)
        config = RuntimeConfig(qwen_1bit_experts=True, qwen_8bit_common_tensors=True,
                               ane_prefill=True, ane_prefill_ratio=0.5)
        validate_runtime_config(config)
        for kind in ("qwen3.8-flash-next", "swift1.5-qwen3.8-flash-next"):
            get_support(kind).validate_config(config)
        for kind in ("deepseek-v4", "deepseek-v4.1"):
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(config)
        with self.assertRaises(ValueError):
            validate_runtime_config(replace(config, qwen_1bit_experts=1))
        with self.assertRaises(ValueError):
            validate_runtime_config(replace(config, staged_expert_streaming=True))
        parser = argparse.ArgumentParser()
        add_flash_arguments(parser)
        self.assertIs(flash_arguments(parser.parse_args([]))["qwen_1bit_experts"], False)
        self.assertIs(flash_arguments(parser.parse_args(["--qwen-1bit-experts"]))
                      ["qwen_1bit_experts"], True)

    def test_prompt_caches_separate_expert_precision_from_common_tensors(self):
        installed = SimpleNamespace(root="/missing", revision="r", model_id="m")
        off = _prompt_cache_contract(installed, RuntimeConfig())
        eight = _prompt_cache_contract(installed, RuntimeConfig(qwen_8bit_common_tensors=True))
        one = _prompt_cache_contract(installed, RuntimeConfig(qwen_1bit_experts=True))
        both = _prompt_cache_contract(installed, RuntimeConfig(
            qwen_8bit_common_tensors=True, qwen_1bit_experts=True))
        self.assertNotIn("qwenExpertWeights", off)
        self.assertNotIn("qwenCommonTensors", off)
        self.assertEqual(eight["qwenCommonTensors"], "affine8-group64-v1")
        self.assertNotIn("qwenExpertWeights", eight)
        self.assertEqual(one["qwenExpertWeights"], "affine1-group64-v1")
        self.assertNotIn("qwenCommonTensors", one)
        self.assertEqual(both["qwenCommonTensors"], "affine8-group64-v1")
        self.assertEqual(both["qwenExpertWeights"], "affine1-group64-v1")

    def test_removed_common_tensor_flag_is_not_one_bit_experts(self):
        raw = asdict(RuntimeConfig())
        for name in DEFAULTS:
            raw.pop(name)
        absent = dict(raw)
        absent["qwen_1bit_common_tensors"] = False
        self.assertFalse(_parse_runtime(absent, "runtime", "qwen3.8-flash-next").qwen_1bit_experts)
        raw["qwen_1bit_common_tensors"] = True
        with self.assertRaises(ModelCatalogError):
            _parse_runtime(raw, "runtime", "qwen3.8-flash-next")


if __name__ == "__main__":
    unittest.main()
