"""Opt-in 1-bit common tensors for the Qwen engine. Synthetic modules and a tiny
model; no checkpoint required."""
from __future__ import annotations

import argparse
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from deepseek_v4_ssd.generation import _prompt_cache_contract
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import validate_runtime_config
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.qwen_common_tensors import (
    ONE_BIT_OUTPUT_CHUNK,
    OneBitLinear,
    pack_affine_1bit,
    quantize_1bit_common_tensors,
    unpack_affine_1bit,
)
from deepseek_v4_ssd.qwen_flash_config import add_flash_arguments, flash_arguments


class Node(nn.Module):
    def __init__(self, **children):
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)


def linear(inputs, outputs, *, bias=False):
    layer = nn.Linear(inputs, outputs, bias=bias)
    layer.weight = (mx.random.normal((outputs, inputs)) * 0.05).astype(mx.bfloat16)
    if bias:
        layer.bias = mx.zeros((outputs,), dtype=mx.bfloat16)
    return layer


class OneBitTensorTests(unittest.TestCase):
    def test_two_level_and_constant_groups_round_trip(self):
        rows, cols = 3, 128
        weight = mx.zeros((rows, cols), dtype=mx.float32)
        weight[0, ::2] = 1
        weight[0, 1::2] = 5
        weight[1] = -2
        weight[2, :64] = mx.arange(64)
        weight[2, 64:] = 7
        source = weight.astype(mx.bfloat16)
        packed, scales, biases = pack_affine_1bit(source)
        self.assertEqual(packed.dtype, mx.uint8)
        self.assertEqual(packed.shape, (rows, cols // 8))
        self.assertEqual(scales.shape, (rows, cols // 64))
        self.assertEqual((scales.dtype, biases.dtype), (mx.bfloat16, mx.bfloat16))
        restored = unpack_affine_1bit(packed, scales, biases, mx.bfloat16)
        self.assertTrue(mx.array_equal(restored[0], source[0]).item())
        self.assertTrue(mx.array_equal(restored[1], source[1]).item())
        # The third row's first group is not two-valued, so it uses only the ends.
        ends = mx.stack([biases[2, 0], biases[2, 0] + scales[2, 0]]).astype(mx.bfloat16)
        group = restored[2, :64]
        self.assertTrue(mx.all((group == ends[0]) | (group == ends[1])).item())
        self.assertTrue(mx.array_equal(restored[2, 64:], source[2, 64:]).item())

    def test_products_match_the_restored_matrix_including_bias_and_chunks(self):
        mx.random.seed(7)
        source = linear(128, ONE_BIT_OUTPUT_CHUNK + 1, bias=True)
        packed = OneBitLinear.from_linear(source)
        self.assertEqual((packed.bits, packed.group_size, packed.mode), (1, 64, "affine"))
        weight = unpack_affine_1bit(packed.weight, packed.scales, packed.biases, mx.bfloat16)
        x = mx.random.normal((1, 4, 128)).astype(mx.bfloat16)
        with patch("deepseek_v4_ssd.qwen_common_tensors.mx.eval", wraps=mx.eval) as evaluated:
            got = packed(x)
        self.assertEqual(evaluated.call_count, 2)
        expected = mx.addmm(source.bias, x, weight.T)
        self.assertTrue(mx.allclose(got, expected, rtol=0, atol=0).item())
        narrow = OneBitLinear.from_linear(linear(64, ONE_BIT_OUTPUT_CHUNK))
        with patch("deepseek_v4_ssd.qwen_common_tensors.mx.eval", wraps=mx.eval) as evaluated:
            narrow(mx.zeros((1, 1, 64), dtype=mx.bfloat16))
        self.assertEqual(evaluated.call_count, 0)

    def test_large_projections_become_1_bit_except_routers(self):
        mx.random.seed(3)
        model = Node(
            model=Node(embed_tokens=nn.Embedding(512, 64), layers=[Node(
                linear_attn=Node(in_proj_qkv=linear(64, 256), in_proj_a=linear(64, 48)),
                mlp=Node(gate=linear(64, 512), shared_expert_gate=linear(64, 1)))]),
            lm_head=linear(64, 512), mtp=Node(fc_hidden=linear(128, 256)))
        before = dict(model.named_modules())
        packed_paths = ["lm_head", "model.layers.0.linear_attn.in_proj_qkv", "mtp.fc_hidden"]
        odd = Node(proj=linear(48, 256))
        with self.assertRaises(ValueError) as error:
            quantize_1bit_common_tensors(odd)
        self.assertIn("proj", str(error.exception))

        self.assertEqual(quantize_1bit_common_tensors(model), 3)
        after = dict(model.named_modules())
        self.assertEqual(sorted(path for path, module in after.items()
                                if isinstance(module, OneBitLinear)), packed_paths)
        for path in ("model.embed_tokens", "model.layers.0.linear_attn.in_proj_a",
                     "model.layers.0.mlp.gate", "model.layers.0.mlp.shared_expert_gate"):
            self.assertIs(after[path], before[path])
        for path in packed_paths:
            packed, source = after[path], before[path]
            x = mx.random.normal((1, 2, source.weight.shape[1])).astype(mx.bfloat16)
            weight = unpack_affine_1bit(packed.weight, packed.scales, packed.biases, x.dtype)
            self.assertTrue(mx.array_equal(packed(x), x @ weight.T).item(), path)

    def test_setting_is_off_by_default_exclusive_and_qwen_only(self):
        self.assertFalse(RuntimeConfig().qwen_1bit_common_tensors)
        config = RuntimeConfig(qwen_1bit_common_tensors=True)
        validate_runtime_config(config)
        for kind in ("qwen3.8-flash-next", "swift1.5-qwen3.8-flash-next"):
            get_support(kind).validate_config(config)
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(replace(config, ane_prefill_ratio=0.5))
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(replace(config, qwen_8bit_common_tensors=True))
        for kind in ("deepseek-v4", "deepseek-v4.1"):
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(config)
        with self.assertRaises(ValueError):
            validate_runtime_config(replace(config, qwen_1bit_common_tensors=1))
        parser = argparse.ArgumentParser()
        add_flash_arguments(parser)
        self.assertIs(flash_arguments(parser.parse_args([]))["qwen_1bit_common_tensors"], False)
        self.assertIs(flash_arguments(parser.parse_args(["--qwen-1bit-common-tensors"]))
                      ["qwen_1bit_common_tensors"], True)

    def test_prompt_caches_for_bf16_8_bit_and_1_bit_stay_apart(self):
        installed = SimpleNamespace(root="/missing", revision="r", model_id="m")
        off = _prompt_cache_contract(installed, RuntimeConfig())
        eight = _prompt_cache_contract(installed, RuntimeConfig(qwen_8bit_common_tensors=True))
        one = _prompt_cache_contract(installed, RuntimeConfig(qwen_1bit_common_tensors=True))
        self.assertNotIn("qwenCommonTensors", off)
        self.assertEqual(eight["qwenCommonTensors"], "affine8-group64-v1")
        self.assertEqual(one["qwenCommonTensors"], "affine1-group64-v1")
        eight.pop("qwenCommonTensors")
        one.pop("qwenCommonTensors")
        self.assertEqual(eight, off)
        self.assertEqual(one, off)

    def test_loading_packs_1_bit_instead_of_8_bit_when_enabled(self):
        support = get_support("swift1.5-qwen3.8-flash-next")
        model, experts = SimpleNamespace(args=object()), object()
        mtp = object()
        calls = []
        config = RuntimeConfig(mtp_enabled=True, qwen_1bit_common_tensors=True,
                               qwen_8bit_common_tensors=True)
        with patch("deepseek_v4_ssd.qwen4_exp.load", return_value=(model, experts)), \
             patch("deepseek_v4_ssd.model._load_qwen_mtp", return_value=(mtp, object())), \
             patch("deepseek_v4_ssd.qwen_common_tensors.quantize_1bit_common_tensors",
                   side_effect=lambda target: calls.append(("1", target, target.mtp))), \
             patch("deepseek_v4_ssd.qwen_common_tensors.quantize_common_tensors",
                   side_effect=lambda target: calls.append(("8", target))):
            self.assertEqual(support.load(object(), config, {}, {}, object()), (model, experts))
        self.assertEqual(calls, [("1", model, mtp)])


if __name__ == "__main__":
    unittest.main()
