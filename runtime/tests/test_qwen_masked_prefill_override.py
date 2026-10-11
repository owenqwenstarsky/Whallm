"""Opt-in diagnostic override that turns masked QSA prefill off at runtime open."""
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepseek_v4_ssd.generation import ModelRuntime, _prompt_cache_contract
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_support import qwen


class MaskedPrefillOverrideTests(unittest.TestCase):
    def open_runtime(self, config, **environment):
        loaded = []
        installed = SimpleNamespace(root=Path('/tmp/tokenizer'), is_qwen=True,
                                    maximum_context=262_144)
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(qwen, '_MASKED_PREFILL_OFF_MARKER', Path(directory) / 'absent'), \
                patch.dict(os.environ, environment), \
                patch('deepseek_v4_ssd.memory_budget.resolve_cache_budgets',
                      side_effect=lambda _installed, value: value), \
                patch('deepseek_v4_ssd.generation.load_model',
                      side_effect=lambda _installed, value: loaded.append(value)
                      or (object(), SimpleNamespace(close=lambda: None))), \
                patch('deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained',
                      return_value=SimpleNamespace(bos_token=None)), \
                redirect_stderr(log):
            for name in ('WHALLM_QWEN_QSA_MASKED_PREFILL_OFF',):
                if name not in environment:
                    os.environ.pop(name, None)
            runtime = ModelRuntime(installed, config)
        self.addCleanup(runtime.close)
        return runtime, loaded[0], log.getvalue()

    def test_override_reaches_model_load_runtime_config_and_cache_contract(self):
        config = RuntimeConfig(qwen_qsa_masked_prefill=True, persistent_prompt_cache=False)
        runtime, loaded, log = self.open_runtime(config, WHALLM_QWEN_QSA_MASKED_PREFILL_OFF='1')
        self.assertFalse(loaded.qwen_qsa_masked_prefill)
        self.assertFalse(runtime.config.qwen_qsa_masked_prefill)
        self.assertTrue(config.qwen_qsa_masked_prefill)  # Caller's settings object is unchanged.
        installed = SimpleNamespace(model_id='m', revision='r', format_version=1, model_kind='qwen')
        self.assertEqual(_prompt_cache_contract(installed, runtime.config),
                         _prompt_cache_contract(installed, RuntimeConfig(persistent_prompt_cache=False)))
        self.assertEqual(log.count('[DEBUG-qwen-qsa-masked-off]'), 1)

    def test_default_and_explicit_zero_keep_the_configured_path(self):
        config = RuntimeConfig(qwen_qsa_masked_prefill=True, persistent_prompt_cache=False)
        for environment in ({}, {'WHALLM_QWEN_QSA_MASKED_PREFILL_OFF': '0'},
                            {'WHALLM_QWEN_QSA_MASKED_PREFILL_OFF': 'yes'}):
            with self.subTest(environment=environment):
                runtime, loaded, log = self.open_runtime(config, **environment)
                self.assertTrue(loaded.qwen_qsa_masked_prefill)
                self.assertTrue(runtime.config.qwen_qsa_masked_prefill)
                self.assertEqual(log, '')

    def test_marker_enables_and_zero_overrides_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'marker'
            with patch.object(qwen, '_MASKED_PREFILL_OFF_MARKER', marker), \
                    patch.dict(os.environ, {}, clear=False):
                os.environ.pop('WHALLM_QWEN_QSA_MASKED_PREFILL_OFF', None)
                self.assertFalse(qwen._masked_prefill_off_enabled())
                marker.touch()
                self.assertTrue(qwen._masked_prefill_off_enabled())
                os.environ['WHALLM_QWEN_QSA_MASKED_PREFILL_OFF'] = '0'
                self.assertFalse(qwen._masked_prefill_off_enabled())

    def test_configs_without_masked_prefill_are_returned_unchanged(self):
        config = RuntimeConfig(qwen_qsa_masked_prefill=False)
        with patch.dict(os.environ, {'WHALLM_QWEN_QSA_MASKED_PREFILL_OFF': '1'}), \
                redirect_stderr(io.StringIO()) as log:
            self.assertIs(qwen.masked_prefill_override(config), config)
        self.assertEqual(log.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
