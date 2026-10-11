"""Reject accidental diagnostic distribution builds before invoking build tools."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class PackageDiagnosticsTests(unittest.TestCase):
    def run_rejected(self, flag, flavor=None, name='WHALLM_QWEN_PREFILL_SYNC'):
        shell = shutil.which('zsh')
        if shell is None:
            self.skipTest('zsh is required for the macOS packaging script')
        script = Path(__file__).resolve().parents[2] / 'Scripts/package-app.sh'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            # No xcrun/swift in PATH: a missing guard must not start a real build.
            env = {'PATH': directory, 'OUTPUT_ROOT': str(output), name: flag}
            if flavor is not None:
                env['WHALLM_BUILD_FLAVOR'] = flavor
            result = subprocess.run([shell, str(script)], env=env,
                capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            return result.stderr

    def test_distribution_rejects_sync_diagnostics(self):
        for flavor in (None, 'distribution'):
            with self.subTest(flavor=flavor):
                self.assertIn('only be packaged in a local build', self.run_rejected('1', flavor))

    def test_invalid_diagnostic_flag_is_rejected(self):
        for name in ('WHALLM_QWEN_PREFILL_SYNC', 'WHALLM_QWEN_QSA_MASKED_PREFILL_OFF'):
            for flag in ('yes', '2', '-1'):
                with self.subTest(name=name, flag=flag):
                    self.assertIn(f'{name} must be 0 or 1', self.run_rejected(flag, 'local', name))

    def test_distribution_rejects_masked_prefill_off(self):
        for flavor in (None, 'distribution'):
            with self.subTest(flavor=flavor):
                self.assertIn('only be packaged in a local build', self.run_rejected(
                    '1', flavor, 'WHALLM_QWEN_QSA_MASKED_PREFILL_OFF'))


if __name__ == '__main__':
    unittest.main()
