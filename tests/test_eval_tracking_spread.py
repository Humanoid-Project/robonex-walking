import ast
import math
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"


def load():
    node = next(n for n in ast.parse(EVAL.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "tracking_spread")
    ns = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(EVAL), "exec"), ns)
    return ns["tracking_spread"]


def accumulate(samples):
    total = samples.sum(axis=2)
    squares = (samples * samples).sum(axis=2)
    got = samples.mean(axis=1).sum(axis=1) / samples.shape[2]
    return total.tolist(), squares.tolist(), samples.shape[2], got


class TrackingSpreadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spread = staticmethod(load())

    def test_keys(self):
        total, squares, count, _ = accumulate(np.zeros((3, 4, 10)))
        self.assertEqual(set(self.spread(total, squares, count)),
                         {"got_vx_env_sd", "got_vy_env_sd", "got_wz_env_sd", "vx_osc_rms", "wz_osc_rms"})

    def test_constant_bias_per_env(self):
        bias = np.array([0.1, 0.2, 0.3, 0.4])
        samples = np.zeros((3, 4, 50))
        samples[0] = bias[:, None]
        samples[2] = -bias[:, None]
        row = self.spread(*accumulate(samples)[:3])
        self.assertAlmostEqual(row["vx_osc_rms"], 0.0, places=7)
        self.assertAlmostEqual(row["wz_osc_rms"], 0.0, places=7)
        self.assertAlmostEqual(row["got_vx_env_sd"], float(np.std(bias)), places=12)
        self.assertAlmostEqual(row["got_wz_env_sd"], float(np.std(bias)), places=12)
        self.assertEqual(row["got_vy_env_sd"], 0.0)

    def test_pure_sine(self):
        amplitude, steps = 0.3, 400
        t = np.arange(steps)
        wave = amplitude * np.sin(2 * np.pi * t / 40)
        samples = np.zeros((3, 8, steps))
        samples[0] = wave + 0.25
        samples[2] = np.cos(2 * np.pi * t / 40) * amplitude
        row = self.spread(*accumulate(samples)[:3])
        self.assertAlmostEqual(row["vx_osc_rms"], amplitude / math.sqrt(2), places=10)
        self.assertAlmostEqual(row["wz_osc_rms"], amplitude / math.sqrt(2), places=10)
        self.assertAlmostEqual(row["got_vx_env_sd"], 0.0, places=10)
        self.assertAlmostEqual(row["got_wz_env_sd"], 0.0, places=10)

    def test_mean_of_env_means_equals_got(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(0.1, 0.05, size=(3, 16, 120)) + rng.normal(0.0, 0.02, size=(3, 16, 1))
        total, squares, count, got = accumulate(samples)
        env_means = np.array(total) / count
        np.testing.assert_allclose(env_means.mean(axis=1), got, rtol=0, atol=1e-15)
        row = self.spread(total, squares, count)
        self.assertAlmostEqual(row["got_vy_env_sd"], float(np.std(samples[1].mean(axis=1))), places=12)
        self.assertAlmostEqual(row["wz_osc_rms"], float(np.sqrt(samples[2].var(axis=1).mean())), places=12)

    def test_main_accumulates_on_the_got_steps(self):
        main = next(n for n in ast.parse(EVAL.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        source = ast.unparse(main)
        warm = source.index("if step < args_cli.warmup_steps:")
        got = source.index("got[2] += ang[:, 2].mean()")
        add = source.index("env_sum += sample")
        counted = source.index("counted += 1.0")
        self.assertTrue(warm < got < add < counted)
        self.assertIn("row.update(tracking_spread(env_sum.cpu().tolist(), env_sumsq.cpu().tolist(), counted.item()))",
                      source)


if __name__ == "__main__":
    unittest.main()
