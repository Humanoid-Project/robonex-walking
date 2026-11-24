import ast
import types
import unittest
from pathlib import Path


def load_helper():
    path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/deploy_effects.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "apply_imu_delay_pin")
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["apply_imu_delay_pin"]


def make_cfg(imu):
    return types.SimpleNamespace(scene=types.SimpleNamespace(imu=imu), sim=types.SimpleNamespace(dt=0.0025))


class ImuDelayPinTest(unittest.TestCase):
    def setUp(self):
        self.apply = load_helper()

    def test_training_range_is_recorded_without_a_pin(self):
        cfg = make_cfg(types.SimpleNamespace(min_delay_steps=1, max_delay_steps=6))
        out = self.apply(cfg)
        self.assertEqual((out["source"], out["physics_steps"]), ("training", [1, 6]))
        self.assertAlmostEqual(out["age_ms"][0], 2.5)
        self.assertAlmostEqual(out["age_ms"][1], 15.0)

    def test_pin_fixes_both_ends(self):
        for steps, age in ((0, 0.0), (1, 2.5), (2, 5.0)):
            cfg = make_cfg(types.SimpleNamespace(min_delay_steps=1, max_delay_steps=6))
            out = self.apply(cfg, steps)
            self.assertEqual((cfg.scene.imu.min_delay_steps, cfg.scene.imu.max_delay_steps), (steps, steps))
            self.assertEqual((out["source"], out["physics_steps"]), ("pin", [steps, steps]))
            self.assertAlmostEqual(out["age_ms"][0], age)

    def test_pin_is_validated(self):
        with self.assertRaises(ValueError):
            self.apply(make_cfg(types.SimpleNamespace(min_delay_steps=1, max_delay_steps=6)), -1)
        with self.assertRaises(KeyError):
            self.apply(make_cfg(types.SimpleNamespace()), 0)
        with self.assertRaises(KeyError):
            self.apply(make_cfg(None), 0)

    def test_eval_and_trace_wire_the_pin(self):
        for name in ("eval_commands.py", "sim_joint_trace.py"):
            text = (Path(__file__).resolve().parents[1] / "scripts/rsl_rl" / name).read_text()
            self.assertIn('"--imu_delay_pin"', text)
            self.assertIn("apply_imu_delay_pin(env_cfg, args_cli.imu_delay_pin)", text)
            self.assertIn('"imu_delay": imu_delay', text)
            self.assertLess(text.index("apply_imu_delay_pin(env_cfg"), text.index("gym.make(args_cli.task"))


if __name__ == "__main__":
    unittest.main()
