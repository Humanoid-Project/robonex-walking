import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
CFG = TASK / "robonex_walking_v2_env_cfg.py"
AGENT = TASK / "agents/rsl_rl_ppo_cfg.py"

RECIPE = {
    ("rewards", "slew_lag", "weight"): 0.0,
    ("rewards", "stand_still", "weight"): -2.0,
    ("rewards", "com_forward_margin", "weight"): -0.5,
    ("rewards", "com_forward_margin", "target"): 0.055,
    ("rewards", "com_forward_margin", "standing"): "cap",
    ("rewards", "track_lin_vel_y_cycle", "weight"): 1.0,
    ("events", "randomize_base_com", "com_range"): {"x": (-0.03, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
}


def constants(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def evaluate(node, consts):
    return eval(compile(ast.Expression(node), "cfg", "eval"), {"__builtins__": {"dict": dict}}, consts)


def term(tree, cls_name, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    node = next(n for n in cls.body if isinstance(n, ast.Assign) and n.targets[0].id == name)
    return {k.arg: k.value for k in node.value.keywords}


class RecipeDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(CFG.read_text())
        self.consts = constants(self.tree)

    def leaf(self, group, name, key):
        kw = term(self.tree, {"rewards": "RewardsCfg", "events": "EventCfg"}[group], name)
        if key == "weight":
            return evaluate(kw["weight"], self.consts)
        params = {ast.literal_eval(k): v for k, v in zip(kw["params"].keys, kw["params"].values)}
        return evaluate(params[key], self.consts)

    def test_code_defaults_are_the_chain_recipe(self):
        for (group, name, key), want in RECIPE.items():
            self.assertEqual(self.leaf(group, name, key), want, (group, name, key))

    def test_imu_delay_default_is_the_chain_recipe(self):
        self.assertEqual(self.consts["IMU_DELAY_STEPS"], (1, 6))

    def test_hf_gyro_default_is_the_chain_recipe(self):
        tree = ast.parse(AGENT.read_text())
        kw = term(tree, "PPORunnerCfg", "algorithm")
        self.assertEqual(ast.literal_eval(kw["hf_gyro_coef"]), 0.0375)


if __name__ == "__main__":
    unittest.main()
