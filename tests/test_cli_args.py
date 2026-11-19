import argparse
import ast
import importlib.util
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_ARGS = ROOT / "scripts/rsl_rl/cli_args.py"
PLAY = ROOT / "scripts/rsl_rl/play.py"


def cli_args():
    spec = importlib.util.spec_from_file_location("cli_args_under_test", CLI_ARGS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def agent_cfg():
    return types.SimpleNamespace(seed=42, resume=False, load_run=".*", load_checkpoint="model_.*.pt", run_name="",
                                 logger="tensorboard", experiment_name="robonex_walking_v2_edu",
                                 wandb_project="isaaclab", neptune_project="isaaclab")


class CliArgsTest(unittest.TestCase):
    def parse(self, argv):
        module = cli_args()
        parser = argparse.ArgumentParser()
        module.add_rsl_rl_args(parser)
        parser.add_argument("--seed", type=int, default=None)
        return module, parser.parse_args(argv)

    def test_experiment_name_is_applied(self):
        module, args = self.parse(["--experiment_name", "probe"])
        cfg = module.update_rsl_rl_cfg(agent_cfg(), args)
        self.assertEqual(cfg.experiment_name, "probe")

    def test_without_experiment_name_the_task_folder_stays(self):
        module, args = self.parse(["--run_name", "x"])
        cfg = module.update_rsl_rl_cfg(agent_cfg(), args)
        self.assertEqual(cfg.experiment_name, "robonex_walking_v2_edu")
        self.assertEqual(cfg.run_name, "x")

    def test_namespace_without_the_attribute_is_accepted(self):
        args = types.SimpleNamespace(seed=None, resume=None, load_run=None, checkpoint=None, run_name=None,
                                     logger=None, log_project_name=None)
        cfg = cli_args().update_rsl_rl_cfg(agent_cfg(), args)
        self.assertEqual(cfg.experiment_name, "robonex_walking_v2_edu")


class PlayExportDirTest(unittest.TestCase):
    def test_export_goes_to_a_checkpoint_specific_folder(self):
        main = next(n for n in ast.parse(PLAY.read_text()).body if getattr(n, "name", None) == "main")
        assigns = {n.targets[0].id: n.value for n in ast.walk(main)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        ns = {"os": __import__("os"), "resume_path": "/logs/run/model_1999.pt"}
        stem = eval(compile(ast.Expression(assigns["checkpoint_stem"]), "play", "eval"), ns)
        ns["checkpoint_stem"] = stem
        export_dir = eval(compile(ast.Expression(assigns["export_model_dir"]), "play", "eval"), ns)
        self.assertEqual(export_dir, "/logs/run/exported/model_1999")


if __name__ == "__main__":
    unittest.main()
