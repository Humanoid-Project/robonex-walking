import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "scripts/export_policy_manifest.py"
DESC = ROOT.parent / "robonex-description"
REAL_V3 = sorted((ROOT / "logs/rsl_rl/robonex_walking_v3").glob("*_WA2s43_*/params/env.yaml"))
MESSAGE = "ver3 export needs a ver3 robot profile in robonex-common (planned 2.6.0)"
ENV = """scene:
  robot:
    spawn:
      usd_path: {usd}
actions:
  joint_pos:
    joint_names: !!python/tuple
    - {joint}
    offset: {{}}
    clip: {{}}
    scale: {{}}
    preserve_order: false
observations:
  policy:
    joint_pos_rel:
      params:
        asset_cfg:
          joint_ids: !!python/object/apply:builtins.slice
          - null
          - null
          - null
"""
V3_USD = "/x/robonex-description/ver3/isaac/{v}/closed_loop_mesh/robonex_ver3_{v}_closed_loop_mesh.usd"
V2_USD = "/x/robonex-description/ver2/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd"


def export_module():
    spec = importlib.util.spec_from_file_location("export_policy_manifest_ver3", EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportVer3RefusalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export = export_module()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def run_env(self, usd, joint="left_hip_yaw_joint", name="2026-11-27_00-00-00_WA9s43_x"):
        path = Path(self.tmp.name) / name / "params" / "env.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(ENV.format(usd=usd, joint=joint))
        return path

    def test_ver3_runs_are_refused_with_the_profile_message(self):
        for variant in ("full", "legs"):
            msg = self.export.ver3_export_refusal(self.run_env(V3_USD.format(v=variant), name=f"r_{variant}"))
            self.assertIn(MESSAGE, msg)
            self.assertIn(f"trained on ver3/{variant}", msg)

    def test_ver2_and_unknown_runs_pass_through(self):
        self.assertIsNone(self.export.ver3_export_refusal(self.run_env(V2_USD, joint="l_hip_yaw_joint", name="a")))
        self.assertIsNone(self.export.ver3_export_refusal(self.run_env("/somewhere/robot.usd", name="b")))
        self.assertIsNone(self.export.ver3_export_refusal(
            self.run_env(V2_USD.replace("/ver2/", "/new_urdf/"), joint="l_hip_yaw_joint", name="c")))

    @unittest.skipUnless(REAL_V3, "WA2s43 run not present")
    def test_real_ver3_run_is_refused(self):
        self.assertIn("trained on ver3/legs", self.export.ver3_export_refusal(REAL_V3[0]))

    @unittest.skipUnless(DESC.is_dir(), "robonex-description not present")
    def test_cli_stops_before_the_ver2_normalisation(self):
        saved = self.run_env(V3_USD.format(v="full"))
        policy = Path(self.tmp.name) / "policy.onnx"
        policy.write_bytes(b"x")
        proc = subprocess.run([sys.executable, str(EXPORT), str(policy), "--checkpoint",
                               str(saved.parent.parent / "model_999.pt")], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(MESSAGE, proc.stderr)
        self.assertNotIn("action normalisation lacks", proc.stderr)
        self.assertFalse((Path(self.tmp.name) / "policy_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
