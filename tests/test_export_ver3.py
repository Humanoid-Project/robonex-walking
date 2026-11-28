import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from robonex_common.joints import POLICY_JOINT_ORDER, VER3_POLICY_JOINT_ORDER
from robonex_common.limits import action_normalization
from robonex_common.models import robot_model
from robonex_common.policy import PolicyContract

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "scripts/export_policy_manifest.py"
DESC = Path(os.environ.get("ROBONEX_DESCRIPTION_ROOT", ROOT.parent / "robonex-description"))
COMMON = Path(os.environ.get("ROBONEX_COMMON_ROOT", ROOT.parent / "robonex-common"))
REAL_V3 = sorted((ROOT / "logs/rsl_rl/robonex_walking_v3").glob("*_WA4s43_v3full_*/model_1999.pt"))
V3_USD = "/x/robonex-description/ver3/isaac/{v}/closed_loop_mesh/robonex_ver3_{v}_closed_loop_mesh.usd"
V2_USD = "/x/robonex-description/ver2/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd"
TERMS = ("joint_pos_rel", "joint_vel_rel", "imu_ang_vel", "projected_gravity", "velocity_commands", "gait_phase", "actions")
CAN_RUN = DESC.is_dir() and COMMON.is_dir()


def export_module():
    spec = importlib.util.spec_from_file_location("export_policy_manifest_ver3", EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def env_dict(model, usd):
    offsets, scales, clips = action_normalization(0.01, model=model)
    roll = robot_model(model).foot_roll
    order = robot_model(model).policy_joint_order
    policy = {"concatenate_terms": True, "history_length": 5, "flatten_history_dim": True}
    policy.update({name: {"func": "x"} for name in TERMS})
    return {
        "sim": {"dt": 0.0025},
        "decimation": 8,
        "scene": {"robot": {"spawn": {"usd_path": usd}}},
        "observations": {"policy": policy},
        "actions": {"joint_pos": {
            "joint_names": list(reversed(order)),
            "preserve_order": False,
            "offset": dict(offsets),
            "scale": dict(scales),
            "clip": {k: list(v) for k, v in clips.items()},
            "foot_roll_limit": roll.limit,
            "foot_roll_coeffs": list(roll.coeffs),
            "foot_roll_pairs": [list(p) for p in roll.pairs],
        }},
    }


def write_onnx(path, width=235):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["obs"], ["actions"])], "g",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, width])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))


class Ver3ExportUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export = export_module()

    def test_ver3_maps_to_its_free_base_scenes_and_tasks(self):
        self.assertEqual(self.export.VER3_DESCRIPTION_MODELS["full"], "ver3/mujoco/robot/full/scene.xml")
        self.assertEqual(self.export.VER3_DESCRIPTION_MODELS["legs"], "ver3/mujoco/robot/legs/scene.xml")
        self.assertEqual(self.export.VER3_TASKS, {"full": "RoboNex-Walking-V3-v0", "legs": "RoboNex-Walking-V3-Legs-v0"})
        self.assertIn("ver3", self.export.ROBOT_MODEL_CHOICES)
        self.assertIn("ver2_edu", self.export.ROBOT_MODEL_CHOICES)

    def test_ver2_defaults_are_unchanged(self):
        self.assertEqual(self.export.TASKS, {"ver2_edu": "RoboNex-Walking-V2-Edu-v0"})
        self.assertEqual(self.export.DESCRIPTION_MODELS, {"ver2_edu": "ver2/mujoco/robot/edu/scene.xml"})

    @unittest.skipUnless(DESC.is_dir(), "robonex-description not present")
    def test_ver3_scenes_have_the_free_root_joint(self):
        for scene in self.export.VER3_DESCRIPTION_MODELS.values():
            self.assertIn('<include file="robonex.xml"/>', (DESC / scene).read_text(), scene)
            self.assertIn('<freejoint name="root"/>', ((DESC / scene).parent / "robonex.xml").read_text(), scene)

    def test_description_model_refusal(self):
        refusal = self.export.description_model_refusal
        full = "ver3/mujoco/robot/full/scene.xml"
        self.assertIsNone(refusal(None, "ver3", full))
        self.assertIsNone(refusal(full, "ver3", full))
        self.assertIn("not a model of robot ver3", refusal("ver2/mujoco/robot/edu/scene.xml", "ver3", full))
        self.assertIn("trained on", refusal("ver3/mujoco/robot/legs/scene.xml", "ver3", full))
        self.assertIn("not a model of robot ver2_edu", refusal(full, "ver2_edu", "ver2/mujoco/robot/edu/scene.xml"))
        self.assertIsNone(refusal("ver2/mujoco/robot/edu/scene.xml", "ver2_edu", "ver2/mujoco/robot/edu/scene.xml"))


@unittest.skipUnless(CAN_RUN, "robonex-description or robonex-common not next to this checkout")
class Ver3ExportCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_dir(self, model, usd, name, joint_order=None):
        run = self.root / name
        (run / "params").mkdir(parents=True)
        (run / "params" / "env.yaml").write_text(yaml.safe_dump(env_dict(model, usd), sort_keys=False))
        if joint_order is not None:
            (run / "params" / "joint_order.json").write_text(json.dumps(joint_order))
        (run / "model_1.pt").write_bytes(b"x")
        write_onnx(run / "policy.onnx")
        return run

    def export(self, run, *extra):
        env = dict(os.environ, ROBONEX_DESCRIPTION_ROOT=str(DESC), ROBONEX_COMMON_ROOT=str(COMMON))
        output = run / "policy_manifest.json"
        proc = subprocess.run([sys.executable, str(EXPORT), str(run / "policy.onnx"), "--checkpoint",
                               str(run / "model_1.pt"), "--output", str(output), *extra],
                              capture_output=True, text=True, timeout=180, env=env)
        return proc, output

    def assert_refused(self, proc, output, text):
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn(text, proc.stderr)
        self.assertFalse(output.exists())

    def test_ver3_full_and_legs_manifests(self):
        for variant, task in (("full", "RoboNex-Walking-V3-v0"), ("legs", "RoboNex-Walking-V3-Legs-v0")):
            run = self.run_dir("ver3", V3_USD.format(v=variant), f"2026-11-27_00-00-00_WA9s43_{variant}")
            proc, output = self.export(run)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            contract = PolicyContract.load(output)
            contract.validate()
            data = json.loads(output.read_text())
            self.assertEqual(data["robot_model"], "ver3")
            self.assertEqual(data["task"], task)
            self.assertEqual(data["description_model"], f"ver3/mujoco/robot/{variant}/scene.xml")
            self.assertEqual(tuple(data["joint_order"]), VER3_POLICY_JOINT_ORDER)
            offsets, scales, clips = action_normalization(0.01, model="ver3")
            self.assertEqual(data["action_offsets"], [offsets[n] for n in VER3_POLICY_JOINT_ORDER])
            self.assertEqual(data["action_scales"], [scales[n] for n in VER3_POLICY_JOINT_ORDER])
            self.assertEqual(data["target_clips"], [list(clips[n]) for n in VER3_POLICY_JOINT_ORDER])
            self.assertEqual(data["foot_roll_pairs"][0][:2], ["left_ankle_upper_joint", "left_ankle_lower_joint"])
            self.assertEqual(data["observation_size"], 235)
            self.assertEqual(data["policy_hz"], 50.0)

    def test_ver3_with_a_matching_joint_order_record(self):
        record = {"task": "RoboNex-Walking-V3-v0", "action_joint_names": list(VER3_POLICY_JOINT_ORDER)}
        run = self.run_dir("ver3", V3_USD.format(v="full"), "2026-11-27_00-00-00_WA9s43_rec", record)
        proc, output = self.export(run)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("joint_order.json", proc.stdout)

    def test_ver3_with_a_wrong_joint_order_record_is_refused(self):
        record = {"action_joint_names": list(reversed(VER3_POLICY_JOINT_ORDER))}
        run = self.run_dir("ver3", V3_USD.format(v="full"), "2026-11-27_00-00-00_WA9s43_bad", record)
        self.assert_refused(*self.export(run), "differs from robonex-common VER3_POLICY_JOINT_ORDER")

    def test_ver3_with_a_record_of_the_other_variant_is_refused(self):
        record = {"task": "RoboNex-Walking-V3-Legs-v0", "action_joint_names": list(VER3_POLICY_JOINT_ORDER)}
        run = self.run_dir("ver3", V3_USD.format(v="full"), "2026-11-27_00-00-00_WA9s43_task", record)
        self.assert_refused(*self.export(run), "but the run's robot is ver3/full")

    def test_ver3_checkpoint_exported_as_ver2_is_refused(self):
        run = self.run_dir("ver3", V3_USD.format(v="full"), "2026-11-27_00-00-00_WA9s43_as2")
        self.assert_refused(*self.export(run, "--robot-model", "ver2_edu"), "trained as ver3")
        self.assert_refused(*self.export(run, "--description-model", "ver2/mujoco/robot/edu/scene.xml"),
                            "not a model of robot ver3")
        self.assert_refused(*self.export(run, "--task", "RoboNex-Walking-V2-Edu-v0"), "--task RoboNex-Walking-V2-Edu-v0")
        self.assert_refused(*self.export(run, "--description-model", "ver3/mujoco/robot/legs/scene.xml"), "trained on")

    def test_ver2_checkpoint_exported_as_ver3_is_refused(self):
        run = self.run_dir("ver2_edu", V2_USD, "2026-11-25_00-00-00_W999s43_as3")
        self.assert_refused(*self.export(run, "--robot-model", "ver3"), "trained as ver2_edu")
        self.assert_refused(*self.export(run, "--description-model", "ver3/mujoco/robot/full/scene.xml"),
                            "not a model of robot ver2_edu")
        self.assert_refused(*self.export(run, "--task", "RoboNex-Walking-V3-v0"), "--task RoboNex-Walking-V3-v0")

    def test_ver3_names_with_a_ver2_usd_are_refused(self):
        run = self.run_dir("ver3", V2_USD, "2026-11-25_00-00-00_W999s43_mix")
        self.assert_refused(*self.export(run), "action normalisation lacks")

    def test_ver2_manifest_keeps_the_ver2_contract(self):
        run = self.run_dir("ver2_edu", V2_USD, "2026-11-25_00-00-00_W999s43_v2")
        proc, output = self.export(run)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(output.read_text())
        self.assertEqual(data["robot_model"], "ver2_edu")
        self.assertEqual(data["task"], "RoboNex-Walking-V2-Edu-v0")
        self.assertEqual(data["description_model"], "ver2/mujoco/robot/edu/scene.xml")
        self.assertEqual(tuple(data["joint_order"]), POLICY_JOINT_ORDER)
        PolicyContract.load(output).validate()

    @unittest.skipUnless(REAL_V3, "WA4s43 run not present")
    def test_real_ver3_run_exports(self):
        checkpoint = REAL_V3[0]
        env = dict(os.environ, ROBONEX_DESCRIPTION_ROOT=str(DESC), ROBONEX_COMMON_ROOT=str(COMMON))
        policy = self.root / "policy.onnx"
        write_onnx(policy)
        proc = subprocess.run([sys.executable, str(EXPORT), str(policy), "--checkpoint", str(checkpoint)],
                              capture_output=True, text=True, timeout=180, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        contract = PolicyContract.load(self.root / "policy_manifest.json")
        contract.validate()
        self.assertEqual((contract.robot_model, contract.task, contract.description_model),
                         ("ver3", "RoboNex-Walking-V3-v0", "ver3/mujoco/robot/full/scene.xml"))


if __name__ == "__main__":
    unittest.main()
