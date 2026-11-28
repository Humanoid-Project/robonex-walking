import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from robonex_common.joints import VER3_POLICY_JOINT_ORDER

TASK = Path(__file__).resolve().parents[1] / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
DESC = Path(__file__).resolve().parents[2] / "robonex-description"
HAS_V3 = (DESC / "ver3/ver3_constants.json").exists()


def _load(name, pkg):
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(TASK)]
        sys.modules[pkg] = mod
    spec = importlib.util.spec_from_file_location(f"{pkg}.{name}", TASK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_V3, "robonex-description ver3 not present")
class V3ContractOrderTest(unittest.TestCase):
    def test_leg_articulation_order_is_commons_ver3_order(self):
        v3 = _load("robot_contract_v3", "robonex_v3_order_test")
        self.assertEqual(v3.LEG_ARTICULATION_ORDER, VER3_POLICY_JOINT_ORDER)
        self.assertEqual(set(v3.LEG_JOINTS), set(VER3_POLICY_JOINT_ORDER))

    def test_import_fails_when_the_description_order_differs(self):
        import tempfile

        constants = json.loads((DESC / "ver3/ver3_constants.json").read_text())
        constants["policy_joint_order"] = list(reversed(constants["policy_joint_order"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ver3" / "ver3_constants.json"
            path.parent.mkdir()
            path.write_text(json.dumps(constants))
            (Path(tmp) / "ver2").mkdir()
            (Path(tmp) / "ver2" / "ver2_constants.json").write_text((DESC / "ver2/ver2_constants.json").read_text())
            import os

            old = os.environ.get("ROBONEX_DESCRIPTION_ROOT")
            os.environ["ROBONEX_DESCRIPTION_ROOT"] = tmp
            try:
                with self.assertRaisesRegex(RuntimeError, "differs from robonex-common VER3_POLICY_JOINT_ORDER"):
                    _load("robot_contract_v3", "robonex_v3_order_bad_test")
            finally:
                if old is None:
                    os.environ.pop("ROBONEX_DESCRIPTION_ROOT")
                else:
                    os.environ["ROBONEX_DESCRIPTION_ROOT"] = old


if __name__ == "__main__":
    unittest.main()
