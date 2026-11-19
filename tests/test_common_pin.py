import ast
import importlib.util
import re
import unittest
from pathlib import Path
from unittest import mock

import robonex_common

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source/robonex_walking"
CONTRACT = PKG / "robonex_walking/tasks/manager_based/robonex_walking/robot_contract_v2.py"
SETUP = PKG / "setup.py"


def pinned_version():
    for node in ast.parse(CONTRACT.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "ROBONEX_COMMON_VERSION":
            return ast.literal_eval(node.value)
    raise KeyError("ROBONEX_COMMON_VERSION")


def load_contract(name):
    spec = importlib.util.spec_from_file_location(name, CONTRACT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CommonPinTest(unittest.TestCase):
    def test_installed_common_is_the_pinned_version(self):
        self.assertEqual(robonex_common.__version__, pinned_version())

    def test_setup_takes_the_pin_from_the_contract(self):
        text = SETUP.read_text()
        self.assertIsNone(re.search(r"robonex-common\.git@v\d", text))
        self.assertIn("@v{ROBONEX_COMMON_VERSION}", text)
        self.assertIn("robot_contract_v2.py", text)

    def test_contract_import_refuses_another_common_version(self):
        with mock.patch.object(robonex_common, "__version__", "0.0.0"):
            with self.assertRaisesRegex(ImportError, "pinned to robonex-common"):
                load_contract("robot_contract_v2_wrong_common")

    def test_contract_imports_with_the_pinned_version(self):
        module = load_contract("robot_contract_v2_pinned_common")
        self.assertEqual(module.ROBONEX_COMMON_VERSION, robonex_common.__version__)


if __name__ == "__main__":
    unittest.main()
