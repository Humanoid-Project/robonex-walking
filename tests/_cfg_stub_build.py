import importlib
import importlib.machinery
import json
import sys
import types


class _Meta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub

    def __getitem__(cls, key):
        return _Stub

    def __or__(cls, other):
        return _Stub

    def __ror__(cls, other):
        return _Stub

    def __repr__(cls):
        return "<stub>"


class _Stub(metaclass=_Meta):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()

    def __getattr__(self, name):
        return _Stub()

    def __repr__(self):
        return "<stub>"


class _AnyModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub


class _Finder:
    PREFIXES = ("omni", "carb", "isaacsim")

    def find_spec(self, name, path=None, target=None):
        if name == "pxr" or name.startswith("pxr.") or name.startswith(self.PREFIXES):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _AnyModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


def _encode(obj):
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return ["__tuple__"] + [_encode(v) for v in obj]
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    if isinstance(obj, str):
        return _portable(obj)
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return _portable(str(obj))


def _portable(text):
    marker = "/robonex-description/"
    return "robonex-description/" + text.split(marker, 1)[1] if marker in text else text


def main(source_root, isaaclab_root, v2_only=False):
    sys.meta_path.insert(0, _Finder())
    sys.path[:0] = [source_root, isaaclab_root]
    pkg = "robonex_walking.tasks.manager_based.robonex_walking"
    v2 = importlib.import_module(pkg + ".robonex_walking_v2_env_cfg")
    from isaaclab.utils import class_to_dict

    before = class_to_dict(v2.RoboNexWalkingV2EduEnvCfg())
    v3 = importlib.import_module(pkg + ".robonex_walking_v3_env_cfg")
    full = class_to_dict(v3.RoboNexWalkingV3EnvCfg())
    legs = class_to_dict(v3.RoboNexWalkingV3LegsEnvCfg())
    after = class_to_dict(v2.RoboNexWalkingV2EduEnvCfg())
    if v2_only:
        json.dump(_encode(before), sys.stdout, sort_keys=True, indent=1)
        sys.stdout.write("\n")
        return
    out = {"v2_before": before, "v2_after": after, "v3": full, "v3_legs": legs, "rename": dict(v3._RENAME)}
    json.dump(_encode(out), sys.stdout, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--v2-golden" in sys.argv[3:])
