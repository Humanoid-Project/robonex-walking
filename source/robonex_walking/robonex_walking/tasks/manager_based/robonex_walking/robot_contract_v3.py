import json
import math

from robonex_common.paths import DESCRIPTION_REPO_NAMES, repo_file

from . import robot_contract_v2 as v2

MODEL_VERSION = "ver3"
VARIANTS = ("full", "legs")


def _constants():
    path = repo_file(
        DESCRIPTION_REPO_NAMES,
        "%s/%s_constants.json" % (MODEL_VERSION, MODEL_VERSION),
        env_var="ROBONEX_DESCRIPTION_ROOT",
        anchors=(__file__,),
    )
    return path, json.loads(path.read_text())


CONSTANTS_PATH, CONSTANTS = _constants()


def robot_usd(variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown ver3 variant {variant!r}; expected one of {VARIANTS}")
    return repo_file(
        DESCRIPTION_REPO_NAMES,
        "%s/isaac/%s/closed_loop_mesh/robonex_%s_%s_closed_loop_mesh.usd" % (MODEL_VERSION, variant, MODEL_VERSION, variant),
        env_var="ROBONEX_DESCRIPTION_ROOT",
        anchors=(__file__,),
    )


BODY_MAP = {
    "l_foot": "left_ankle_pitch_link",
    "r_foot": "right_ankle_pitch_link",
    "l_ankle_link": "left_ankle_roll_link",
    "r_ankle_link": "right_ankle_roll_link",
}


def v3_name(name):
    if name in BODY_MAP:
        return BODY_MAP[name]
    if name.startswith("l_"):
        return "left_" + name[2:]
    if name.startswith("r_"):
        return "right_" + name[2:]
    return name


LEG_JOINTS = tuple(v3_name(n) for n in v2.LEG_JOINTS)
V2_TO_V3 = {n: v3_name(n) for n in v2.LEG_JOINTS}
V2_TO_V3.update(BODY_MAP)
V2_TO_V3.update({n: v3_name(n) for n in ("l_hip_roll_link", "r_hip_roll_link", "l_knee_link", "r_knee_link")})
FEET = ("left_ankle_pitch_link", "right_ankle_pitch_link")

ARM_JOINTS = tuple(
    "%s_%s_joint" % (side, role)
    for side in ("left", "right")
    for role in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
)
NECK_JOINTS = ("neck_yaw_joint", "neck_pitch_joint")
UPPER_JOINTS = ARM_JOINTS + NECK_JOINTS

MASS_KG = float(CONSTANTS["variant_mass_kg"][next(iter(CONSTANTS["variant_mass_kg"]))])
BASE_HEIGHT = CONSTANTS["home_base_height"]
FOOT_ORIGIN_REST_HEIGHT = CONSTANTS["foot_origin_rest_height"]
FOOT_SOLE_CORNERS = tuple(tuple(c) for c in CONSTANTS["foot_sole_corners"])

ACTION_OFFSETS = {V2_TO_V3[k]: v for k, v in v2.ACTION_OFFSETS.items()}
ACTION_SCALES = {V2_TO_V3[k]: v for k, v in v2.ACTION_SCALES.items()}
ACTION_CLIPS = {V2_TO_V3[k]: v for k, v in v2.ACTION_CLIPS.items()}
FOOT_ROLL_PAIRS = tuple((v3_name(u), v3_name(l), s) for u, l, s in v2.FOOT_ROLL_PAIRS)
JOINT_ZERO_OFFSET_RANGE = {name: (0.0, 0.0) for name in LEG_JOINTS}

_default = dict(CONSTANTS["default_actuated_pos"])
for _name in LEG_JOINTS:
    if abs(_default[_name] - ACTION_OFFSETS[_name]) > 1e-9:
        raise RuntimeError(f"{_name}: ver3 default {_default[_name]} differs from the ver2 leg default {ACTION_OFFSETS[_name]}")
for _name in UPPER_JOINTS:
    if abs(_default.get(_name, 0.0)) > 1e-9:
        raise RuntimeError(f"{_name}: ver3 default pose is expected to be 0, got {_default[_name]}")
_limits = {k: tuple(v) for k, v in CONSTANTS["provisional_limits"].items()}
for _name in LEG_JOINTS:
    v2_lim = v2.JOINT_LIMITS[next(k for k, v in V2_TO_V3.items() if v == _name)]
    if max(abs(a - b) for a, b in zip(_limits[_name], v2_lim)) > 1e-9:
        raise RuntimeError(f"{_name}: ver3 limits {_limits[_name]} differ from ver2 {v2_lim}")
NECK_LIMITS = {name: _limits[name] for name in NECK_JOINTS}


def _isaac_passive_name(name):
    for ax, idx in (("_x", ":0"), ("_y", ":1"), ("_z", ":2")):
        if "_coupler_joint" in name and name.endswith(ax) and name.removesuffix(ax).endswith("_coupler_joint"):
            return name[: -len(ax)] + idx
    return name


def closed_loop_default_joint_pos(variant):
    pos = {name: float(_default[name]) for name in LEG_JOINTS}
    pos.update({_isaac_passive_name(n): v for n, v in CONSTANTS["home_passive_pos"].items()})
    if variant == "full":
        pos.update({name: 0.0 for name in UPPER_JOINTS})
    return pos


UPPER_MOTOR = {
    "shoulder_pitch": "rs02",
    "shoulder_roll": "rs10p",
    "shoulder_yaw": "rs00",
    "elbow": "rs00",
    "wrist_roll": "rs05",
    "neck_yaw": "rs05",
    "neck_pitch": "rs05",
}
UPPER_GAINS = {
    "shoulder_pitch": (40.0, 2.0),
    "shoulder_roll": (40.0, 2.0),
    "shoulder_yaw": (40.0, 2.0),
    "elbow": (40.0, 2.0),
    "wrist_roll": (10.0, 0.5),
    "neck_yaw": (20.0, 1.0),
    "neck_pitch": (20.0, 1.0),
}
UPPER_ARMATURE = {"rs02": 0.0042, "rs10p": 0.02, "rs00": 0.0042, "rs05": 0.001}
PEAK_TORQUE = {"rs02": 17.0, "rs03": 60.0, "rs05": 5.5, "rs00": 14.0, "rs10p": 42.0}
NO_LOAD_SPEED = {"rs02": 42.9, "rs03": 20.9, "rs05": 50.3, "rs00": 33.0, "rs10p": 13.1}

if not math.isfinite(MASS_KG) or MASS_KG < 20.0:
    raise RuntimeError(f"ver3 mass {MASS_KG} looks wrong ({CONSTANTS_PATH})")
