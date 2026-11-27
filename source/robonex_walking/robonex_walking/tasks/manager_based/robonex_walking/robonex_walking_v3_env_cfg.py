import dataclasses

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from . import robot_contract_v3 as v3
from .robonex_walking_v2_env_cfg import CONTACT_FORCE_LIMIT_BW, GRAVITY, RoboNexWalkingV2EduEnvCfg

_RENAME = dict(v3.V2_TO_V3)

NECK_TARGET_MARGIN_RAD = 0.05
NECK_TARGET_RANGE_SCALE = 0.0
NECK_TARGET_RESAMPLE_S = (1.0, 3.0)
NECK_TARGET_BLEND_S = 1.0


def _rename(obj, seen=None):
    seen = set() if seen is None else seen
    if isinstance(obj, str):
        return _RENAME.get(obj, obj)
    if isinstance(obj, (int, float, bool)) or obj is None or callable(obj) and not dataclasses.is_dataclass(obj):
        return obj
    if id(obj) in seen:
        return obj
    if isinstance(obj, dict):
        seen.add(id(obj))
        items = [(_rename(k, seen), _rename(v, seen)) for k, v in obj.items()]
        obj.clear()
        obj.update(items)
        return obj
    if isinstance(obj, list):
        seen.add(id(obj))
        obj[:] = [_rename(v, seen) for v in obj]
        return obj
    if isinstance(obj, tuple):
        return type(obj)(_rename(v, seen) for v in obj) if not hasattr(obj, "_fields") else type(obj)(*(_rename(v, seen) for v in obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        seen.add(id(obj))
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            new = _rename(value, seen)
            if new is not value:
                setattr(obj, field.name, new)
        return obj
    return obj


def _upper_actuators():
    actuators = {}
    for role, motor in v3.UPPER_MOTOR.items():
        kp, kd = v3.UPPER_GAINS[role]
        actuators["upper_" + role] = ImplicitActuatorCfg(
            joint_names_expr=[".*%s_joint" % ("_" + role if role.startswith(("shoulder", "elbow", "wrist")) else role)],
            stiffness=kp,
            damping=kd,
            armature=v3.UPPER_ARMATURE[motor],
            effort_limit_sim=v3.PEAK_TORQUE[motor],
            velocity_limit_sim=v3.NO_LOAD_SPEED[motor],
        )
    return actuators


@configclass
class RoboNexWalkingV3EnvCfg(RoboNexWalkingV2EduEnvCfg):
    model_variant: str = "full"
    feet_body_names: tuple = v3.FEET

    def __post_init__(self) -> None:
        super().__post_init__()
        _rename(self.actions)
        _rename(self.observations)
        _rename(self.events)
        _rename(self.rewards)
        _rename(self.terminations)
        _rename(self.curriculum)
        _rename(self.commands)
        self.actions.joint_pos.joint_names = list(v3.LEG_JOINTS)
        self.actions.joint_pos.offset = dict(v3.ACTION_OFFSETS)
        self.actions.joint_pos.scale = dict(v3.ACTION_SCALES)
        self.actions.joint_pos.clip = dict(v3.ACTION_CLIPS)
        self.actions.joint_pos.foot_roll_pairs = v3.FOOT_ROLL_PAIRS
        robot = self.scene.robot
        robot.spawn.usd_path = str(v3.robot_usd(self.model_variant))
        robot.init_state.pos = (0.0, 0.0, v3.BASE_HEIGHT)
        robot.init_state.joint_pos = v3.closed_loop_default_joint_pos(self.model_variant)
        legs = {k: v for k, v in robot.actuators.items() if k in ("rs02", "rs03")}
        for cfg in legs.values():
            for attr in ("stiffness", "damping"):
                value = getattr(cfg, attr)
                if isinstance(value, dict):
                    setattr(cfg, attr, {_RENAME.get(k, k): w for k, w in value.items()})
        if self.model_variant == "full":
            legs.update(_upper_actuators())
        robot.actuators = legs
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*_ankle_pitch_link"
        self.scene.illegal_contacts.prim_path = (
            "{ENV_REGEX_NS}/Robot/(base_link|.*_knee_link|.*_ankle_roll_link|.*_hip_.*_link"
            "|.*_shoulder_.*_link|.*_elbow_link|.*_wrist_roll_link|neck_.*_link|head_link)"
        )
        self.foot_origin_rest_height = v3.FOOT_ORIGIN_REST_HEIGHT
        self.foot_sole_corners = v3.FOOT_SOLE_CORNERS
        self.rewards.feet_contact_force.params["threshold"] = round(CONTACT_FORCE_LIMIT_BW * v3.MASS_KG * GRAVITY, 1)
        self._apply_w197_recipe()
        if self.model_variant == "full":
            self.events.neck_target = self._neck_target_term()

    def _neck_target_term(self):
        step_s = self.sim.dt * self.decimation
        return EventTerm(
            func=mdp.randomize_neck_target,
            mode="interval",
            interval_range_s=(step_s, step_s),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(v3.NECK_JOINTS), preserve_order=True),
                "limits": {name: tuple(lim) for name, lim in v3.NECK_LIMITS.items()},
                "margin": NECK_TARGET_MARGIN_RAD,
                "range_scale": NECK_TARGET_RANGE_SCALE,
                "resample_s": NECK_TARGET_RESAMPLE_S,
                "blend_s": NECK_TARGET_BLEND_S,
                "start_at_default": False,
                "write_state_at_reset": True,
            },
        )

    def _apply_w197_recipe(self) -> None:
        r = self.rewards
        r.slew_lag.weight = 0.0
        r.stand_still.weight = -2.0
        r.com_forward_margin.weight = -0.5
        r.com_forward_margin.params["target"] = 0.055
        r.com_forward_margin.params["standing"] = "cap"
        r.track_lin_vel_y_cycle.weight = 1.0
        r.joint_load.weight = -0.02
        r.joint_load.params["asset_cfg"].joint_names = [
            "left_ankle_upper_joint", "left_ankle_lower_joint", "right_ankle_upper_joint", "right_ankle_lower_joint"]
        r.feet_gait.params["threshold"] = 0.625
        r.feet_clearance.params["stance_fraction"] = 0.625
        r.feet_width.params["lateral_deadband"] = 0.44
        self.scene.imu.min_delay_steps = 1
        self.scene.imu.max_delay_steps = 6
        for term in (self.observations.policy.joint_pos_rel, self.observations.policy.joint_vel_rel):
            term.params["min_delay_steps"] = 1
            term.params["max_delay_steps"] = 2
        self.events.randomize_base_com.params["com_range"]["x"] = (-0.03, 0.0)
        deg = 3.141592653589793 / 180.0
        zero = {"hip_yaw": 1.0, "hip_pitch": 1.5, "hip_roll": 1.0, "knee_pitch": 2.0, "ankle_upper": 2.0, "ankle_lower": 2.0}
        self.events.randomize_joint_zero.params["offset_range"] = {
            f"{side}_{role}_joint": (-d * deg, d * deg) for side in ("left", "right") for role, d in zero.items()}
        self.events.randomize_hip_yaw_stiffness.params["stiffness_distribution_params"] = (1.0, 1.0)


@configclass
class RoboNexWalkingV3LegsEnvCfg(RoboNexWalkingV3EnvCfg):
    model_variant: str = "legs"
