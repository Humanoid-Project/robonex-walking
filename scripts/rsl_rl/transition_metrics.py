import math

import torch

TRANSITION_COMMANDS = {
    "tr_vx01": (0.1, 0.0, 0.0),
    "tr_vx02": (0.2, 0.0, 0.0),
    "tr_vx03": (0.3, 0.0, 0.0),
    "tr_vy01": (0.0, 0.1, 0.0),
    "tr_wz01": (0.0, 0.0, 0.1),
}
TRANSITION_CELLS = {}
for _name, _command in TRANSITION_COMMANDS.items():
    TRANSITION_CELLS[_name] = {"command": _command, "push": False}
    TRANSITION_CELLS[_name + "_push"] = {"command": _command, "push": True}

PHASE_BINS = 8
TOUCHDOWN_WINDOW_S = 0.5
SETTLE_SAMPLE_S = 2.0
LIFT_COUNT_FROM_S = 1.0
PUSH_AT_S = 2.0


def expand_cells(names):
    out = []
    for name in names:
        if name == "transition":
            out.extend(TRANSITION_CELLS)
        else:
            out.append(name)
    return out


def steps_per_period(period, step_dt):
    steps = period / step_dt
    if abs(steps - round(steps)) > 1e-6:
        raise ValueError(f"gait period {period} s is not a whole number of {step_dt} s steps")
    return int(round(steps))


def draw_switch_steps(num_envs, warmup_steps, period_steps, generator=None):
    return warmup_steps + torch.randint(0, period_steps, (num_envs,), generator=generator)


def window_steps(after_s, step_dt):
    return int(round(after_s / step_dt))


def _stats(values):
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    values = values.double()
    return {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.5)),
        "p95": float(torch.quantile(values, 0.95)),
    }


class TransitionRecorder:
    def __init__(self, switch_step, step_dt, after_steps, joint_names, push_step=None):
        self.switch = switch_step.long()
        self.n = int(self.switch.numel())
        self.dt = float(step_dt)
        self.after = int(after_steps)
        self.joint_names = list(joint_names)
        self.push_step = push_step
        self.td_steps = int(round(TOUCHDOWN_WINDOW_S / self.dt))
        self.sample_rel = int(round(SETTLE_SAMPLE_S / self.dt))
        self.lift_rel = int(round(LIFT_COUNT_FROM_S / self.dt))
        if self.sample_rel > self.after:
            raise ValueError("the measurement window must reach the +2 s sample")
        nan = float("nan")
        j = len(self.joint_names)
        self.phase = torch.full((self.n,), nan)
        self.failed_before = torch.zeros(self.n, dtype=torch.bool)
        self.failed = torch.zeros(self.n, dtype=torch.bool)
        self.fell = torch.zeros(self.n, dtype=torch.bool)
        self.fall_rel = torch.full((self.n,), nan)
        self.td_rel = torch.full((self.n,), -1, dtype=torch.long)
        self.peak_td = torch.full((self.n, j), nan)
        self.peak_2s = torch.full((self.n, j), nan)
        self.fore_aft = torch.full((self.n,), nan)
        self.lateral = torch.full((self.n,), nan)
        self.width = torch.full((self.n,), nan)
        self.com_dist = torch.full((self.n,), nan)
        self.com_fore_aft = torch.full((self.n,), nan)
        self.lifts = torch.zeros(self.n)
        self.prev_contact = None
        self.complete = torch.zeros(self.n, dtype=torch.bool)

    def last_step(self):
        return int(self.switch.max()) + self.after

    def update(self, step, phase, failed, contact, error, foot_delta, com_offset, reference_width):
        phase, failed, contact = phase.cpu(), failed.cpu(), contact.cpu()
        error, foot_delta, com_offset = error.cpu(), foot_delta.cpu(), com_offset.cpu()
        rel = step - self.switch
        live = ~self.failed
        at_switch = (rel == 0) & live & ~failed
        self.phase = torch.where(at_switch, phase.float(), self.phase)
        before = (rel < 0) & live & failed
        self.failed_before |= before
        during = (rel >= 0) & (rel <= self.after) & live & failed & ~self.failed_before
        self.fell |= during
        self.fall_rel = torch.where(during, rel.float() * self.dt, self.fall_rel)
        self.failed |= failed
        ok = (rel >= 0) & (rel <= self.after) & ~self.failed
        double = contact.all(dim=1)
        new_td = ok & double & (self.td_rel < 0)
        self.td_rel = torch.where(new_td, rel, self.td_rel)
        in_td = ok & (self.td_rel >= 0) & (rel <= self.td_rel + self.td_steps)
        self.peak_td = torch.where(in_td.unsqueeze(1), torch.fmax(self.peak_td, error.float()), self.peak_td)
        in_2s = ok & (rel <= self.sample_rel)
        self.peak_2s = torch.where(in_2s.unsqueeze(1), torch.fmax(self.peak_2s, error.float()), self.peak_2s)
        sample = ok & (rel == self.sample_rel)
        dx, dy = foot_delta[:, 0].float().abs(), foot_delta[:, 1].float().abs()
        self.fore_aft = torch.where(sample, dx, self.fore_aft)
        self.width = torch.where(sample, dy, self.width)
        self.lateral = torch.where(sample, (dy - float(reference_width)).abs(), self.lateral)
        self.com_dist = torch.where(sample, torch.linalg.norm(com_offset.float(), dim=1), self.com_dist)
        self.com_fore_aft = torch.where(sample, com_offset[:, 0].float(), self.com_fore_aft)
        if self.prev_contact is not None:
            lift = (self.prev_contact & ~contact).sum(dim=1).float()
            counting = ok & (rel > self.lift_rel)
            self.lifts = self.lifts + torch.where(counting, lift, torch.zeros_like(lift))
        self.prev_contact = contact.clone()
        self.complete |= ok & (rel == self.after)

    def push_due(self, step):
        if self.push_step is None:
            return torch.zeros(self.n, dtype=torch.bool)
        return (step - self.switch == self.push_step) & ~self.failed

    def summary(self):
        valid = ~self.failed_before & torch.isfinite(self.phase)
        n_valid = int(valid.sum())
        settled = valid & ~self.fell
        td_time = torch.where(self.td_rel >= 0, self.td_rel.float() * self.dt, torch.full_like(self.phase, float("nan")))
        td_time = torch.where(valid, td_time, torch.full_like(td_time, float("nan")))
        deg = 180.0 / math.pi
        peak_td = self.peak_td * deg
        peak_max = torch.where(torch.isfinite(peak_td).any(dim=1), torch.nan_to_num(peak_td, nan=-1.0).amax(dim=1),
                               torch.full_like(self.phase, float("nan")))
        peak_max = torch.where(valid, peak_max, torch.full_like(peak_max, float("nan")))
        peak_2s = self.peak_2s * deg
        peak_2s_max = torch.where(torch.isfinite(peak_2s).any(dim=1), torch.nan_to_num(peak_2s, nan=-1.0).amax(dim=1),
                                  torch.full_like(self.phase, float("nan")))
        peak_2s_max = torch.where(valid, peak_2s_max, torch.full_like(peak_2s_max, float("nan")))
        lifts = torch.where(settled & self.complete, self.lifts, torch.full_like(self.lifts, float("nan")))
        per_env = {
            "time_to_double_support_s": td_time,
            "peak_error_touchdown_deg_max": peak_max,
            "peak_error_2s_deg_max": peak_2s_max,
            "fore_aft_offset_2s_m": torch.where(valid, self.fore_aft, torch.full_like(self.fore_aft, float("nan"))),
            "lateral_offset_2s_m": torch.where(valid, self.lateral, torch.full_like(self.lateral, float("nan"))),
            "stance_width_2s_m": torch.where(valid, self.width, torch.full_like(self.width, float("nan"))),
            "com_support_dist_2s_m": torch.where(valid, self.com_dist, torch.full_like(self.com_dist, float("nan"))),
            "com_support_fore_aft_2s_m": torch.where(valid, self.com_fore_aft, torch.full_like(self.com_fore_aft, float("nan"))),
            "foot_lifts_after_1s": lifts,
        }
        out = {
            "envs": self.n,
            "valid_envs": n_valid,
            "failed_before_switch": int(self.failed_before.sum()),
            "falls": int((self.fell & valid).sum()),
            "fall_fraction": float((self.fell & valid).sum()) / max(n_valid, 1),
            "fall_time_s": _stats(torch.where(valid, self.fall_rel, torch.full_like(self.fall_rel, float("nan")))),
            "no_double_support": int((valid & (self.td_rel < 0)).sum()),
            "peak_error_touchdown_over_25deg": float(((peak_max > 25.0) & valid).sum()) / max(n_valid, 1),
            "peak_error_2s_over_25deg": float(((peak_2s_max > 25.0) & valid).sum()) / max(n_valid, 1),
            "re_step_fraction": float(((lifts > 0) & torch.isfinite(lifts)).sum()) / max(int(torch.isfinite(lifts).sum()), 1),
        }
        for key, values in per_env.items():
            out[key] = _stats(values)
        out["peak_error_touchdown_deg"] = {
            name: _stats(torch.where(valid, peak_td[:, i], torch.full_like(peak_max, float("nan"))))
            for i, name in enumerate(self.joint_names)
        }
        bins = torch.clamp((self.phase * PHASE_BINS).floor(), 0, PHASE_BINS - 1)
        by_phase = []
        for b in range(PHASE_BINS):
            member = valid & (bins == b)
            m = int(member.sum())
            row = {"bin": b, "phase_low": b / PHASE_BINS, "phase_high": (b + 1) / PHASE_BINS, "n": m,
                   "fall_fraction": float((self.fell & member).sum()) / max(m, 1)}
            for key, values in per_env.items():
                row[key] = _stats(torch.where(member, values, torch.full_like(values, float("nan"))))
            by_phase.append(row)
        out["by_phase"] = by_phase
        return out


def definition(warmup_steps, after_steps, period_steps, step_dt, push_velocity_range, reference_width):
    return {
        "cells": {name: {"command": list(cell["command"]), "push": cell["push"]} for name, cell in TRANSITION_CELLS.items()},
        "step_dt_s": step_dt,
        "warmup_s": warmup_steps * step_dt,
        "switch_step": "warmup_steps + U{0..%d} per env; the clock is episode time from a common reset, so the phase at "
                       "the switch is uniform over the %d clock steps" % (period_steps - 1, period_steps),
        "switch_time_zero": "the env.step whose observation first carries the zero command",
        "after_s": after_steps * step_dt,
        "push": {"at_s": PUSH_AT_S, "velocity_range": push_velocity_range,
                 "how": "push_standing_by_setting_velocity once per env, after the +2 s sample"},
        "phase_bins": PHASE_BINS,
        "phase": "gait_clock_phase at the switch step (offset included)",
        "metrics": {
            "falls": "non-time_out termination in [0, after_s]; envs failed before the switch are excluded",
            "time_to_double_support_s": "first step >= 0 with both feet in contact (> 1 N); 0 if already in double support",
            "peak_error_touchdown_deg": "max |wrapped target - position| per joint over [touchdown, touchdown + 0.5 s]",
            "peak_error_2s_deg_max": "max over joints of |target - position| over [0, 2 s]",
            "fore_aft_offset_2s_m": "|x_r - x_l| of the foot origins in the yaw frame at +2 s",
            "stance_width_2s_m": "|y_r - y_l| of the foot origins in the yaw frame at +2 s",
            "lateral_offset_2s_m": "|stance width - %.4f| (feet_width standing_width) at +2 s" % reference_width,
            "com_support_dist_2s_m": "planar distance of the whole-body COM from the centre of the 8 sole corners at +2 s",
            "com_support_fore_aft_2s_m": "signed yaw-frame x of that offset (COM ahead positive)",
            "foot_lifts_after_1s": "contact -> no-contact transitions of both feet in (1 s, after_s], envs that did not fall",
        },
    }
