import argparse
import json
import re
import sys
from pathlib import Path

TOL = 1.0e-6
NAME_PATTERN = re.compile(r"^(?P<recipe>.+?)s(?P<seed>\d+)_(?P<rest>.+)$")
EVAL_SEED_PATTERN = re.compile(r"^(?P<condition>.+)_s\d+$")
METRICS = ("wz_gain_vx0.1", "wz_offset_vx0.1", "wz_gain_vx0.3", "wz_offset_vx0.3", "wz_gain_vx0", "wz_offset_vx0",
           "straight_wz_drift", "straight_err_wz", "vx_gain", "vx_offset", "vx_at_vx0", "vy_gain", "vy_offset")
LABELS = ("wz g@.1", "wz o@.1", "wz g@.3", "wz o@.3", "wz g@0", "wz o@0", "drift", "err_wz", "vx g", "vx o", "vx@0",
          "vy g", "vy o")


def _eq(a, b):
    return abs(a - b) <= TOL


def fit(xs, ys):
    if len({round(x, 6) for x in xs}) < 2:
        return None, None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    gain = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return gain, my - gain * mx


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def eval_points(data):
    points = []
    for name, cell in data.items():
        if name.startswith("_") or not isinstance(cell, dict) or "command" not in cell:
            continue
        points.append({"cell": name, "command": tuple(float(v) for v in cell["command"]),
                       "got": (cell["got_vx"], cell["got_vy"], cell["got_wz"]), "err_wz": cell.get("err_wz")})
    return points


def player_points(data):
    points = []
    for segment in data.get("segments", ()):
        if not segment.get("samples") or "mean_vx" not in segment:
            continue
        points.append({"cell": None, "command": tuple(float(v) for v in segment["command"]),
                       "got": (segment["mean_vx"], segment["mean_vy"], segment["mean_wz"]), "err_wz": None})
    return points


def readout(points):
    def select(pred):
        return [p for p in points if pred(*p["command"])]

    out = {}
    for label, vx_level in (("vx0.1", 0.1), ("vx0.3", 0.3), ("vx0", 0.0)):
        turn = select(lambda vx, vy, wz: _eq(vx, vx_level) and _eq(vy, 0.0) and not _eq(wz, 0.0))
        gain, offset = fit([p["command"][2] for p in turn], [p["got"][2] for p in turn])
        out[f"wz_gain_{label}"], out[f"wz_offset_{label}"] = gain, offset
    straight = select(lambda vx, vy, wz: not _eq(vx, 0.0) and _eq(vy, 0.0) and _eq(wz, 0.0))
    out["straight_wz_drift"] = _mean(p["got"][2] for p in straight)
    errors = [p["err_wz"] for p in straight if p["err_wz"] is not None]
    out["straight_err_wz"] = _mean(errors) if errors and len(errors) == len(straight) else None
    out["vx_gain"], out["vx_offset"] = fit([p["command"][0] for p in straight], [p["got"][0] for p in straight])
    still = select(lambda vx, vy, wz: _eq(vx, 0.0) and max(abs(vy), abs(wz)) >= 0.2 - TOL)
    out["vx_at_vx0"] = _mean(p["got"][0] for p in still)
    strafe = select(lambda vx, vy, wz: _eq(vx, 0.0) and not _eq(vy, 0.0) and _eq(wz, 0.0))
    out["vy_gain"], out["vy_offset"] = fit([p["command"][1] for p in strafe], [p["got"][1] for p in strafe])
    return out


def classify(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    match = NAME_PATTERN.match(path.stem)
    recipe, seed, rest = (match["recipe"], match["seed"], match["rest"]) if match else (path.stem, None, "")
    if "segments" in data:
        scene = Path(str(data.get("scene", ""))).parent.name or "unknown"
        friction = "fric" if data.get("joint_friction") else "nofric"
        return {"kind": "mujoco", "recipe": recipe, "seed": seed, "condition": f"mujoco_{scene}_{friction}",
                "points": player_points(data)}
    seeded = EVAL_SEED_PATTERN.match(rest)
    condition = seeded["condition"] if seeded else rest
    return {"kind": "eval", "recipe": recipe, "seed": seed, "condition": condition, "points": eval_points(data)}


def build(paths):
    files = []
    groups = {}
    for path in paths:
        item = classify(path)
        files.append({"file": Path(path).name, "kind": item["kind"], "recipe": item["recipe"], "seed": item["seed"],
                      "condition": item["condition"], "cells": len(item["points"]), **readout(item["points"])})
        key = (item["recipe"], item["condition"], item["seed"])
        groups.setdefault(key, []).extend(item["points"])
    seeds = []
    for (recipe, condition, seed), points in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        seeds.append({"recipe": recipe, "condition": condition, "seed": seed, "points": len(points), **readout(points)})
    tags = []
    by_tag = {}
    for row in seeds:
        by_tag.setdefault((row["recipe"], row["condition"]), []).append(row)
    for (recipe, condition), rows in by_tag.items():
        entry = {"recipe": recipe, "condition": condition, "seeds": [r["seed"] for r in rows]}
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            entry[metric] = None if not values else {"mean": sum(values) / len(values), "min": min(values),
                                                     "max": max(values), "n": len(values)}
        tags.append(entry)
    return {"files": files, "seeds": seeds, "tags": tags}


def _fmt(value, signed):
    if value is None:
        return "-"
    return f"{value:+.4f}" if signed else f"{value:.3f}"


SIGNED = {m for m in METRICS if "gain" not in m and m != "straight_err_wz"}


def table(rows, name_of, value_of=lambda row, metric: row[metric]):
    header = "| " + " | ".join(("row",) + LABELS) + " |"
    lines = [header, "|" + "---|" * (len(LABELS) + 1)]
    for row in rows:
        cells = [_fmt(value_of(row, m), m in SIGNED) for m in METRICS]
        lines.append("| " + " | ".join([name_of(row)] + cells) + " |")
    return "\n".join(lines)


def render(result):
    out = ["## per file", table(result["files"], lambda r: r["file"]),
           "", "## per training seed (eval seeds and player commands pooled)",
           table(result["seeds"], lambda r: f"{r['recipe']}s{r['seed']} {r['condition']} ({r['points']} pts)"),
           "", "## per tag over training seeds: mean [min, max]"]
    lines = ["| row | metric | mean | min | max | n |", "|---|---|---|---|---|---|"]
    for tag in result["tags"]:
        name = f"{tag['recipe']} {tag['condition']} seeds {','.join(str(s) for s in tag['seeds'])}"
        for metric in METRICS:
            v = tag[metric]
            if v is None:
                continue
            signed = metric in SIGNED
            lines.append(f"| {name} | {metric} | {_fmt(v['mean'], signed)} | {_fmt(v['min'], signed)} | "
                         f"{_fmt(v['max'], signed)} | {v['n']} |")
    out.append("\n".join(lines))
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Signed tracking gain and offset from eval_commands and MuJoCo player JSONs.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="write the readout as JSON")
    args = parser.parse_args(argv)
    result = build(args.paths)
    print(render(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
