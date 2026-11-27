import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT / "scripts/rsl_rl/eval/tracking_readout.py"

WA2S43_LEGS = {
    "fwd_slow": ((0.1, 0.0, 0.0), 0.1210290938615799, 0.0006250623846426606, 0.013800117187201977, 0.08341784030199051),
    "fwd_max": ((0.3, 0.0, 0.0), 0.3308803141117096, -0.0013230754993855953, 0.01375332660973072, 0.07812374085187912),
    "fwd_goal": ((0.5, 0.0, 0.0), 0.532939612865448, -0.0043610758148133755, 0.011103094555437565, 0.08149203658103943),
    "stop": ((0.0, 0.0, 0.0), 0.0029698642902076244, -0.0002515254891477525, 0.000874139426741749, 0.07331238687038422),
    "back": ((-0.2, 0.0, 0.0), -0.1696283370256424, 0.0017142642755061388, 0.017568787559866905, 0.08581501245498657),
    "turn_l": ((0.1, 0.0, 0.2), 0.12508824467658997, -0.0005351334111765027, 0.20406627655029297, 0.0905895009636879),
    "turn_r": ((0.1, 0.0, -0.2), 0.1237347200512886, 0.0016278864350169897, -0.17352792620658875, 0.0909038707613945),
    "turn_still": ((0.0, 0.0, 0.2), 0.026277896016836166, 0.0035626711323857307, 0.2058410793542862, 0.09325889497995377),
    "strafe_l": ((0.0, 0.2, 0.0), 0.027916282415390015, 0.19435745477676392, 0.019989091902971268, 0.08597046881914139),
    "strafe_r": ((0.0, -0.2, 0.0), 0.024309657514095306, -0.1936684101819992, 0.005380780436098576, 0.08480086922645569),
    "diag": ((0.2, 0.1, 0.2), 0.23175878822803497, 0.09021720290184021, 0.2043285369873047, 0.08962542563676834),
    "strafe_l_slow": ((0.0, 0.1, 0.0), 0.025760941207408905, 0.09340886026620865, 0.010504207573831081, 0.08387448638677597),
    "strafe_r_slow": ((0.0, -0.1, 0.0), 0.025212464854121208, -0.09268511086702347, 0.016645925119519234, 0.08500730246305466),
    "diag_l": ((0.2, 0.1, 0.0), 0.22996826469898224, 0.09737012535333633, 0.020550936460494995, 0.08431028574705124),
    "diag_r": ((0.2, -0.1, 0.0), 0.2284269630908966, -0.09712187200784683, 0.0021066004410386086, 0.08195225894451141),
    "turn_walk_l": ((0.3, 0.0, 0.2), 0.3330767750740051, -0.0034804977476596832, 0.18958447873592377, 0.0836355984210968),
    "turn_walk_r": ((0.3, 0.0, -0.2), 0.3304964303970337, 0.0005074597429484129, -0.1601044088602066, 0.08842934668064117),
}


def load():
    spec = importlib.util.spec_from_file_location("tracking_readout", READOUT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_json(cells):
    out = {"_meta": {"num_envs": 1}}
    for name, (command, vx, vy, wz, err_wz) in cells.items():
        out[name] = {"command": list(command), "got_vx": vx, "got_vy": vy, "got_wz": wz, "err_wz": err_wz}
    return out


def synthetic(gain, offset):
    cells = {}
    for name, (command, *_rest) in WA2S43_LEGS.items():
        vx, vy, wz = command
        cells[name] = (command, gain * vx + offset, gain * vy + offset, gain * wz + offset, 0.05)
    return cells


def player_json(segments, scene="full", friction=True):
    return {"scene": "/x/robonex-description/ver3/mujoco/robot/%s/scene.xml" % scene,
            "joint_friction": {"a": 0.14} if friction else None,
            "segments": [{"command": list(c), "samples": 10, "mean_vx": vx, "mean_vy": vy, "mean_wz": wz}
                         for c, (vx, vy, wz) in segments]}


class TrackingReadoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.write_text(json.dumps(data))
        return path

    def test_synthetic_gain_and_offset_exact(self):
        r = self.m.readout(self.m.eval_points(eval_json(synthetic(0.8, 0.02))))
        for metric in ("wz_gain_vx0.1", "wz_gain_vx0.3", "vx_gain", "vy_gain"):
            self.assertAlmostEqual(r[metric], 0.8, places=12, msg=metric)
        for metric in ("wz_offset_vx0.1", "wz_offset_vx0.3", "vx_offset", "vy_offset", "straight_wz_drift", "vx_at_vx0"):
            self.assertAlmostEqual(r[metric], 0.02, places=12, msg=metric)
        self.assertAlmostEqual(r["straight_err_wz"], 0.05, places=12)
        self.assertIsNone(r["wz_gain_vx0"])

    def test_wa2s43_legs_numbers(self):
        r = self.m.readout(self.m.eval_points(eval_json(WA2S43_LEGS)))
        expected = {"wz_gain_vx0.1": 0.944, "wz_offset_vx0.1": 0.0153, "wz_gain_vx0.3": 0.874, "wz_offset_vx0.3": 0.0147,
                    "straight_wz_drift": 0.0141, "straight_err_wz": 0.0822, "vx_gain": 1.006, "vx_offset": 0.0278,
                    "vx_at_vx0": 0.0262, "vy_gain": 0.962, "vy_offset": 0.0004}
        for metric, value in expected.items():
            self.assertAlmostEqual(r[metric], value, delta=6e-4 if "gain" in metric else 1e-4, msg=metric)
        self.assertAlmostEqual(r["wz_gain_vx0.1"], (0.20406627655029297 + 0.17352792620658875) / 0.4, places=12)
        self.assertAlmostEqual(r["wz_offset_vx0.1"], (0.20406627655029297 - 0.17352792620658875) / 2, places=12)

    def test_file_names_group_by_recipe_condition_and_training_seed(self):
        paths = [self.write("W9s43_pd1_s0.json", eval_json(synthetic(0.8, 0.02))),
                 self.write("W9s43_pd1_s1.json", eval_json(synthetic(0.8, 0.04))),
                 self.write("W9s44_pd1_s0.json", eval_json(synthetic(0.6, 0.0))),
                 self.write("W9s44_nopush_s0.json", eval_json(synthetic(1.0, 0.0)))]
        result = self.m.build(paths)
        seeds = {(r["recipe"], r["condition"], r["seed"]): r for r in result["seeds"]}
        self.assertEqual(set(seeds), {("W9", "pd1", "43"), ("W9", "pd1", "44"), ("W9", "nopush", "44")})
        self.assertAlmostEqual(seeds[("W9", "pd1", "43")]["vx_offset"], 0.03, places=12)
        self.assertEqual(seeds[("W9", "pd1", "43")]["points"], 34)
        tag = next(t for t in result["tags"] if t["condition"] == "pd1")
        self.assertEqual(tag["seeds"], ["43", "44"])
        self.assertAlmostEqual(tag["wz_gain_vx0.1"]["mean"], 0.7, places=12)
        self.assertAlmostEqual(tag["wz_gain_vx0.1"]["min"], 0.6, places=12)
        self.assertAlmostEqual(tag["wz_gain_vx0.1"]["max"], 0.8, places=12)

    def test_player_jsons_pool_commands_per_scene(self):
        g, o = 0.9, 0.01
        commands = [(0.1, 0, 0), (0.3, 0, 0), (-0.2, 0, 0), (0, 0, 0.1), (0, 0, 0.2), (0, 0, -0.2), (0, 0.1, 0), (0, -0.1, 0)]
        paths = [self.write("WA2s43_%d.json" % i, player_json([(c, tuple(g * v + o for v in c))])) for i, c in enumerate(commands)]
        paths.append(self.write("WA2s43_legs.json", player_json([((0.1, 0, 0), (5.0, 5.0, 5.0))], scene="legs")))
        result = self.m.build(paths)
        full = next(r for r in result["seeds"] if r["condition"] == "mujoco_full_fric")
        self.assertEqual(full["points"], 8)
        for metric in ("wz_gain_vx0", "vx_gain", "vy_gain"):
            self.assertAlmostEqual(full[metric], g, places=12, msg=metric)
        for metric in ("wz_offset_vx0", "vx_offset", "vy_offset", "straight_wz_drift", "vx_at_vx0"):
            self.assertAlmostEqual(full[metric], o, places=12, msg=metric)
        self.assertIsNone(full["wz_gain_vx0.1"])
        self.assertIsNone(full["straight_err_wz"])
        self.assertTrue(any(r["condition"] == "mujoco_legs_fric" for r in result["seeds"]))

    def test_player_segment_without_samples_is_skipped(self):
        data = player_json([((0.1, 0, 0), (0.1, 0, 0))])
        data["segments"].append({"command": [0.3, 0, 0], "samples": 0})
        self.assertEqual(len(self.m.player_points(data)), 1)

    def test_main_prints_and_writes_json(self):
        path = self.write("WA2s43_on_legs_std_s0.json", eval_json(WA2S43_LEGS))
        out = self.dir / "out/readout.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(self.m.main([str(path), "--json", str(out)]), 0)
        self.assertIn("| WA2s43_on_legs_std_s0.json | 0.944 | +0.0153 | 0.874 |", buffer.getvalue())
        saved = json.loads(out.read_text())
        self.assertEqual(saved["seeds"][0]["condition"], "on_legs_std")


if __name__ == "__main__":
    unittest.main()
