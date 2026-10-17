"""Shared data utilities + an automatic PushT "video judge" (our DreamGen-Bench analogue)."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "part2_world_models/results"
CKPT = ROOT / "checkpoints"

_CACHE = {}


def load_pusht():
    if "pusht" not in _CACHE:
        _CACHE["pusht"] = dict(np.load(DATA / "pusht_96.npz"))
    return _CACHE["pusht"]


def splits():
    return json.load(open(DATA / "splits.json"))


def episodes(split):
    """List of dicts(frames, state, action) for the episodes in a split."""
    d = load_pusht()
    out = []
    for e in splits()[split]:
        s, t = d["ep_start"][e], d["ep_end"][e]
        out.append({"frames": d["frames"][s:t], "state": d["state"][s:t], "action": d["action"][s:t], "ep": int(e)})
    return out


def norm_xy(x):  # PushT pixel coords [0, 512] -> [-1, 1]
    return (np.asarray(x, np.float32) - 256.0) / 256.0


def denorm_xy(x):
    return np.asarray(x, np.float32) * 256.0 + 256.0


# ------------------------------------------------------------------ automatic video judge
# Colours measured on the (AV1-decoded) dataset: goal (140,232,140), T block (138,161,179),
# agent (81,123,253), background ~248 with grid lines ~232. Draw order goal < T < agent.
T_RGB = np.array([138, 161, 179], np.float32)
GOAL_RGB = np.array([140, 232, 140], np.float32)
AGENT_RGB = np.array([81, 123, 253], np.float32)
AGENT_TOL = 70.0  # generated videos render the small agent blurrier/fainter than the AV1 dataset


def color_mask(frames, rgb, tol=40.0):
    return np.linalg.norm(frames.astype(np.float32) - rgb, axis=-1) < tol


def goal_mask():
    """Goal region = union of goal-coloured pixels and T pixels lying inside the goal's hull,
    estimated once from the real dataset (the goal pose is fixed in PushT)."""
    if "goal" not in _CACHE:
        f = load_pusht()["frames"][::25]
        g = color_mask(f, GOAL_RGB).mean(0) > 0.02
        _CACHE["goal"] = ndimage.binary_closing(g, iterations=2)
    return _CACHE["goal"]


def ref_areas():
    if "areas" not in _CACHE:
        f = load_pusht()["frames"][::50]
        _CACHE["areas"] = (float(np.median(color_mask(f, T_RGB).sum((1, 2)))),
                           float(np.median(color_mask(f, AGENT_RGB, AGENT_TOL).sum((1, 2)))))
    return _CACHE["areas"]


def _largest_frac(m):
    lab, n = ndimage.label(m)
    if n == 0:
        return 0.0
    return np.bincount(lab.ravel())[1:].max() / m.sum()


def judge_video(frames, contact_px=4.0, move_px=1.5):
    """Score one video (T,96,96,3) uint8. Returns dict of per-video metrics:
      coverage_*   : |T ∩ goal| / |goal| at start / end / max   (instruction following proxy)
      shape_err    : mean |area_T / ref - 1|                     (object permanence / rigidity)
      fragmented   : fraction of frames where the T mask breaks apart
      agent_ok     : fraction of frames with an agent blob of plausible size
      telekinesis  : fraction of frame pairs where the T moves (centroid > move_px) while the
                     agent is farther than contact_px from it -> violates contact physics
      teleport     : fraction of frame pairs where the T jumps > 4 px (real demos: max 3.2 px)
      pa_score     : 1 - mean(violations)                        (physics alignment proxy)
    """
    g = goal_mask()
    ta, aa = ref_areas()
    tm = color_mask(frames, T_RGB)
    am = color_mask(frames, AGENT_RGB, AGENT_TOL)
    cov = (tm & g).sum((1, 2)) / g.sum()
    area = tm.sum((1, 2))
    shape_err = np.abs(area / ta - 1)
    # the agent is drawn on top of the T and can split it, so "fragmented" means the largest
    # connected piece holds < 70% of the T pixels rather than "more than one component"
    frag = np.array([_largest_frac(m) < 0.7 for m in tm])
    a_area = am.sum((1, 2))
    agent_ok = (a_area >= 0.4 * aa) & (a_area <= 1.8 * aa)
    cents = np.array([ndimage.center_of_mass(m) if m.any() else (np.nan, np.nan) for m in tm])
    tele = []
    for t in range(len(frames) - 1):
        moved = np.linalg.norm(cents[t + 1] - cents[t]) > move_px
        if not am[t].any() or not tm[t].any():
            tele.append(False)
            continue
        dist = ndimage.distance_transform_edt(~tm[t])[am[t]].min()
        tele.append(bool(moved and dist > contact_px))
    tele = np.array(tele, float) if tele else np.zeros(1)
    # teleport: T centroid jumps farther in one frame than ever observed in real demos (max 3.2 px)
    jump = np.linalg.norm(np.diff(cents, axis=0), axis=1)
    teleport = np.nan_to_num(jump, nan=0.0) > 4.0
    viol = np.stack([shape_err[:-1] > 0.25, frag[:-1], ~agent_ok[:-1], tele.astype(bool) | teleport], 0)
    return {
        "coverage_start": float(cov[0]), "coverage_end": float(cov[-1]), "coverage_max": float(cov.max()),
        "shape_err": float(shape_err.mean()), "fragmented": float(frag.mean()),
        "agent_ok": float(agent_ok.mean()), "telekinesis": float(tele.mean()), "teleport": float(teleport.mean()),
        "pa_score": float(1 - viol.any(0).mean()),
    }
