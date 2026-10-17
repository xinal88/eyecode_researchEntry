#!/usr/bin/env python3
"""Quantify the SLAM map against Hiwonder's reference map of the same world (map_01).

Both maps are in the world frame (SLAM starts at the spawn pose 0,0,0 and map_01 was built the same
way), so they are compared cell by cell after resampling onto one 0.05 m grid:
  * occupied precision / recall within a tolerance (default 2 cells = 0.10 m)
  * known (free + occupied) area and free area in m^2, and the overlap of the free space (IoU)
Writes results/map_metrics.json and results/map_comparison.png.

  python3 scripts/map_metrics.py maps/rosorin_house.yaml ws/src/simulations/robot_gazebo/maps/map_01.yaml
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parents[1]


def load(yaml_path):
    meta = {}
    for line in Path(yaml_path).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    img = np.array(Image.open(Path(yaml_path).parent / meta["image"]).convert("L"))
    res = float(meta["resolution"])
    ox, oy = [float(x) for x in meta["origin"].strip("[]").split(",")[:2]]
    occ = img < 100            # trinary maps: 0 occupied, 254 free, 205 unknown
    free = img > 230
    return occ, free, res, ox, oy


def to_world_grid(mask, res, ox, oy, x0, y0, W, H):
    """Resample a map mask onto a common grid whose lower-left corner is (x0, y0)."""
    out = np.zeros((H, W), bool)
    h = mask.shape[0]
    rr, cc = np.nonzero(mask)
    xs = ox + (cc + 0.5) * res
    ys = oy + (h - rr - 0.5) * res
    gc = ((xs - x0) / 0.05).astype(int)
    gr = H - 1 - ((ys - y0) / 0.05).astype(int)
    ok = (gc >= 0) & (gc < W) & (gr >= 0) & (gr < H)
    out[gr[ok], gc[ok]] = True
    return out


def main():
    ours_yaml = sys.argv[1] if len(sys.argv) > 1 else HERE / "maps/rosorin_house.yaml"
    ref_yaml = sys.argv[2] if len(sys.argv) > 2 else HERE / "ws/src/simulations/robot_gazebo/maps/map_01.yaml"
    tol = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    A, B = load(ours_yaml), load(ref_yaml)
    xs = [A[3], B[3], A[3] + A[0].shape[1] * A[2], B[3] + B[0].shape[1] * B[2]]
    ys = [A[4], B[4], A[4] + A[0].shape[0] * A[2], B[4] + B[0].shape[0] * B[2]]
    x0, y0 = min(xs), min(ys)
    W, H = int((max(xs) - x0) / 0.05) + 1, int((max(ys) - y0) / 0.05) + 1
    occ_a, free_a = (to_world_grid(m, A[2], A[3], A[4], x0, y0, W, H) for m in A[:2])
    occ_b, free_b = (to_world_grid(m, B[2], B[3], B[4], x0, y0, W, H) for m in B[:2])
    # compare only inside the reference map's footprint (the house)
    house = ndimage.binary_fill_holes(ndimage.binary_dilation(occ_b | free_b, iterations=2))
    da = ndimage.distance_transform_edt(~occ_a)
    db = ndimage.distance_transform_edt(~occ_b)
    oa, ob = occ_a & house, occ_b & house
    precision = float((db[oa] <= tol).mean()) if oa.any() else 0.0
    recall = float((da[ob] <= tol).mean()) if ob.any() else 0.0
    fa, fb = free_a & house, free_b & house
    m = {
        "occupied_precision@%dcm" % (tol * 5): round(precision, 3),
        "occupied_recall@%dcm" % (tol * 5): round(recall, 3),
        "free_space_iou": round(float((fa & fb).sum() / max((fa | fb).sum(), 1)), 3),
        "known_area_m2_ours": round(float(((occ_a | free_a) & house).sum() * 0.0025), 2),
        "known_area_m2_reference": round(float(((occ_b | free_b) & house).sum() * 0.0025), 2),
        "free_area_m2_ours": round(float(fa.sum() * 0.0025), 2),
        "free_area_m2_reference": round(float(fb.sum() * 0.0025), 2),
    }
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    json.dump(m, open(out / "map_metrics.json", "w"), indent=2)
    # overlay: reference walls grey, ours dark blue, agreement black, free space light
    img = np.full((H, W, 3), 255, np.uint8)
    img[free_a | free_b] = (236, 240, 245)
    img[occ_b] = (170, 170, 170)
    img[occ_a] = (42, 120, 214)
    img[occ_a & (db <= tol)] = (20, 20, 20)
    Image.fromarray(img).resize((W * 3, H * 3), Image.NEAREST).save(out / "map_comparison.png")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
