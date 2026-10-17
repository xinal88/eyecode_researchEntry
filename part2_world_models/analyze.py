"""Aggregate all Part 2 results into results/SUMMARY.md + figures for the report.

  python analyze.py
"""
import json
import re
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import RESULTS, episodes, judge_video  # noqa: E402

FIG = RESULTS / "figures"
# validated reference palette, light mode (slots 1-3), text in neutral inks
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    md = ["# Part 2 results summary\n"]

    # ---- tokenizer
    tok = load(RESULTS / "tokenizer/pusht_tokenizer_metrics.json")
    if tok:
        md += ["## Cosmos Tokenizer (CV4x8x8) reconstruction on PushT\n",
               "| input res | latent (C,T,H,W) | PSNR (dB) | SSIM |", "|---|---|---|---|"]
        for s, r in tok["results"].items():
            md.append(f"| {s} | {tuple(r['latent_shape'])} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
        md.append("")
    gz = load(RESULTS / "tokenizer/camera_tokenizer_metrics.json")
    if gz:
        md += ["Gazebo (ROSOrin depth-camera RGB) clip: " + ", ".join(
            f"{s}px PSNR {r['psnr']:.2f} dB SSIM {r['ssim']:.3f}" for s, r in gz["results"].items()), ""]

    # ---- IDM
    idm = load(RESULTS / "idm_target_val_metrics.json")
    if idm:
        md += ["## Inverse dynamics model (trained on 25 target demos, evaluated on 20 held-out demos)\n",
               f"state MAE {idm['state_mae_px']:.2f} px, action MAE {idm['action_mae_px']:.2f} px, "
               f"action L2 {idm['action_l2_px']:.2f} px (PushT workspace = 512 px)\n"]

    # ---- world models
    wms = {}
    for d in sorted((RESULTS / "world_model").glob("*")):
        log = load(d / "train_log.json")
        if log:
            v = [l for l in log if "val_psnr" in l]
            wms[d.name] = v
    if wms:
        md += ["## World model validation (first-frame conditioned, 33 frames, held-out demos)\n",
               "| model | steps | val PSNR (dB) | val SSIM |", "|---|---|---|---|"]
        for k, v in wms.items():
            md.append(f"| {k} | {v[-1]['step']} | {v[-1]['val_psnr']:.2f} | {v[-1]['val_ssim']:.4f} |")
        md.append("")

    # ---- judge: real vs dreams
    real = [judge_video(e["frames"]) for e in episodes("val")]
    rows = {"real demos (val)": {k: float(np.mean([r[k] for r in real])) for k in real[0]}}
    for d in sorted((RESULTS / "dreams").glob("*")):
        s = load(d / "judge_scores.json")
        if s:
            rows[d.name] = {k: float(np.mean([r[k] for r in s])) for k in s[0]}
            rows[d.name]["n"] = len(s)
    thr = float(np.percentile([r["coverage_end"] for r in real], 10))
    keys = ["coverage_start", "coverage_end", "shape_err", "fragmented", "agent_ok", "telekinesis", "teleport", "pa_score"]
    md += ["## Automatic video judge (DreamGen-Bench analogue)\n",
           f"Instruction-following proxy: final T-goal pixel coverage (real demos reach >= {thr:.2f} in 90% of cases). "
           "Physics-alignment proxy: share of frames without rigidity / fragmentation / agent / contact violations.\n",
           "| videos | " + " | ".join(keys) + " | IF success |", "|---" * (len(keys) + 2) + "|"]
    for name, r in rows.items():
        sc = (real if name.startswith("real") else load(RESULTS / "dreams" / name / "judge_scores.json"))
        ifs = np.mean([x["coverage_end"] >= thr for x in sc])
        md.append(f"| {name} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + f" | {ifs:.2f} |")
    md.append("")

    # ---- policies: per run, then grouped over seeds (run name without its _sN suffix)
    pols = {}
    for d in sorted((RESULTS / "policy").glob("*")):
        e = load(d / "eval.json")
        if e:
            pols[d.name] = e
    if pols:
        md += ["## Diffusion Policy, closed-loop PushT evaluation (50 episodes per run, seeds 100000-100049)\n",
               "| run | real demos | neural traj. | success rate [95% CI] | mean max coverage |",
               "|---|---|---|---|---|"]
        for k, e in pols.items():
            n = e["n_episodes"]
            s_ = int(round(e["success_rate"] * n))
            lo, hi = wilson(s_, n)
            md.append(f"| {k} | {e['n_real']} | {e['n_neural']} | {e['success_rate']:.2f} [{lo:.2f}, {hi:.2f}] | "
                      f"{e['mean_max_coverage']:.3f} |")
        md.append("")
        groups = {}
        for k, e in pols.items():
            groups.setdefault(re.sub(r"_s\d+$", "", k), []).append(e)
        order = ["real25", "real25_dream100", "real25_dream300", "real25_dream1000", "real25_dream300pa",
                 "real25_scratchdream300", "dream300_only", "real186_upper"]
        names = [g for g in order if g in groups] + [g for g in groups if g not in order]
        rows = []
        for g in names:
            es = groups[g]
            cov = np.array([e["mean_max_coverage"] for e in es])
            k_ = sum(sum(e["per_episode_success"]) for e in es)
            n_ = sum(e["n_episodes"] for e in es)
            lo, hi = wilson(k_, n_)
            rows.append({"config": g, "seeds": len(es), "n_real": es[0]["n_real"], "n_neural": es[0]["n_neural"],
                         "coverage_mean": float(cov.mean()), "coverage_std": float(cov.std(ddof=1)) if len(es) > 1 else 0.0,
                         "success": k_ / n_, "success_lo": lo, "success_hi": hi, "episodes": n_})
        md += ["### Grouped over training seeds\n",
               "| config | seeds | real | neural | mean max coverage (mean ± sd over seeds) | success, pooled episodes [95% CI] |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['config']} | {r['seeds']} | {r['n_real']} | {r['n_neural']} | "
                      f"{r['coverage_mean']:.3f} ± {r['coverage_std']:.3f} | "
                      f"{r['success']:.3f} [{r['success_lo']:.2f}, {r['success_hi']:.2f}] (n={r['episodes']}) |")
        md.append("")
        json.dump(rows, open(RESULTS / "policy_grouped.json", "w"), indent=2)
        # figure: grouped coverage, seeds as dots; the DreamGen config is the accent
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.3))
        y = np.arange(len(rows))
        for i, r in enumerate(rows):
            c = C1 if r["config"] == "real25_dream300" else "#9a9893"
            ax.barh(i, r["coverage_mean"], color=c, height=0.6)
            pts = [e["mean_max_coverage"] for e in groups[r["config"]]]
            ax.scatter(pts, [i] * len(pts), s=12, color=INK, zorder=3)
            ax.text(r["coverage_mean"] + 0.015, i - 0.22, f"{r['coverage_mean']:.2f}", color=INK, fontsize=8)
        ax.axvline(rows[0]["coverage_mean"], color=INK2, lw=1, ls="--")
        ax.set_yticks(y, [f"{r['config']} ({r['seeds']}x)" for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("mean max goal coverage over 50 episodes (bar = mean over seeds, dots = seeds)")
        fig.tight_layout()
        fig.savefig(FIG / "policy_results.png", dpi=180)
        plt.close(fig)

    # ---- WM training curves
    if wms:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for (k, v), c in zip(wms.items(), [C1, C2, C3]):
            ax.plot([l["step"] for l in v], [l["val_psnr"] for l in v], color=c, lw=2, marker="o", ms=4, label=k)
        ax.set_xlabel("training step")
        ax.set_ylabel("val PSNR of generated future (dB)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "wm_val_psnr.png", dpi=180)
        plt.close(fig)

    (RESULTS / "SUMMARY.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
