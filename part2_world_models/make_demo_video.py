"""Cut the Part 2 results into one ~80 s demo video: results/part2_demo.mp4.

Real demo vs. dreams, a grid of dreams with their judge scores, failure cases, world-model validation,
closed-loop policy rollouts and the final numbers. Only uses files already produced by the pipeline
(results/, data/pusht_96.npz); needs ffmpeg.

  python part2_world_models/make_demo_video.py
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from common import DATA, RESULTS, splits

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
W, H, FPS, BG = 1280, 720, 30, "0x101820"
REAL_EP = 80  # val episode, 117 frames = the length of a dream


class Video:
    def __init__(self, tmp):
        self.tmp, self.parts, self.n_txt = Path(tmp), [], 0

    def text(self, s, x, y, size=28, bold=True, color="white"):
        self.n_txt += 1
        f = self.tmp / f"t{self.n_txt}.txt"
        f.write_text(s)
        # x: pixels, "c" = centred on the frame, or (x0, width) = centred over a panel
        xs = "(w-text_w)/2" if x == "c" else f"{x[0]}+({x[1]}-text_w)/2" if isinstance(x, tuple) else str(x)
        return (f"drawtext=fontfile={FONT if bold else FONT_R}:textfile={f}:expansion=none:fontsize={size}:fontcolor={color}"
                f":x={xs}:y={y}")

    def scene(self, dur, media=(), texts=()):
        """media = [(path, x, y, w, h)] (videos are padded with their last frame, images held);
        texts = drawtext filters (see text())."""
        args = ["-f", "lavfi", "-i", f"color=c={BG}:s={W}x{H}:d={dur}:r={FPS}"]
        graph, last = [], "0:v"
        for i, (path, x, y, w, h) in enumerate(media, 1):
            if Path(path).suffix == ".png":
                args += ["-loop", "1", "-t", str(dur), "-i", str(path)]
                graph.append(f"[{i}:v]scale={w}:{h}:flags=lanczos[m{i}]")
            else:
                args += ["-i", str(path)]
                graph.append(f"[{i}:v]scale={w}:{h}:flags=neighbor,fps={FPS},"
                             f"tpad=stop_mode=clone:stop_duration={dur}[m{i}]")
            graph.append(f"[{last}][m{i}]overlay={x}:{y}:eof_action=repeat[o{i}]")
            last = f"o{i}"
        graph.append(f"[{last}]" + (",".join(texts) if texts else "null") + "[out]")
        out = self.tmp / f"s{len(self.parts):02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, "-filter_complex", ";".join(graph),
                        "-map", "[out]", "-t", str(dur), "-an", "-c:v", "libx264", "-threads", "4", "-preset",
                        "medium", "-crf", "22", "-pix_fmt", "yuv420p", "-r", str(FPS), str(out)], check=True)
        self.parts.append(out)

    def concat(self, out):
        lst = self.tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in self.parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", str(out)], check=True)


def real_demo_mp4(path, ep):
    d = np.load(DATA / "pusht_96.npz")
    fr = d["frames"][d["ep_start"][ep]:d["ep_end"][ep]]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "96x96",
                    "-r", "10", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)],
                   input=fr.tobytes(), check=True)


def main():
    R = RESULTS
    judge = {n: json.load(open(R / "dreams" / n / "judge_scores.json")) for n in ("dreams_wm_ft", "dreams_wm_scratch")}
    grouped = {g["config"]: g for g in json.load(open(R / "policy_grouped.json"))}
    assert REAL_EP in splits()["val"]
    with tempfile.TemporaryDirectory() as tmp:
        v = Video(tmp)
        t = v.text

        # 0. title
        v.scene(5, texts=[t("Part 2 - World Models: DreamGen x Cosmos", "c", 250, 46),
                          t("small-scale reproduction on PushT (lerobot/pusht, 206 human demos)", "c", 330, 28, False),
                          t("official Cosmos Tokenizer CV4x8x8  +  Cosmos-style Video2World latent DiT (57.7M)", "c", 380,
                            24, False, "0xb0b8c0")])
        # 1. pipeline
        steps = ["1. World model: pre-train on 161 demos -> fine-tune on the 25 target demos",
                 "2. Dreams: roll it out from NEW initial states (fresh env resets), 117 frames each",
                 "3. Pseudo-actions: inverse dynamics model trained on the 25 target demos",
                 "4. Policy: Diffusion Policy co-trained 1:1 on 25 real demos + N dreams",
                 "Eval: closed loop in gym-pusht, 50 fixed seeds; success = T covers >= 95% of goal"]
        v.scene(8, texts=[t("DreamGen pipeline, scaled down", "c", 120, 40)] +
                [t(s, 90, 230 + 70 * i, 26, False) for i, s in enumerate(steps)])

        # 2. real demo vs dreams
        real = Path(tmp) / "real.mp4"
        real_demo_mp4(real, REAL_EP)
        ft, sc = 4, 1
        jf, js = judge["dreams_wm_ft"][ft], judge["dreams_wm_scratch"][sc]
        S, Y, xs = 384, 190, [32, 448, 864]
        v.scene(11.7, media=[(real, xs[0], Y, S, S),
                             (R / f"dreams/dreams_wm_ft/dream_{ft}.mp4", xs[1], Y, S, S),
                             (R / f"dreams/dreams_wm_scratch/dream_{sc}.mp4", xs[2], Y, S, S)],
                texts=[t("Real demo vs. dreams generated from a new initial frame", "c", 40, 36),
                       t("real human demo (val)", (xs[0], S), 140, 23),
                       t("dream: pre-trained + fine-tuned WM", (xs[1], S), 140, 21),
                       t("dream: WM trained from scratch", (xs[2], S), 140, 21),
                       t("judge PA 0.86 (real demos, avg)", (xs[0], S), Y + S + 20, 21, False),
                       t(f"judge PA {jf['pa_score']:.2f}, goal coverage {jf['coverage_end']:.2f}", (xs[1], S), Y + S + 20, 21, False),
                       t(f"judge PA {js['pa_score']:.2f}, goal coverage {js['coverage_end']:.2f}", (xs[2], S), Y + S + 20, 21, False),
                       t("only the first frame is given; every later frame is imagined by the world model", "c", 650, 22,
                         False, "0xb0b8c0")])

        # 3. grid of dreams with judge scores
        S, gx, gy = 260, [100, 510, 920], [120, 430]
        media, texts = [], [t("Six dreams (fine-tuned world model) and their automatic judge scores", "c", 30, 29)]
        for i in range(6):
            x, y = gx[i % 3], gy[i // 3]
            j = judge["dreams_wm_ft"][i]
            media.append((R / f"dreams/dreams_wm_ft/dream_{i}.mp4", x, y, S, S))
            texts.append(t(f"coverage {j['coverage_end']:.2f}  PA {j['pa_score']:.2f}", (x, S), y + S + 8, 20, False))
        texts.append(t("instruction following ok (88% pass, as 90% of real demos) - physics weak (PA 0.48 vs 0.86 real)",
                       "c", 85, 22, False, "0xffc070"))
        v.scene(11.7, media, texts)

        # 4. failure cases
        v.scene(7, media=[(R / "misc/wm_pre_dream_failures.png", 20, 200, 1240, 372)],
                texts=[t("Failure cases of the pre-trained-only WM (time ->)", "c", 60, 32),
                       t("the block bends, splits and fades, the agent vanishes: object impermanence, non-rigid deformation",
                         "c", 120, 22, False, "0xb0b8c0")])
        # 5. world-model validation
        v.scene(6, media=[(R / "world_model/wm_ft/val_step3000.png", 60, 90, 560, 560)],
                texts=[t("World model, held-out demos", 670, 160, 32),
                       t("odd rows: real future", 670, 240, 24, False),
                       t("even rows: generated from frame 1", 670, 282, 24, False),
                       t("val PSNR 26.7 dB, SSIM 0.93", 670, 350, 24, False),
                       t("pre-training helps:", 670, 400, 24, False, "0xffc070"),
                       t("26.7 dB vs 25.7 dB from scratch", 670, 438, 24, False, "0xffc070")])

        # 6. policy rollouts
        S, xs, Y = 290, [24, 340, 656, 972], 200
        roll = [("real25/eval_ep0_fail.mp4", "25 real demos", "fail"),
                ("real25_s5/eval_ep2_success.mp4", "25 real demos", "success"),
                ("real25_dream300_s6/eval_ep2_success.mp4", "25 real + 300 dreams", "success"),
                ("real186_upper/eval_ep2_success.mp4", "186 real (upper bound)", "success")]
        v.scene(15, media=[(R / "policy" / p, x, Y, S, S) for (p, _, _), x in zip(roll, xs)],
                texts=[t("Closed-loop Diffusion Policy rollouts in gym-pusht", "c", 50, 36)] +
                [t(lab, (x, S), Y - 45, 22) for (_, lab, _), x in zip(roll, xs)] +
                [t(res, (x, S), Y + S + 15, 24, True, "0x7fe08a" if res == "success" else "0xff7070")
                 for (_, _, res), x in zip(roll, xs)] +
                [t("short episodes end at success; the frame is then held", "c", 620, 20, False, "0xb0b8c0")])

        # 7. results
        a, b = grouped["real25"], grouped["real25_dream300"]
        v.scene(10, media=[(R / "figures/policy_results.png", 20, 120, 760, 472)],
                texts=[t("Results (8 seeds x 50 episodes per arm)", "c", 40, 36),
                       t("goal coverage", 810, 170, 30),
                       t(f"{a['coverage_mean']:.3f} -> {b['coverage_mean']:.3f}", 810, 215, 30, True, "0x7fe08a"),
                       t("Mann-Whitney p = 0.021", 810, 260, 24, False),
                       t("success rate", 810, 340, 30),
                       t(f"{a['success']:.1%} -> {b['success']:.1%}", 810, 385, 30, True, "0xffc070"),
                       t("Fisher p = 0.28 (not significant)", 810, 430, 24, False),
                       t("dreams only: 0%", 810, 500, 24, False), t("186 real demos: 40%", 810, 535, 24, False)])
        # 8. takeaway
        lines = ["Dreams from new initial states make the policy push the T closer to the goal,",
                 "but not reliably over the 95% success line.",
                 "Limitation: dreams follow the task but not physics (PA 0.48 vs 0.86):",
                 "the block deforms, objects vanish, and the IDM still labels those frames.",
                 "Fix: a larger pre-trained WM (Cosmos-Predict2), physics-aware dream filtering,",
                 "then more dreams (1000 dreams: coverage 0.63, single seed)."]
        v.scene(9, texts=[t("Takeaway", "c", 120, 42)] +
                [t(s, 90, 220 + 60 * i, 24, i < 2, "white" if i < 2 else "0xd0d8e0")
                 for i, s in enumerate(lines)])

        out = R / "part2_demo.mp4"
        v.concat(out)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                         capture_output=True, text=True).stdout.strip()
    print(f"wrote {out} ({float(dur):.0f} s)")


if __name__ == "__main__":
    main()
