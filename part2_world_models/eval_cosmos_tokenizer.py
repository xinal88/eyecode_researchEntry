"""Evaluate Cosmos Tokenizer reconstruction quality (PSNR / SSIM) on robot video.

Reproduces, at small scale, the tokenizer evaluation protocol of the Cosmos paper
(Sec. 5 / Table: reconstruction PSNR & SSIM) on out-of-domain robot clips:
  * PushT (2D manipulation, 96x96 renders, 10 fps), several working resolutions
  * optionally any mp4 (e.g. the ROSOrin Gazebo camera recording from Part 1)

Usage:
  python eval_cosmos_tokenizer.py --sizes 96 192 256 --clips 40
  python eval_cosmos_tokenizer.py --video ../part1_rosorin_sim/media/camera.mp4 --sizes 256
"""
import argparse
import json
from pathlib import Path

import av
import imageio.v2 as imageio
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from cosmos_tok import ROOT, CosmosTokenizer

OUT = ROOT / "part2_world_models/results/tokenizer"


def metrics(gt, rec):
    psnr = [peak_signal_noise_ratio(g, r, data_range=255) for g, r in zip(gt, rec)]
    ssim = [structural_similarity(g, r, channel_axis=-1, data_range=255) for g, r in zip(gt, rec)]
    return float(np.mean(psnr)), float(np.mean(ssim))


def load_clips(args):
    if args.video:
        frames = np.stack([f.to_ndarray(format="rgb24") for f in av.open(args.video).decode(video=0)])
        frames = frames[:: args.stride]
        h, w = frames.shape[1:3]
        s = min(h, w)  # centre square crop so a single "size" applies
        frames = frames[:, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        n = (len(frames) - 1) // (args.T - 1)
        return [frames[i * (args.T - 1): i * (args.T - 1) + args.T] for i in range(min(n, args.clips))], Path(args.video).stem
    d = dict(np.load(ROOT / "data/pusht_96.npz"))  # decompress once
    rng = np.random.default_rng(0)
    clips = []
    for e in rng.permutation(len(d["ep_start"]))[: args.clips]:
        s, t = d["ep_start"][e], d["ep_end"][e]
        if t - s >= args.T:
            o = rng.integers(s, t - args.T + 1)
            clips.append(d["frames"][o:o + args.T])
    return clips, "pusht"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[96, 192, 256])
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--T", type=int, default=17, help="clip length, must be 1+4k")
    ap.add_argument("--video", default=None)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tok = CosmosTokenizer()
    clips, name = load_clips(args)
    native = clips[0].shape[1]
    report = {"dataset": name, "n_clips": len(clips), "T": args.T, "native_res": native, "results": {}}
    for size in args.sizes:
        gts, recs = [], []
        for c in clips:
            x = tok.to_tensor(c, size)
            z = tok.encode(x)
            rec = tok.to_uint8(tok.decode(z), size=native)[0]
            gts.append(c)
            recs.append(rec)
        gt, rec = np.concatenate(gts), np.concatenate(recs)
        psnr, ssim = metrics(gt, rec)
        # Baseline: plain bicubic down/up-sampling with the same spatial compression (8x)
        # tells us how much of the error is just resolution loss.
        report["results"][size] = {
            "psnr": psnr, "ssim": ssim,
            "latent_shape": list(z.shape[1:]),
            "compression": float(np.prod(c.shape) / np.prod(z.shape[1:])),
        }
        print(f"[{name}] size {size}: PSNR {psnr:.2f} dB  SSIM {ssim:.4f}  latent {tuple(z.shape[1:])}")
        # side-by-side strip of 6 frames: top GT, bottom reconstruction
        idx = np.linspace(0, args.T - 1, 6).astype(int)
        strip = np.concatenate([np.concatenate(list(clips[0][idx]), 1), np.concatenate(list(recs[0][idx]), 1)], 0)
        imageio.imwrite(OUT / f"{name}_recon_{size}.png", strip)
    with open(OUT / f"{name}_tokenizer_metrics.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
