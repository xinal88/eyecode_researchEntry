"""DreamGen step 1: train / fine-tune the latent video world model.

Stage A ("pre-training", stands in for Cosmos/WAN web-scale pre-training):
  python train_world_model.py --split pretrain --steps 20000 --name wm_pre
Stage B (DreamGen: fine-tune the world model on the target robot's own demos):
  python train_world_model.py --split target --init wm_pre --steps 3000 --lr 5e-5 --name wm_ft
Ablation (no pre-training, target data only):
  python train_world_model.py --split target --steps 8000 --name wm_scratch

Clips: 33 frames @10 fps (3.3 s) at 192x192 -> Cosmos CV4x8x8 latents [16, 9, 24, 24].
"""
import argparse
import copy
import json
import time

import imageio.v2 as imageio
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common import CKPT, DATA, RESULTS, episodes
from cosmos_tok import CosmosTokenizer
from world_model import LatentVideoDiT, fm_loss, sample

T_PIX, SIZE, STRIDE = 33, 192, 4


def windows(frames):
    n = len(frames)
    starts = list(range(0, max(n - T_PIX, 0) + 1, STRIDE))
    out = []
    for s in starts:
        w = frames[s:s + T_PIX]
        if len(w) < T_PIX:  # short episode: hold the last frame (robot at rest)
            w = np.concatenate([w, np.repeat(w[-1:], T_PIX - len(w), 0)])
        out.append(w)
    return out


@torch.no_grad()
def encode_split(tok, split):
    path = DATA / f"latents_{SIZE}_{split}.pt"
    if path.exists():
        return torch.load(path)
    clips = [w for ep in episodes(split) for w in windows(ep["frames"])]
    t0 = time.time()
    lat = []
    for i in range(0, len(clips), 8):
        x = torch.cat([tok.to_tensor(w, SIZE) for w in clips[i:i + 8]])
        lat.append(tok.encode(x).cpu())
    lat = torch.cat(lat)
    data = {"latents": lat}
    if split == "val":  # pixel clips are only needed as ground truth for validation
        data["clips"] = torch.from_numpy(np.stack(clips))
    torch.save(data, path)
    print(f"encoded {split}: {tuple(lat.shape)} in {time.time() - t0:.0f}s")
    return data


def psnr_ssim(gt, pr):
    p = np.mean([peak_signal_noise_ratio(a, b, data_range=255) for a, b in zip(gt, pr)])
    s = np.mean([structural_similarity(a, b, channel_axis=-1, data_range=255) for a, b in zip(gt, pr)])
    return float(p), float(s)


@torch.no_grad()
def validate(model, tok, val, mean, std, n=8, steps=30, save=None):
    """Condition on the first frame of real held-out clips and compare generated vs real
    future (PSNR/SSIM are pessimistic for a stochastic model but track progress)."""
    model.eval()
    idx = torch.linspace(0, len(val["latents"]) - 1, n).long()
    z = ((val["latents"][idx].cuda().float() - mean) / std)
    g = torch.Generator(device="cuda").manual_seed(0)
    with torch.autocast("cuda", torch.bfloat16):
        out = sample(model, z, n_cond=1, steps=steps, generator=g)
    pix = tok.to_uint8(tok.decode(out.float() * std + mean), size=96)
    gt = val["clips"][idx].numpy()
    p, s = psnr_ssim(gt[:, 1:].reshape(-1, 96, 96, 3), pix[:, 1:].reshape(-1, 96, 96, 3))
    if save:
        cols = np.linspace(0, T_PIX - 1, 8).astype(int)
        rows = [np.concatenate([np.concatenate(list(gt[i, cols]), 1), np.concatenate(list(pix[i, cols]), 1)], 0)
                for i in range(min(n, 4))]
        imageio.imwrite(save, np.concatenate(rows, 0))
    model.train()
    return p, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--init", default=None)
    ap.add_argument("--name", default="wm_pre")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(0)
    out_dir = RESULTS / "world_model" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = CosmosTokenizer()
    data = encode_split(tok, args.split)
    val = encode_split(tok, "val")
    lat = data["latents"].cuda()
    model = LatentVideoDiT(16, lat.shape[2], lat.shape[-1], patch=2, dim=args.dim, depth=args.depth).cuda()
    if args.init:
        ck = torch.load(CKPT / f"{args.init}.pt", map_location="cuda")
        model.load_state_dict(ck["ema"])
        mean, std = ck["mean"].cuda(), ck["std"].cuda()
    else:
        pre = encode_split(tok, "pretrain")["latents"]  # normalisation stats always from pre-train set
        mean = pre.float().mean((0, 2, 3, 4)).view(1, -1, 1, 1, 1).cuda()
        std = pre.float().std((0, 2, 3, 4)).view(1, -1, 1, 1, 1).cuda()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  train clips {len(lat)}  "
          f"val clips {len(val['latents'])}")
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    warm = 500 if not args.init else 100
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm))
    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = torch.randint(0, len(lat), (args.bs,), device="cuda")
        x0 = (lat[i].float() - mean) / std
        n_cond = torch.randint(1, 3, (args.bs,))  # 1 image or 5 frames of context
        with torch.autocast("cuda", torch.bfloat16):
            loss = fm_loss(model, x0, n_cond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(0.9995, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        if step % 100 == 0:
            log.append({"step": step, "loss": loss.item(), "time": time.time() - t0})
        if step % 2000 == 0 or step == args.steps:
            p, s = validate(ema, tok, val, mean, std, save=out_dir / f"val_step{step}.png")
            log[-1].update({"val_psnr": p, "val_ssim": s})
            print(f"step {step} loss {loss.item():.4f} val PSNR {p:.2f} SSIM {s:.4f} "
                  f"({(time.time() - t0) / step:.3f}s/it)", flush=True)
            torch.save({"ema": ema.state_dict(), "mean": mean.cpu(), "std": std.cpu(), "args": vars(args)},
                       CKPT / f"{args.name}.pt")
    json.dump(log, open(out_dir / "train_log.json", "w"))


if __name__ == "__main__":
    main()
