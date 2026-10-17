"""DreamGen steps 2+3: generate "dreams" and turn them into neural trajectories.

  2. Roll out the fine-tuned world model from *new* initial frames (fresh PushT resets
     never seen in the demos). Long videos are produced autoregressively: chunk 1 is
     conditioned on one image (1 latent frame), each further chunk on the last 5 generated
     frames (2 causal latent frames), mirroring Cosmos Video2World extension.
  3. Score each dream with the automatic judge (instruction-following / physics proxies)
     and pseudo-label states + actions with the IDM.

  python dream.py --wm wm_ft --n 300 --chunks 4 --seed0 0 --out dreams_wm_ft
"""
import argparse
import json
import time

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the env)
import imageio.v2 as imageio
import numpy as np
import torch

from common import CKPT, DATA, RESULTS, judge_video
from cosmos_tok import CosmosTokenizer
from idm import label_video, load_idm
from world_model import LatentVideoDiT, sample

SIZE = 192


def initial_frames(n, seed0):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    return np.stack([env.reset(seed=seed0 + i)[0]["pixels"] for i in range(n)])


@torch.no_grad()
def rollout(model, tok, mean, std, first, chunks, steps, gen):
    """first: (B,96,96,3) uint8 -> video (B, 33 + 28*(chunks-1), 96, 96, 3) uint8."""
    B = len(first)
    F_lat = model.frames
    ctx_pix = tok.to_tensor(first[:, None].reshape(-1, 96, 96, 3), SIZE)  # 1,3,B,H,W (T=B)
    ctx_pix = ctx_pix[0].permute(1, 0, 2, 3)[:, :, None]  # B,3,1,H,W
    videos, n_cond = [], 1
    for c in range(chunks):
        z_ctx = (tok.encode(ctx_pix).float() - mean) / std  # B,16,n_cond,24,24
        cond = torch.zeros(B, 16, F_lat, z_ctx.shape[-2], z_ctx.shape[-1], device="cuda")
        cond[:, :, :n_cond] = z_ctx
        with torch.autocast("cuda", torch.bfloat16):
            z = sample(model, cond, n_cond=n_cond, steps=steps, generator=gen)
        pix = tok.decode(z.float() * std + mean).float().clamp(-1, 1)  # B,3,33,192,192
        new = pix if c == 0 else pix[:, :, 5:]
        videos.append(tok.to_uint8(new, size=96))
        ctx_pix, n_cond = pix[:, :, -5:], 2
    return np.concatenate(videos, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="wm_ft")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bs", type=int, default=25)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"dreams_{args.wm}"
    vis_dir = RESULTS / "dreams" / out
    vis_dir.mkdir(parents=True, exist_ok=True)

    tok = CosmosTokenizer()
    ck = torch.load(CKPT / f"{args.wm}.pt", map_location="cuda")
    a = ck["args"]
    model = LatentVideoDiT(16, 9, 24, patch=2, dim=a["dim"], depth=a["depth"]).cuda().eval()
    model.load_state_dict(ck["ema"])
    mean, std = ck["mean"].cuda(), ck["std"].cuda()
    idm = load_idm("idm_target")

    firsts = initial_frames(args.n, args.seed0)
    gen = torch.Generator(device="cuda").manual_seed(args.seed0)
    L = 33 + 28 * (args.chunks - 1)
    # frames are streamed into a memory-mapped .npy (1000 dreams = 3.2 GB would not fit in RAM
    # next to the other jobs on this 24 GB box); the small .npz is written last = "done" marker
    vids = np.lib.format.open_memmap(DATA / f"{out}_frames.npy", mode="w+", dtype=np.uint8,
                                     shape=(args.n, L, 96, 96, 3))
    t0 = time.time()
    for i in range(0, args.n, args.bs):
        v = rollout(model, tok, mean, std, firsts[i:i + args.bs], args.chunks, args.steps, gen)
        vids[i:i + len(v)] = v
        print(f"dreamed {i + len(v)}/{args.n}  ({time.time() - t0:.0f}s)", flush=True)
    vids.flush()
    gen_time = (time.time() - t0) / args.n

    states, actions, scores = [], [], []
    for j in range(args.n):
        v = np.asarray(vids[j])
        s, act = label_video(idm, v)
        states.append(s)
        actions.append(act)
        scores.append(judge_video(v))
        if (j + 1) % 100 == 0:
            print(f"labelled+judged {j + 1}/{args.n}", flush=True)
    json.dump(scores, open(vis_dir / "judge_scores.json", "w"))
    summ = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    summ["gen_seconds_per_video"] = gen_time
    json.dump(summ, open(vis_dir / "judge_summary.json", "w"), indent=2)
    for j in range(min(6, args.n)):
        imageio.mimsave(vis_dir / f"dream_{j}.mp4", np.asarray(vids[j]).repeat(4, 1).repeat(4, 2), fps=10)
    np.savez(DATA / f"{out}.npz", state=np.stack(states), action=np.stack(actions),
             seeds=np.arange(args.seed0, args.seed0 + args.n))
    print("judge summary", json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()
