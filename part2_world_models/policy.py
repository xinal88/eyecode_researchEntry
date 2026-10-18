"""DreamGen step 4: train a visuomotor Diffusion Policy on real + neural trajectories and
evaluate it closed-loop in the PushT simulator.

Compact re-implementation of Diffusion Policy (Chi et al. 2023, the policy class DreamGen
also uses): 2 observation steps (image + agent position), CNN encoder with spatial-softmax
keypoints, 1D temporal conv U-Net with FiLM conditioning, 100-step DDPM training,
10-step DDIM inference, action chunks of 16 with 8 executed.

  python policy.py --name real25           --real target
  python policy.py --name real25_dream300  --real target --dreams dreams_wm_ft --n_dreams 300
  python policy.py --name dream300_only    --dreams dreams_wm_ft --n_dreams 300
"""
import argparse
import copy
import json
import math
import time

import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler

from common import CKPT, DATA, RESULTS, episodes, norm_xy

N_OBS, HORIZON, N_EXEC, CROP = 2, 16, 8, 84


# ------------------------------------------------------------------------------- data
class TrajData:
    def __init__(self, trajs):
        self.trajs = trajs
        self.index = [(i, t) for i, tr in enumerate(trajs) for t in range(len(tr["frames"]))]

    def __len__(self):
        return len(self.index)

    def sample(self, rng, bs):
        img, st, act = [], [], []
        for k in rng.integers(0, len(self.index), bs):
            i, t = self.index[k]
            tr = self.trajs[i]
            L = len(tr["frames"])
            oi = np.clip(np.arange(t - N_OBS + 1, t + 1), 0, L - 1)
            ai = np.clip(np.arange(t, t + HORIZON), 0, L - 1)
            img.append(tr["frames"][oi])
            st.append(tr["state"][oi])
            act.append(tr["action"][ai])
        return np.stack(img), norm_xy(np.stack(st)), norm_xy(np.stack(act))


def load_dreams(name, n, min_pa=None):
    """First n neural trajectories of a dream set; optionally keep only those whose judge
    physics-alignment score is >= min_pa (DreamGen-Bench-style filtering)."""
    mm = DATA / f"{name}_frames.npy"
    d = np.load(DATA / f"{name}.npz")
    if not mm.exists():  # older layout: frames stored inside the npz
        np.save(mm, d["frames"])
    frames = np.load(mm, mmap_mode="r")
    scores = json.load(open(RESULTS / "dreams" / name / "judge_scores.json")) if min_pa is not None else None
    trajs = []
    for i in range(min(n, len(frames))):
        if min_pa is not None and scores[i]["pa_score"] < min_pa:
            continue
        trajs.append({"frames": np.ascontiguousarray(frames[i]), "state": d["state"][i], "action": d["action"][i]})
    return trajs


# ------------------------------------------------------------------------------- model
class SpatialSoftmax(nn.Module):
    def __init__(self, c, kp=32):
        super().__init__()
        self.proj = nn.Conv2d(c, kp, 1)

    def forward(self, x):
        x = self.proj(x)
        B, K, H, W = x.shape
        p = x.flatten(2).softmax(-1)
        ys = torch.linspace(-1, 1, H, device=x.device)
        xs = torch.linspace(-1, 1, W, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.cat([(p * gx.flatten()).sum(-1), (p * gy.flatten()).sum(-1)], -1)  # B, 2K


class ObsEncoder(nn.Module):
    def __init__(self, out=128):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.ReLU())
        self.cnn = nn.Sequential(blk(3, 32, 1), blk(32, 64, 2), blk(64, 128, 2), blk(128, 128, 1))
        self.ss = SpatialSoftmax(128, 32)
        self.fc = nn.Linear(64, out)

    def forward(self, img):  # B, N_OBS, H, W, 3 uint8
        B = img.shape[0]
        x = img.flatten(0, 1).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        return self.fc(self.ss(self.cnn(x))).view(B, -1)


def time_emb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock1D(nn.Module):
    def __init__(self, i, o, cond):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.GroupNorm(8, o), nn.Mish())
        self.c2 = nn.Sequential(nn.Conv1d(o, o, 5, padding=2), nn.GroupNorm(8, o), nn.Mish())
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond, 2 * o))
        self.skip = nn.Conv1d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, c):
        h = self.c1(x)
        s, b = self.film(c)[:, :, None].chunk(2, 1)
        h = self.c2(h * (1 + s) + b)
        return h + self.skip(x)


class CondUnet1D(nn.Module):
    def __init__(self, act_dim=2, cond_dim=260, dims=(128, 256, 512)):
        super().__init__()
        cd = cond_dim + 128
        self.tmlp = nn.Sequential(nn.Linear(128, 512), nn.Mish(), nn.Linear(512, 128))
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        ch = [act_dim] + list(dims)
        for i in range(len(dims)):
            self.down.append(nn.ModuleList([ResBlock1D(ch[i], ch[i + 1], cd), ResBlock1D(ch[i + 1], ch[i + 1], cd),
                                            nn.Conv1d(ch[i + 1], ch[i + 1], 3, 2, 1) if i < len(dims) - 1 else nn.Identity()]))
        self.mid = nn.ModuleList([ResBlock1D(dims[-1], dims[-1], cd), ResBlock1D(dims[-1], dims[-1], cd)])
        for i in reversed(range(1, len(dims))):
            self.up.append(nn.ModuleList([ResBlock1D(dims[i] * 2, dims[i - 1], cd), ResBlock1D(dims[i - 1], dims[i - 1], cd),
                                          nn.ConvTranspose1d(dims[i - 1], dims[i - 1], 4, 2, 1)]))
        self.final = nn.Sequential(nn.Conv1d(dims[0], dims[0], 5, padding=2), nn.GroupNorm(8, dims[0]), nn.Mish(),
                                   nn.Conv1d(dims[0], act_dim, 1))

    def forward(self, x, t, cond):  # x: B, T, A
        c = torch.cat([self.tmlp(time_emb(t, 128)), cond], -1)
        h = x.transpose(1, 2)
        skips = []
        for r1, r2, dn in self.down:
            h = r2(r1(h, c), c)
            skips.append(h)
            h = dn(h)
        for m in self.mid:
            h = m(h, c)
        for r1, r2, upm in self.up:
            h = torch.cat([h, skips.pop()], 1)
            h = upm(r2(r1(h, c), c))
        return self.final(h).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ObsEncoder(128)
        self.net = CondUnet1D(2, N_OBS * 128 + N_OBS * 2)
        kw = dict(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True,
                  prediction_type="epsilon")
        self.train_sched = DDPMScheduler(**kw)
        self.infer_sched = DDIMScheduler(**kw)

    def cond(self, img, st):
        return torch.cat([self.enc(img), st.flatten(1)], -1)

    def loss(self, img, st, act):
        c = self.cond(img, st)
        noise = torch.randn_like(act)
        t = torch.randint(0, 100, (act.shape[0],), device=act.device)
        return F.mse_loss(self.net(self.train_sched.add_noise(act, noise, t), t, c), noise)

    @torch.no_grad()
    def act(self, img, st, steps=10):
        c = self.cond(img, st)
        x = torch.randn(img.shape[0], HORIZON, 2, device=img.device)
        self.infer_sched.set_timesteps(steps)
        for t in self.infer_sched.timesteps:
            x = self.infer_sched.step(self.net(x, t.expand(len(x)).to(img.device), c), t, x).prev_sample
        return x


def crop(img, rng=None):
    """Random (train) / centre (eval) 84x84 crop, resized back to 96 is unnecessary: the CNN
    is fully convolutional + spatial softmax, so train and eval just use the same 84 crop."""
    H = img.shape[-3]
    if rng is None:
        o = (H - CROP) // 2
        return img[..., o:o + CROP, o:o + CROP, :]
    oy, ox = rng.integers(0, H - CROP + 1, 2)
    return img[..., oy:oy + CROP, ox:ox + CROP, :]


# ------------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(policy, n_eps=50, seed0=100000, max_steps=300, save_dir=None, n_video=4):
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                     max_episode_steps=max_steps) for _ in range(n_eps)]
    obs = [e.reset(seed=seed0 + i)[0] for i, e in enumerate(envs)]
    hist = [[o, o] for o in obs]
    done = np.zeros(n_eps, bool)
    success = np.zeros(n_eps, bool)
    max_cov = np.zeros(n_eps)
    frames = [[o["pixels"]] for o in obs[:n_video]]
    steps = 0
    while not done.all() and steps < max_steps:
        act_ids = np.where(~done)[0]
        img = torch.from_numpy(np.stack([np.stack([h["pixels"] for h in hist[i][-N_OBS:]]) for i in act_ids])).cuda()
        st = torch.from_numpy(norm_xy(np.stack([np.stack([h["agent_pos"] for h in hist[i][-N_OBS:]]) for i in act_ids]))).cuda()
        a = policy.act(crop(img), st).cpu().numpy() * 256 + 256
        for k in range(N_EXEC):
            for j, i in enumerate(act_ids):
                if done[i]:
                    continue
                o, r, term, trunc, info = envs[i].step(np.clip(a[j, k], 0, 512))
                hist[i].append(o)
                max_cov[i] = max(max_cov[i], info.get("coverage", 0.0))
                success[i] |= bool(info.get("is_success", False))
                if i < n_video:
                    frames[i].append(o["pixels"])
                if term or trunc:
                    done[i] = True
            steps += 1
    if save_dir:
        for i in range(min(n_video, n_eps)):
            imageio.mimsave(save_dir / f"eval_ep{i}_{'success' if success[i] else 'fail'}.mp4",
                            np.stack(frames[i]).repeat(4, 1).repeat(4, 2), fps=10)
    return {"success_rate": float(success.mean()), "mean_max_coverage": float(max_cov.mean()),
            "n_episodes": n_eps, "per_episode_success": success.tolist(), "per_episode_coverage": max_cov.tolist()}


# ------------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--real", default=None, help="split name of real demos (e.g. target)")
    ap.add_argument("--extra_real", default=None, help="add another real split (upper bound)")
    ap.add_argument("--dreams", default=None)
    ap.add_argument("--n_dreams", type=int, default=0)
    ap.add_argument("--min_pa", type=float, default=None, help="filter dreams by judge PA score")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--eval_eps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true",
                    help="skip training, evaluate checkpoints/policy_<name>.pt (e.g. a run whose eval crashed)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = RESULTS / "policy" / args.name
    out.mkdir(parents=True, exist_ok=True)

    real = []
    for sp in [args.real, args.extra_real]:
        if sp:
            real += episodes(sp)
    neural = load_dreams(args.dreams, args.n_dreams, args.min_pa) if args.dreams else []
    sets = [TrajData(t) for t in (real, neural) if t]
    print(f"[{args.name}] real trajs {len(real)}  neural trajs {len(neural)}  "
          f"transitions {[len(s) for s in sets]}", flush=True)
    if args.eval_only:
        ema = DiffusionPolicy().cuda().eval()
        ema.load_state_dict(torch.load(CKPT / f"policy_{args.name}.pt", map_location="cuda"))
        res = evaluate(ema, args.eval_eps, save_dir=out)
        res.update({"name": args.name, "n_real": len(real), "n_neural": len(neural), "steps": args.steps,
                    "eval_only": True})
        json.dump(res, open(out / "eval.json", "w"), indent=2)
        print(f"[{args.name}] SUCCESS {res['success_rate']:.3f}  mean max coverage {res['mean_max_coverage']:.3f}", flush=True)
        return

    pol = DiffusionPolicy().cuda()
    ema = copy.deepcopy(pol).eval()
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-4, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # co-training: each batch is split evenly between the real and the neural datasets
        parts = [s.sample(rng, args.bs // len(sets)) for s in sets]
        img, st, act = (np.concatenate(z) for z in zip(*parts))
        img = torch.from_numpy(crop(img, rng)).cuda()
        with torch.autocast("cuda", torch.bfloat16):
            loss = pol.loss(img, torch.from_numpy(st).cuda(), torch.from_numpy(act).cuda().float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), pol.parameters()):
                pe.lerp_(pm, 1 - d)
        if step % 2000 == 0:
            print(f"[{args.name}] step {step} loss {loss.item():.4f} ({(time.time() - t0) / step:.3f}s/it)", flush=True)
    torch.save(ema.state_dict(), CKPT / f"policy_{args.name}.pt")
    res = evaluate(ema, args.eval_eps, save_dir=out)
    res.update({"name": args.name, "n_real": len(real), "n_neural": len(neural), "steps": args.steps,
                "train_minutes": (time.time() - t0) / 60})
    json.dump(res, open(out / "eval.json", "w"), indent=2)
    print(f"[{args.name}] SUCCESS {res['success_rate']:.3f}  mean max coverage {res['mean_max_coverage']:.3f}", flush=True)


if __name__ == "__main__":
    main()
