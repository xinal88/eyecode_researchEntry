"""DreamGen step 3: Inverse Dynamics Model (IDM) that turns action-free videos into
"neural trajectories" by pseudo-labelling actions (and the proprioceptive agent state,
which the policy also consumes but a generated video does not provide).

Input : 4 consecutive frames [t-1, t, t+1, t+2] (edge padded), 96x96 RGB.
Output: agent state s_t and commanded action a_t, both in PushT pixel space.

  python idm.py train --split target --steps 8000
  python idm.py eval  --split val
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CKPT, RESULTS, denorm_xy, episodes, norm_xy

CTX = (-1, 0, 1, 2)


def window_idx(n):
    idx = np.arange(n)[:, None] + np.array(CTX)[None]
    return np.clip(idx, 0, n - 1)


class IDM(nn.Module):
    def __init__(self):
        super().__init__()
        c = len(CTX) * 3

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())
        self.net = nn.Sequential(blk(c, 64), blk(64, 128), blk(128, 256), blk(256, 256))  # 96 -> 6
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256 * 6 * 6, 512), nn.SiLU(), nn.Linear(512, 4))

    def forward(self, x):  # x: B, len(CTX), 96, 96, 3 uint8
        x = x.float().div(127.5).sub(1).permute(0, 1, 4, 2, 3).flatten(1, 2)
        return self.head(self.net(x))  # [state_xy, action_xy] normalised


def build(split):
    X, Y = [], []
    for ep in episodes(split):
        w = window_idx(len(ep["frames"]))
        X.append(ep["frames"][w])
        Y.append(np.concatenate([norm_xy(ep["state"]), norm_xy(ep["action"])], 1))
    return np.concatenate(X), np.concatenate(Y).astype(np.float32)


@torch.no_grad()
def label_video(model, frames, device="cuda", bs=256):
    """frames (T,96,96,3) uint8 -> state (T,2), action (T,2) in pixel space."""
    model.eval()
    w = window_idx(len(frames))
    out = []
    for i in range(0, len(w), bs):
        x = torch.from_numpy(frames[w[i:i + bs]]).to(device)
        out.append(model(x).float().cpu().numpy())
    out = np.concatenate(out)
    return denorm_xy(out[:, :2]), denorm_xy(out[:, 2:])


def load_idm(name="idm_target", device="cuda"):
    m = IDM().to(device)
    m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=device))
    return m.eval()


def train(args):
    torch.manual_seed(0)
    X, Y = build(args.split)
    Xv, Yv = build("val")
    print(f"IDM train samples {len(X)}  val {len(Xv)}")
    dev = "cuda"
    m = IDM().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    for step in range(1, args.steps + 1):
        m.train()
        i = rng.integers(0, len(X), args.bs)
        x = torch.from_numpy(X[i]).to(dev)
        # light photometric/shift augmentation so the IDM tolerates world-model artifacts
        if True:
            sh = rng.integers(-3, 4, 2)
            x = torch.roll(x, shifts=(int(sh[0]), int(sh[1])), dims=(2, 3))
        y = torch.from_numpy(Y[i]).to(dev)
        y = y.clone()
        y[:, [0, 2]] += float(sh[1]) * 512 / 96 / 256  # shift labels consistently (x ~ width)
        y[:, [1, 3]] += float(sh[0]) * 512 / 96 / 256
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.smooth_l1_loss(m(x).float(), y, beta=0.02)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps:
            print(f"step {step} loss {loss.item():.5f}  val {evaluate(m, Xv, Yv)}", flush=True)
    torch.save(m.state_dict(), CKPT / f"idm_{args.split}.pt")
    res = evaluate(m, Xv, Yv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RESULTS / f"idm_{args.split}_val_metrics.json", "w"), indent=2)
    print("final", res)


@torch.no_grad()
def evaluate(m, X, Y, bs=512):
    m.eval()
    P = np.concatenate([m(torch.from_numpy(X[i:i + bs]).cuda()).float().cpu().numpy() for i in range(0, len(X), bs)])
    err = np.abs(denorm_xy(P) - denorm_xy(Y))  # pixel space (0..512)
    return {"state_mae_px": float(err[:, :2].mean()), "action_mae_px": float(err[:, 2:].mean()),
            "action_l2_px": float(np.linalg.norm(denorm_xy(P[:, 2:]) - denorm_xy(Y[:, 2:]), axis=1).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--split", default="target")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=128)
    train(ap.parse_args())
