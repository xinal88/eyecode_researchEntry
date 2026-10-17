"""Small Cosmos-style latent video world model (Video2World) used for DreamGen step 1.

Design (scaled-down from Cosmos-Predict / Cosmos-1.0-Diffusion Video2World):
  * operates in the latent space of the official Cosmos Tokenizer CV4x8x8 (frozen);
  * DiT backbone with adaLN-zero timestep conditioning and full spatio-temporal attention
    over 3D patches (factorised sin-cos position embedding over frame / row / column);
  * rectified-flow (flow matching) objective with logit-normal time sampling;
  * Video2World conditioning exactly as in Cosmos: the first `n_cond` latent frames are
    replaced by the clean latents of the conditioning video, flagged with an indicator
    channel, and excluded from the loss. n_cond is drawn from {1, 2} during training so
    the same model can (a) start a "dream" from a single image and (b) extend a video
    autoregressively from its last 5 pixel frames (= 2 causal latent frames).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_1d(n, dim):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    emb = torch.zeros(n, dim)
    emb[:, 0::2] = torch.sin(pos * freq)
    emb[:, 1::2] = torch.cos(pos * freq)
    return emb


def sincos_3d(f, h, w, dim):
    # split channels: 1/4 time, 3/8 rows, 3/8 cols (all even)
    dt = dim // 4 // 2 * 2
    dh = (dim - dt) // 2 // 2 * 2
    dw = dim - dt - dh
    et = sincos_1d(f, dt)[:, None, None].expand(f, h, w, dt)
    eh = sincos_1d(h, dh)[None, :, None].expand(f, h, w, dh)
    ew = sincos_1d(w, dw)[None, None, :].expand(f, h, w, dw)
    return torch.cat([et, eh, ew], -1).reshape(f * h * w, dim)


class TimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = 128
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = (t[:, None] * 1000.0) * freq[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.qk_norm = nn.LayerNorm(dim // heads, eps=1e-6)  # QK-norm for stability (as in Cosmos)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate="tanh"),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
        h = self.norm1(x) * (1 + s1) + b1
        B, N, D = h.shape
        q, k, v = self.qkv(h).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.qk_norm(q), self.qk_norm(k)
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + g1 * self.proj(h)
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class LatentVideoDiT(nn.Module):
    def __init__(self, in_ch=16, frames=9, size=12, patch=1, dim=512, depth=12, heads=8):
        super().__init__()
        self.in_ch, self.frames, self.size, self.patch = in_ch, frames, size, patch
        self.gh = size // patch
        pdim = (in_ch + 1) * patch * patch  # +1 = conditioning-indicator channel
        self.embed = nn.Linear(pdim, dim)
        self.register_buffer("pos", sincos_3d(frames, self.gh, self.gh, dim)[None], persistent=False)
        self.temb = TimestepEmbed(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, in_ch * patch * patch)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def patchify(self, x):  # B,C,F,H,W -> B, F*gh*gh, C*p*p
        B, C, Fr, H, W = x.shape
        p = self.patch
        x = x.reshape(B, C, Fr, H // p, p, W // p, p).permute(0, 2, 3, 5, 1, 4, 6)
        return x.reshape(B, Fr * (H // p) * (W // p), C * p * p)

    def unpatchify(self, x, Fr):
        B = x.shape[0]
        p, g, C = self.patch, self.gh, self.in_ch
        x = x.reshape(B, Fr, g, g, C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
        return x.reshape(B, C, Fr, g * p, g * p)

    def forward(self, x, t, cond_mask):
        """x: B,C,F,H,W noisy latents (conditioning frames already clean); t: B in [0,1];
        cond_mask: B,F bool. Returns predicted velocity (eps - x0) for all frames."""
        B, C, Fr, H, W = x.shape
        ind = cond_mask[:, None, :, None, None].float().expand(B, 1, Fr, H, W)
        h = self.embed(self.patchify(torch.cat([x, ind], 1))) + self.pos[:, : Fr * self.gh * self.gh]
        c = self.temb(t)
        for blk in self.blocks:
            h = blk(h, c)
        s, b = self.ada_out(c)[:, None].chunk(2, -1)
        h = self.norm_out(h) * (1 + s) + b
        return self.unpatchify(self.out(h), Fr)


# --------------------------------------------------------------------------- flow matching
def make_cond_mask(B, Fr, n_cond, device):
    m = torch.zeros(B, Fr, dtype=torch.bool, device=device)
    for i, n in enumerate(n_cond.tolist()):
        m[i, :n] = True
    return m


def fm_loss(model, x0, n_cond):
    """Rectified flow: x_t = (1-t) x0 + t eps, v* = eps - x0, t ~ logit-normal(0, 1)."""
    B = x0.shape[0]
    t = torch.sigmoid(torch.randn(B, device=x0.device))
    eps = torch.randn_like(x0)
    tt = t[:, None, None, None, None]
    xt = (1 - tt) * x0 + tt * eps
    mask = make_cond_mask(B, x0.shape[2], n_cond, x0.device)
    m5 = mask[:, None, :, None, None]
    xt = torch.where(m5, x0, xt)
    v = model(xt, t, mask)
    err = (v - (eps - x0)) ** 2
    w = (~m5).float().expand_as(err)
    return (err * w).sum() / w.sum()


@torch.no_grad()
def sample(model, cond, n_cond, steps=40, frames=None, generator=None):
    """cond: B,C,F,H,W tensor whose first n_cond frames are the clean conditioning latents."""
    B, C, Fr, H, W = cond.shape
    frames = frames or Fr
    x = torch.randn(B, C, frames, H, W, device=cond.device, generator=generator, dtype=cond.dtype)
    mask = make_cond_mask(B, frames, torch.full((B,), n_cond), cond.device)
    m5 = mask[:, None, :, None, None]
    ts = torch.linspace(1.0, 0.0, steps + 1, device=cond.device)
    for i in range(steps):
        x = torch.where(m5, cond[:, :, :frames], x)
        v = model(x, ts[i].expand(B), mask)
        x = x + (ts[i + 1] - ts[i]) * v
    return torch.where(m5, cond[:, :, :frames], x)
