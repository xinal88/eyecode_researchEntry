# Part 2 — World Models: DreamGen × Cosmos, reproduced at small scale on PushT

Assigned papers: **DreamGen** (Jang et al., arXiv:2505.12705) and **Cosmos World Foundation Model Platform**
(NVIDIA, arXiv:2501.03575). Reading notes, contributions and a side-by-side comparison: [PAPER_NOTES.md](PAPER_NOTES.md).

## What is reproduced
The full DreamGen pipeline, end to end, with a Cosmos component at its core, scaled down to a task where
closed-loop policy evaluation is cheap and exact (the PushT simulator):

| DreamGen step | This implementation | File |
|---|---|---|
| 0. latent space | **official pretrained Cosmos Tokenizer CV4×8×8** (frozen, causal, 16-ch latents) | `cosmos_tok.py` |
| 1. world model, pre-train → fine-tune | Cosmos-style Video2World latent DiT (57.7 M params, flow matching, frame-concat conditioning) pre-trained on 161 demos, LoRA-free fine-tune on the 25 target demos | `world_model.py`, `train_world_model.py` |
| 2. dreams | rollouts from **new initial states** (fresh env resets), 117 frames each, autoregressive 4×33-frame chunks (1 then 2 latent conditioning frames) | `dream.py` |
| 3. pseudo-actions | inverse dynamics model trained on the 25 target demos (frames t-1…t+2 → state, action) | `idm.py` |
| bench | automatic judge mirroring DreamGen Bench: instruction-following (goal coverage) + physics-alignment (rigidity, fragmentation, agent presence, contact-less motion, teleport) | `common.py` |
| 4. policy | Diffusion Policy (CNN + spatial softmax, 1D U-Net, DDPM-100 / DDIM-10), co-trained **1:1 real : neural** as in the paper, evaluated closed-loop on 50 fixed seeds | `policy.py` |

Data split (LeRobot `lerobot/pusht`, 206 human demos, 10 fps, 96×96): **target** 25 demos (the only real data the
policy, IDM and fine-tune see), **pretrain** 161 demos (stand-in for the web-scale pre-training of Cosmos/WAN),
**val** 20 demos (world-model / IDM metrics only). See `data/splits.json`, written by `make_splits.py` (seeded
permutation; verified bit-exact against the val clips of the original world-model run).

## Reproduce
```bash
# from the repo root; python 3.12 venv with torch (CUDA), see requirements.txt
python part2_world_models/prepare_pusht.py                       # decode dataset -> data/pusht_96.npz
python part2_world_models/make_splits.py                         # -> data/splits.json (target/val/pretrain)
python part2_world_models/eval_cosmos_tokenizer.py --sizes 96 128 192 256   # Cosmos tokenizer PSNR/SSIM
cd part2_world_models
python idm.py train --split target --steps 8000                                   # step 3 model
python train_world_model.py --split pretrain --steps 8000 --name wm_pre           # step 1a
python train_world_model.py --split target --init wm_pre --steps 3000 --lr 5e-5 --name wm_ft   # step 1b
python train_world_model.py --split target --steps 8000 --name wm_scratch         # ablation: no pre-training
python dream.py --wm wm_ft --n 1000 --bs 50 --chunks 4 --steps 25 --out dreams_wm_ft            # steps 2+3
python dream.py --wm wm_scratch --n 300 --out dreams_wm_scratch
python policy.py --name real25 --real target                                      # step 4 baselines / variants
python policy.py --name real25_dream300 --real target --dreams dreams_wm_ft --n_dreams 300
PY=/path/to/python bash run_queue.sh <gpu> A|B|C|D|E   # the full policy grid (resumable, run inside tmux)
python analyze.py                  # -> results/SUMMARY.md + results/figures/*.png
python make_demo_video.py          # -> results/part2_demo.mp4
```
Downloads: `lerobot/pusht` (7.7 MB) and `nvidia/Cosmos-0.1-Tokenizer-CV4x8x8` encoder/decoder JIT (147 MB).
Compute used: one shared RTX 5880 Ada (48 GB); world model ≈ 0.35 s/step, each policy run ≈ 30 min.

## Outputs
* `results/part2_demo.mp4` — 83 s demo video: real demo vs. dreams, judge scores, failure cases, policy rollouts, results
* `results/SUMMARY.md` — all tables (tokenizer, IDM, world model, judge, policies with 95 % CIs)
* `results/figures/` — policy bar chart, world-model validation curve
* `results/world_model/*/val_step*.png` — real (top) vs generated (bottom) futures on held-out demos
* `results/dreams/*/dream_*.mp4`, `judge_scores.json` — example neural trajectories and their judge scores
* `results/policy/*/eval.json`, `eval_ep*.mp4` — closed-loop results and rollouts
* `results/misc/wm_pre_dream_failures.png` — failure cases (object impermanence, deformation)

## Why not the official Cosmos-Predict2 / GR00T-Dreams checkpoints?
`Cosmos-Predict2-2B-Video2World` needs a 3.9 GB DiT checkpoint plus a T5-XXL text encoder (> 10 GB); on this
shared server downloads are capped at ≈ 0.5–1 MB/s, so fetching them would take many hours and slow the
machine down for other users. The task allows reduced models, so the official **Cosmos tokenizer** is used as
the latent space and a scaled-down Cosmos-style Video2World model is trained on top of it. The code paths
(tokenizer → latent DiT → autoregressive Video2World extension) match the paper's design, so swapping in a
Cosmos-Predict2 checkpoint only replaces `LatentVideoDiT` + `sample()`.
