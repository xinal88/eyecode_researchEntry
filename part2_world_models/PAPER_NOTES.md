# Paper notes — World Models

## Paper 1 — DreamGen: Unlocking Generalization in Robot Learning through Video World Models
Jang, Ye, Lin, … , Zhu, Fan (NVIDIA GEAR et al.), arXiv:2505.12705 (assigned as CoRL 2025). Code: github.com/NVIDIA/GR00T-Dreams

**Problem.** Robot policies generalise poorly beyond the behaviours and environments covered by
expensive teleoperation data. Simulation-based synthetic data has a sim-to-real gap and needs
hand-built assets/tasks. Can a *video* world model be used as a scalable synthetic data engine instead?

**Key idea — a 4-step pipeline producing "neural trajectories":**
1. Fine-tune an image-to-video world model (WAN2.1 mainly; also Cosmos, CogVideoX, Hunyuan) on the
   target robot's teleop videos with LoRA (rank 4, α 4, lr 1e-4) so it learns the embodiment.
2. Roll it out from an initial frame + language instruction ("dreams"), including novel behaviours / scenes.
3. Recover pseudo-actions: an IDM (diffusion transformer, SigLIP-2 encoder, flow matching, 2 frames →
   action chunk, trained on the same robot data) or a latent action model (LAPA, VQ-VAE latent actions).
4. Train visuomotor policies (Diffusion Policy, π0, GR00T N1) on neural trajectories, co-trained 1:1 with real data.

**Contributions / results.**
* RoboCasa (24 tasks, GR00T N1): neural trajectories scaled up to 333× the demos (240k) with log-linear
  gains; 30 / 100 / 300 demos: 17.4→23.3 %, 32.1→39.9 %, 49.6→57.6 % success; neural trajectories only: 20.6 %.
* Real robots (low data): GR1 37→46 %, Franka 23→37 %, SO-100 21→45.5 % average success.
* Generalisation from a single pick-and-place dataset on GR1: 22 new behaviours; new behaviours in seen
  environments 11.2→43.2 %, unseen environments 0→28.5 %.
* DreamGen Bench: instruction following (Qwen2.5-VL / GPT-4o judge) + physics alignment
  (VideoCon-Physics + Qwen2.5-VL); bench score correlates positively with downstream policy success.

**Limitations (authors).** Simple tasks, limited behaviour diversity, very high compute (54 h on 1,500 L40
for 240k RoboCasa videos), manually supplied initial frames, evaluator hallucination, open problem of
true zero-shot generalisation.

## Paper 2 — Cosmos World Foundation Model Platform for Physical AI
NVIDIA (Agarwal et al., 77 authors), arXiv:2501.03575. Code: github.com/nvidia-cosmos

**Problem.** Physical AI needs a "digital twin" of the world to train and evaluate policies safely; building
a capable world model from scratch is prohibitively expensive for most developers.

**Key idea.** A *world foundation model* (WFM) platform: pre-train general video world models once on
internet-scale physical video, then post-train (fine-tune) them cheaply for a specific Physical-AI setup.

**Components.**
* Data curation: ~20 M hours of raw video → ~100 M pre-training clips (2–60 s); split / filter (motion,
  quality, overlay, type) / VLM captioning (VILA) / semantic dedup (−30 %) / sharding.
* Cosmos Tokenizer: causal wavelet encoder-decoder with factorised spatio-temporal conv + attention;
  continuous (CV, 16-ch) and discrete (DV, FSQ 64k vocab) variants; 4×8×8, 8×8×8, 8×16×16 compression.
  DAVIS: CV4×8×8 35.85 dB PSNR / 0.920 SSIM vs 29.29 / 0.864 for CogVideoX tokenizer; 2–12× faster.
* WFMs: diffusion (EDM, DiT with 3D patchify 1×2×2, 3D RoPE, AdaLN-LoRA, T5-XXL cross-attn) 7B/14B
  Text2World → Video2World (conditioning frames concatenated in time); autoregressive (Llama-style GPT on
  DV8×16×16 tokens) 4B/12B → 5B/13B Video2World with a diffusion decoder; prompt upsampler (12B). ~10k H100 × 3 months.
* Post-training: camera control (pose success 82 % vs 43 % CamCo), robot manipulation (instruction video
  prediction: 78 % human preference vs VideoLDM; action-conditioned Bridge: PSNR 21.1 / SSIM 0.82 /
  FVD 190 vs IRASim 19.1 / 0.64 / 593), multi-view driving (FVD 210 vs 884).
* Evaluation: 3D consistency (Sampson error), physics alignment in PhysX/Isaac scenarios (more conditioning
  frames → higher PSNR 17.3→21.1 and object IoU 0.33→0.59; bigger models not clearly more physical).
* Guardrails: pre-Guard (keyword blocking, Aegis LLM) + post-Guard (video safety classifier, face blur).

**Limitations (authors).** Not yet reliable physics simulators: object impermanence, inaccurate contact-rich
dynamics, inconsistent instruction following, gravity / light / fluid violations; evaluation of physical
fidelity is hard and subjective.

## How the two papers relate
| | Cosmos | DreamGen |
|---|---|---|
| Role | *Builds* general world models (platform: data, tokenizer, WFMs, post-training) | *Uses* a fine-tuned world model as a synthetic **data generator** for policies |
| Output that matters | Video realism, 3D/physics consistency, controllability | Downstream policy success on new behaviours / environments |
| Actions | Optional conditioning input (action-conditioned post-training) | Not generated → recovered afterwards with IDM / LAPA |
| Evaluation | Tokenizer PSNR/SSIM/rFVD, Sampson error, physics alignment, human preference | DreamGen Bench (IF + PA via VLM judges) + closed-loop policy success |
| Compute | ~10k H100 × 3 months pre-training | LoRA fine-tuning + massive inference (1,500 L40 × 54 h) |

DreamGen is effectively a *post-training application* of the Cosmos vision: Cosmos (and WAN) is one of the
world models DreamGen fine-tunes and benchmarks.
