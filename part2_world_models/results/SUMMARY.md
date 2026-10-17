# Part 2 results summary

## Cosmos Tokenizer (CV4x8x8) reconstruction on PushT

| input res | latent (C,T,H,W) | PSNR (dB) | SSIM |
|---|---|---|---|
| 96 | (16, 5, 12, 12) | 30.49 | 0.9432 |
| 128 | (16, 5, 16, 16) | 31.77 | 0.9517 |
| 192 | (16, 5, 24, 24) | 35.85 | 0.9813 |
| 256 | (16, 5, 32, 32) | 35.99 | 0.9826 |

## Inverse dynamics model (trained on 25 target demos, evaluated on 20 held-out demos)

state MAE 0.89 px, action MAE 1.69 px, action L2 2.67 px (PushT workspace = 512 px)

## World model validation (first-frame conditioned, 33 frames, held-out demos)

| model | steps | val PSNR (dB) | val SSIM |
|---|---|---|---|
| wm_ft | 3000 | 26.69 | 0.9319 |
| wm_pre | 8000 | 26.24 | 0.9255 |
| wm_scratch | 8000 | 25.71 | 0.9198 |

## Automatic video judge (DreamGen-Bench analogue)

Instruction-following proxy: final T-goal pixel coverage (real demos reach >= 0.73 in 90% of cases). Physics-alignment proxy: share of frames without rigidity / fragmentation / agent / contact violations.

| videos | coverage_start | coverage_end | shape_err | fragmented | agent_ok | telekinesis | teleport | pa_score | IF success |
|---|---|---|---|---|---|---|---|---|---|
| real demos (val) | 0.078 | 0.775 | 0.044 | 0.136 | 0.998 | 0.000 | 0.000 | 0.863 | 0.90 |
| dreams_wm_ft | 0.075 | 0.844 | 0.178 | 0.002 | 0.532 | 0.000 | 0.002 | 0.478 | 0.88 |
| dreams_wm_scratch | 0.076 | 0.828 | 0.201 | 0.015 | 0.389 | 0.001 | 0.005 | 0.328 | 0.88 |

## Diffusion Policy, closed-loop PushT evaluation (50 episodes per run, seeds 100000-100049)

| run | real demos | neural traj. | success rate [95% CI] | mean max coverage |
|---|---|---|---|---|
| dream300_only | 0 | 300 | 0.00 [0.00, 0.07] | 0.208 |
| real186_upper | 186 | 0 | 0.40 [0.28, 0.54] | 0.860 |
| real25 | 25 | 0 | 0.06 [0.02, 0.16] | 0.540 |
| real25_dream100 | 25 | 100 | 0.02 [0.00, 0.10] | 0.569 |
| real25_dream1000 | 25 | 1000 | 0.06 [0.02, 0.16] | 0.626 |
| real25_dream100_s1 | 25 | 100 | 0.06 [0.02, 0.16] | 0.532 |
| real25_dream300 | 25 | 300 | 0.08 [0.03, 0.19] | 0.649 |
| real25_dream300_s1 | 25 | 300 | 0.10 [0.04, 0.21] | 0.616 |
| real25_dream300_s3 | 25 | 300 | 0.14 [0.07, 0.26] | 0.616 |
| real25_dream300pa | 25 | 322 | 0.06 [0.02, 0.16] | 0.599 |
| real25_dream300pa_s1 | 25 | 322 | 0.04 [0.01, 0.13] | 0.508 |
| real25_s1 | 25 | 0 | 0.16 [0.08, 0.29] | 0.556 |
| real25_s2 | 25 | 0 | 0.10 [0.04, 0.21] | 0.493 |
| real25_s3 | 25 | 0 | 0.08 [0.03, 0.19] | 0.515 |
| real25_scratchdream300 | 25 | 300 | 0.10 [0.04, 0.21] | 0.573 |

### Grouped over training seeds

| config | seeds | real | neural | mean max coverage (mean ± sd over seeds) | success, pooled episodes [95% CI] |
|---|---|---|---|---|---|
| real25 | 4 | 25 | 0 | 0.526 ± 0.028 | 0.100 [0.07, 0.15] (n=200) |
| real25_dream100 | 2 | 25 | 100 | 0.550 ± 0.026 | 0.040 [0.02, 0.10] (n=100) |
| real25_dream300 | 3 | 25 | 300 | 0.627 ± 0.019 | 0.107 [0.07, 0.17] (n=150) |
| real25_dream1000 | 1 | 25 | 1000 | 0.626 ± 0.000 | 0.060 [0.02, 0.16] (n=50) |
| real25_dream300pa | 2 | 25 | 322 | 0.554 ± 0.065 | 0.050 [0.02, 0.11] (n=100) |
| real25_scratchdream300 | 1 | 25 | 300 | 0.573 ± 0.000 | 0.100 [0.04, 0.21] (n=50) |
| dream300_only | 1 | 0 | 300 | 0.208 ± 0.000 | 0.000 [0.00, 0.07] (n=50) |
| real186_upper | 1 | 186 | 0 | 0.860 ± 0.000 | 0.400 [0.28, 0.54] (n=50) |
