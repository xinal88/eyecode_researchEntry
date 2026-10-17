# EyeCode Research Entry Task — World Models

Submission for the EyeCode research entry task (topic: World Models).

| Part | What | Where |
|---|---|---|
| 1 | ROSOrin Pro Gazebo simulation: run & control the robot, LiDAR/IMU/camera visualisation, SLAM mapping, map saving, single-goal and multi-point autonomous navigation | [`part1_rosorin_sim/`](part1_rosorin_sim/README.md) |
| 2 | Paper reading (DreamGen, Cosmos) + small-scale reproduction of the DreamGen pipeline on top of the official Cosmos Tokenizer, with quantitative evaluation | [`part2_world_models/`](part2_world_models/README.md) |
| — | Technical report (PDF / DOCX) | [`report/`](report/) |

## Repository layout
```
part1_rosorin_sim/      ROS 2 Humble + Ignition Fortress workspace (official Hiwonder packages + fixes),
                        Docker / chroot runtime, demo scripts, saved map, metrics
part2_world_models/     DreamGen x Cosmos code: tokenizer eval, world model, dreams, IDM, judge, Diffusion Policy
  PAPER_NOTES.md        summaries + comparison of the two papers
  results/              SUMMARY.md, figures, judge scores, policy evaluations, example videos
report/                 technical report
data/, checkpoints/     created by the scripts (not committed: see .gitignore)
third_party/            official Hiwonder simulation package as downloaded (pristine copy for the patch diff)
requirements.txt        Python packages for Part 2
```

## Quick start
```bash
# Part 1 (needs Docker, or sudo for the chroot runtime)
cd part1_rosorin_sim && ./run.sh image && ./run.sh all
# Part 2 (Python 3.12, CUDA GPU)
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python part2_world_models/prepare_pusht.py   # then follow part2_world_models/README.md
```

## Environment used
Shared Ubuntu 24.04 server (an Incus container): 8 CPU cores, 24 GB RAM, RTX 5880 Ada 48 GB GPUs shared with
other users, downloads capped at ≈0.5–1 MB/s. Every long job ran in a named `tmux` session with its log on disk.
