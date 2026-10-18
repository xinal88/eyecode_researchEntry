"""Write data/splits.json: the target / val / pretrain episode split of lerobot/pusht (206 demos).

  target   25 demos  (the only real data the policy, IDM and world-model fine-tune see)
  val      20 demos  (world-model / IDM metrics only)
  pretrain 161 demos (stand-in for web-scale pre-training)

The split is a seeded permutation; it reproduces the logged sizes of the original runs exactly
(target 2860 transitions in logs/policy_real25.log, val 2547 IDM samples in logs/idm_target.log).

  python part2_world_models/make_splits.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    eps = pd.read_parquet(ROOT / "data/lerobot_pusht/meta/episodes/chunk-000/file-000.parquet")
    lengths = (eps["dataset_to_index"] - eps["dataset_from_index"]).to_numpy()
    p = np.random.default_rng(0).permutation(len(eps))
    # sorted: verified against the original run (the val clips stored in data/latents_192_val.pt are
    # reproduced bit-exactly with sorted episode order, not with permutation order)
    splits = {"target": sorted(p[:25].tolist()), "val": sorted(p[25:45].tolist()),
              "pretrain": sorted(p[45:].tolist())}
    assert lengths[splits["target"]].sum() == 2860 and lengths[splits["val"]].sum() == 2547
    json.dump(splits, open(ROOT / "data/splits.json", "w"), indent=1)
    print({k: (len(v), int(lengths[v].sum())) for k, v in splits.items()})


if __name__ == "__main__":
    main()
