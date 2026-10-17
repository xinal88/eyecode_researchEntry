"""Decode the LeRobot PushT dataset (AV1 video + parquet) into one compact .npz.

Output: data/pusht_96.npz with
  frames  (N, 96, 96, 3) uint8
  state   (N, 2) float32   agent (x, y) position in PushT pixel space [0, 512]
  action  (N, 2) float32   commanded target position in the same space
  ep_start, ep_end (E,)    frame index bounds per episode (end exclusive)
"""
import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/lerobot_pusht"))
    ap.add_argument("--out", default=str(ROOT / "data/pusht_96.npz"))
    args = ap.parse_args()
    src = Path(args.src)

    df = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
    eps = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")

    container = av.open(str(src / "videos/observation.image/chunk-000/file-000.mp4"))
    frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    print("decoded video frames", frames.shape, "parquet rows", len(df))
    assert len(frames) == len(df), "video/parquet length mismatch"

    np.savez_compressed(
        args.out,
        frames=frames,
        state=np.stack(df["observation.state"].to_numpy()).astype(np.float32),
        action=np.stack(df["action"].to_numpy()).astype(np.float32),
        ep_start=eps["dataset_from_index"].to_numpy(),
        ep_end=eps["dataset_to_index"].to_numpy(),
    )
    lengths = eps["dataset_to_index"] - eps["dataset_from_index"]
    print(f"saved {args.out}: {len(eps)} episodes, length mean {lengths.mean():.1f} "
          f"min {lengths.min()} max {lengths.max()}")


if __name__ == "__main__":
    main()
