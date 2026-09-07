"""Plot one tracking frame on a field. Phase 0.2 sanity check.

    .venv/bin/python scripts/plot_frame.py                       # first play, at the snap
    .venv/bin/python scripts/plot_frame.py --play 57583_000082 --event ball_snap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from field import plot_tracking_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/train_player_tracking.csv")
    ap.add_argument("--play", help="{gameKey}_{playID:06d}, e.g. 57583_000082")
    ap.add_argument("--event", default="ball_snap", help="tracking event to render")
    ap.add_argument("--out", default="notes/figures/frame.png")
    args = ap.parse_args()

    tr = pd.read_csv(args.csv, parse_dates=["time"])
    tr["play"] = tr.gameKey.astype(str) + "_" + tr.playID.astype(str).str.zfill(6)
    play = args.play or tr.play.iloc[0]
    T = tr[tr.play == play]
    if T.empty:
        raise SystemExit(f"no such play: {play}")

    hit = T.loc[T.event == args.event, "time"]
    t = hit.iloc[0] if len(hit) else T.time.iloc[len(T) // 2]
    frame = T[T.time == t]

    assert len(frame) == 22, f"expected 22 players, got {len(frame)}"

    plot_tracking_frame(
        frame, title=f"{play}  |  {args.event}  |  {t:%H:%M:%S.%f}"[:-4] + " UTC"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"{play}: {len(frame)} players at {t}")
    print(f"  x {frame.x.min():.1f}..{frame.x.max():.1f}   y {frame.y.min():.1f}..{frame.y.max():.1f}")
    print(f"  line of scrimmage ~ x={frame.x.median():.1f} "
          f"(the {50 - abs(frame.x.median() - 60):.0f} yard line)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
