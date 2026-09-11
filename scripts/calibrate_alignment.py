"""Calibrate SNAP_FRAME by sweeping it and finding the error minimum.

    .venv/bin/python scripts/calibrate_alignment.py
    .venv/bin/python scripts/calibrate_alignment.py --lo -24 --hi 36 --step 2

Done when the sweep shows a clear minimum at the chosen anchor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from align import (FPS_BROADCAST, ClipAlignment, _clip_arrays, parabolic_min,  # noqa: E402
                   reprojection_error, sweep_snap_frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=-24)
    ap.add_argument("--hi", type=float, default=36)
    ap.add_argument("--step", type=float, default=2)
    ap.add_argument("--n-probe", type=int, default=13)
    ap.add_argument("--out", default="/private/tmp/claude-501/-Users-marcvidal-Documents-Code-NFL-Mini-Map/400d40a6-0970-4ea4-a11d-6589eb8b6d8d/scratchpad/sweep.csv")
    args = ap.parse_args()

    cand = np.arange(args.lo, args.hi + args.step / 2, args.step)
    lb = pd.read_csv("data/raw/train_labels.csv")
    tr = pd.read_csv("data/raw/train_player_tracking.csv", parse_dates=["time"])
    for d in (lb, tr):
        d["play"] = d.gameKey.astype(str) + "_" + d.playID.astype(str).str.zfill(6)
    lb = lb[~lb.isSidelinePlayer].copy()
    lb["cx"] = lb.left + lb.width / 2
    lb["cy"] = lb.top + lb.height / 2

    curves, rows = [], []
    for play, T in tr.groupby("play"):
        snap = T.loc[T.event == "ball_snap", "time"]
        if not len(snap):
            continue
        snap_t = pd.Timestamp(snap.iloc[0])
        for view, L in lb[lb.play == play].groupby("view"):
            A = _clip_arrays(T, L)
            if len(A["players"]) < 10:
                continue
            e = sweep_snap_frame(A, snap_t, cand, n_probe=args.n_probe)
            curves.append(e)
            best = parabolic_min(cand, e)
            rows.append(dict(play=play, view=view, best_frame=best,
                             err_at_best=np.nanmin(e),
                             err_at_0=e[np.argmin(np.abs(cand))]))
            print(f"  {play} {view:8s} best={best:+6.2f}f  err={np.nanmin(e):5.1f}px", flush=True)

    C = np.vstack(curves)
    med = np.nanmedian(C, axis=0)
    D = pd.DataFrame(rows)
    pd.DataFrame({"snap_frame": cand, "median_err_px": med}).to_csv(args.out, index=False)
    D.to_csv(args.out.replace("sweep", "perclip"), index=False)

    star = parabolic_min(cand, med)
    print("\n=== SWEEP (median over all clips) ===")
    for c, m in zip(cand, med):
        bar = "#" * int(round(m / max(med) * 46))
        print(f"  snap_frame {c:+7.1f}  ({c/FPS_BROADCAST*1000:+7.1f} ms)  {m:6.1f}px  {bar}")
    print(f"\nglobal minimum at snap_frame = {star:.2f}  ({star/FPS_BROADCAST*1000:.1f} ms after clip start)")
    print(f"per-clip best: median {D.best_frame.median():+.2f}  IQR "
          f"{D.best_frame.quantile(.25):+.2f}..{D.best_frame.quantile(.75):+.2f}  std {D.best_frame.std():.2f}")
    P = D.pivot_table(index="play", columns="view", values="best_frame").dropna()
    d = (P.Endzone - P.Sideline).abs()
    print(f"cross-view agreement: median {d.median():.2f} frames, within 3 frames on {(d<=3).mean():.0%}")


if __name__ == "__main__":
    main()
