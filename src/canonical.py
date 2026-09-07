# Canonical play direction + unit contract for NFL tracking data.


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3          # 53.333 yd = 160 ft, the exact NFL width
YARDS_PER_TICK_HZ = 10.0


@dataclass(frozen=True)
class PlayDirection:
    direction: int      # +1 offense attacks +x, -1 attacks -x
    offense: str        # 'H' or 'V'
    spread_off: float   # x-spread of the offense at the snap, yards
    spread_def: float
    agree: bool         # did formation and orientation agree?

    @property
    def needs_flip(self) -> bool:
        return self.direction < 0


def infer_play_direction(snap: pd.DataFrame) -> PlayDirection:
    """Infer play direction from the 22 tracking rows at `ball_snap`.

    `snap` needs columns `player`, `x`, `o`.
    """
    if len(snap) != 22:
        raise ValueError(f"expected 22 players at the snap, got {len(snap)}")

    team = snap["player"].str[0]
    spread = snap.groupby(team)["x"].std()
    centre = snap.groupby(team)["x"].mean()
    if set(spread.index) != {"H", "V"}:
        raise ValueError(f"expected H and V, got {sorted(spread.index)}")

    offense, defense = spread.idxmin(), spread.idxmax()
    d_formation = int(np.sign(centre[defense] - centre[offense]))
    d_orient = int(np.sign(np.sin(np.deg2rad(snap.loc[team == offense, "o"])).mean()))

    return PlayDirection(
        direction=d_formation or d_orient,
        offense=offense,
        spread_off=float(spread[offense]),
        spread_def=float(spread[defense]),
        agree=d_formation == d_orient,
    )


def canonicalize_tracking(df: pd.DataFrame, direction: int) -> pd.DataFrame:
    """Rotate the play 180 deg about the field centre so the offense attacks +x.

    This is a ROTATION, not a mirror: both x and y flip, and bearings gain 180.
    A mirror (x only) would preserve nothing useful - it would swap strong side
    for weak side and turn every right-handed formation into a left-handed one.
    A rotation keeps handedness intact, so 'the play went to the offense's
    right' stays true after normalising.
    """
    if direction > 0:
        return df.copy()

    out = df.copy()
    out["x"] = FIELD_LENGTH - out["x"]
    out["y"] = FIELD_WIDTH - out["y"]
    for col in ("o", "dir"):
        if col in out.columns:
            out[col] = (out[col] + 180.0) % 360.0
    return out


def canonicalize_boxes(
    boxes: pd.DataFrame, direction: int, frame_width: int = 1280
) -> pd.DataFrame:
    """Mirror image-space helmet boxes to match `canonicalize_tracking`.

    A 180 deg field rotation is equivalent to viewing from the opposite
    sideline, which to first order is a horizontal mirror of the frame. Pixels also mirrored when fed to model.
    """
    if direction > 0:
        return boxes.copy()

    out = boxes.copy()
    out["left"] = frame_width - (out["left"] + out["width"])
    return out


def play_key(df: pd.DataFrame) -> pd.Series:
    """`{gameKey}_{playID:06d}` - the join key used across every table."""
    return df["gameKey"].astype(str) + "_" + df["playID"].astype(str).str.zfill(6)


if __name__ == "__main__":
    tr = pd.read_csv("data/raw/train_player_tracking.csv", parse_dates=["time"])
    tr["play"] = play_key(tr)

    n = flipped = disagreed = 0
    checks = []
    for play, T in tr.groupby("play"):
        snap_t = T.loc[T.event == "ball_snap", "time"]
        if not len(snap_t):
            continue
        snap = T[T.time == snap_t.iloc[0]]
        if len(snap) != 22:
            continue
        pd_ = infer_play_direction(snap)
        n += 1
        flipped += pd_.needs_flip
        disagreed += not pd_.agree

        c = canonicalize_tracking(snap, pd_.direction)
        team = c["player"].str[0]
        # after canonicalising, the offense must attack +x by BOTH signals
        checks.append((
            c.loc[team != pd_.offense, "x"].mean() > c.loc[team == pd_.offense, "x"].mean(),
            np.sin(np.deg2rad(c.loc[team == pd_.offense, "o"])).mean() > 0,
        ))

    form_ok = sum(a for a, _ in checks)
    orient_ok = sum(b for _, b in checks)
    print(f"plays: {n}   flipped: {flipped} ({flipped/n:.0%})   signal disagreements: {disagreed}")
    print(f"after canonicalisation, offense attacks +x by formation : {form_ok}/{n}")
    print(f"after canonicalisation, offense attacks +x by orientation: {orient_ok}/{n}")
    assert form_ok == n == orient_ok, "canonicalisation failed"

    # round-trip: flipping twice is the identity
    s = tr.head(500)
    rt = canonicalize_tracking(canonicalize_tracking(s, -1), -1)
    for col in ("x", "y", "o", "dir"):
        assert np.allclose(rt[col], s[col], atol=1e-9), col
    print("round-trip (flip twice == identity): ok")
    print("canonical.py checks passed")
