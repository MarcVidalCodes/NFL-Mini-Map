# Map video frame index -> tracking timestamp.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FPS_BROADCAST = 60000 / 1001        # 59.94005994...
FPS_NTSC_HALF = 30000 / 1001        # 29.97002997...
TRACKING_HZ = 10.0
FIRST_FRAME_INDEX = 1               # train_labels.csv is 1-based

#: Video frame on which `ball_snap` falls. CALIBRATED, not assumed:
#: `scripts/calibrate_alignment.py` sweeps this over [-24, +36] and the
#: reprojection-error curve has a clear minimum here (11.8 px, rising to
#: ~29 px at both ends). A 1-frame grid with 21 probes/clip puts the
#: parabolic minimum at 4.95 frames = 82.6 ms after the clip's first frame.
SNAP_FRAME = 4.95

#: Half the tracking sample interval. `ball_snap` is stamped on a 10 Hz tick,
#: so the true snap is within +-50 ms (+-3 frames) of it. That quantisation is
#: a property of the PLAY, not the clip, which is why a per-clip offset fitted
#: on one camera transfers to the other (see `refine_alignment`).
SNAP_QUANTISATION_S = 1.0 / TRACKING_HZ / 2

ANGLE_COLS = ("o", "dir")
LINEAR_COLS = ("x", "y", "s", "a", "dis")


@dataclass(frozen=True)
class ClipAlignment:
    """Frame <-> tracking-time mapping for one clip."""

    snap_time: pd.Timestamp
    snap_frame: float = SNAP_FRAME
    fps: float = FPS_BROADCAST
    offset_s: float = 0.0           # optional per-clip refinement

    def frame_to_seconds(self, frame) -> np.ndarray:
        """Seconds relative to the snap. Negative = before the snap."""
        f = np.asarray(frame, dtype=float)
        return (f - self.snap_frame) / self.fps + self.offset_s

    def frame_to_time(self, frame):
        """Absolute tracking timestamp(s) for video frame index/indices."""
        secs = self.frame_to_seconds(frame)
        deltas = pd.to_timedelta(secs, unit="s")
        return self.snap_time + deltas

    def time_to_frame(self, t) -> np.ndarray:
        """Inverse: fractional video frame for a tracking timestamp."""
        secs = (pd.to_datetime(t) - self.snap_time).total_seconds() \
            if np.ndim(t) == 0 else \
            (pd.DatetimeIndex(t) - self.snap_time).total_seconds().to_numpy()
        return (np.asarray(secs) - self.offset_s) * self.fps + self.snap_frame


def alignment_for_play(play_tracking: pd.DataFrame, **kw) -> ClipAlignment:
    """Build a ClipAlignment from one play's tracking rows."""
    snap = play_tracking.loc[play_tracking["event"] == "ball_snap", "time"]
    if not len(snap):
        raise ValueError("play has no ball_snap event")
    return ClipAlignment(snap_time=pd.Timestamp(snap.iloc[0]), **kw)


def sample_tracking(
    play_tracking: pd.DataFrame,
    times,
    players: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Linearly interpolate tracking state to arbitrary `times`.

    Returns {column: array of shape (len(times), len(players))}, plus
    ``players``. Angles are interpolated through sin/cos so the 360->0 wrap is
    handled correctly; every other column is plain linear interpolation.
    Requests outside the tracking window return NaN rather than clamping - a
    silently clamped frame looks like a stationary player, which is worse than
    an obvious hole.
    """
    P = play_tracking.pivot_table(
        index="time", columns="player",
        values=[c for c in LINEAR_COLS + ANGLE_COLS if c in play_tracking.columns],
    )
    grid = (P.index - P.index[0]).total_seconds().to_numpy()
    want = (pd.DatetimeIndex(times) - P.index[0]).total_seconds().to_numpy()
    available = list(P[LINEAR_COLS[0]].columns)
    players = players or available
    idx = [available.index(p) for p in players]
    inside = (want >= grid[0]) & (want <= grid[-1])

    out: dict[str, np.ndarray] = {"players": np.array(players)}
    for col in LINEAR_COLS:
        if col not in P.columns.get_level_values(0):
            continue
        A = P[col].to_numpy()[:, idx]
        out[col] = np.where(
            inside[:, None],
            np.column_stack([np.interp(want, grid, A[:, j]) for j in range(A.shape[1])]),
            np.nan,
        )
    for col in ANGLE_COLS:
        if col not in P.columns.get_level_values(0):
            continue
        th = np.deg2rad(P[col].to_numpy()[:, idx])
        s = np.column_stack([np.interp(want, grid, np.sin(th[:, j])) for j in range(th.shape[1])])
        c = np.column_stack([np.interp(want, grid, np.cos(th[:, j])) for j in range(th.shape[1])])
        out[col] = np.where(inside[:, None], np.rad2deg(np.arctan2(s, c)) % 360.0, np.nan)
    return out


def sample_tracking_nearest(play_tracking, times, players=None) -> dict[str, np.ndarray]:
    """Nearest-neighbour sampling. Present only to quantify what it costs."""
    P = play_tracking.pivot_table(index="time", columns="player",
                                  values=[c for c in LINEAR_COLS if c in play_tracking.columns])
    grid = (P.index - P.index[0]).total_seconds().to_numpy()
    want = (pd.DatetimeIndex(times) - P.index[0]).total_seconds().to_numpy()
    available = list(P[LINEAR_COLS[0]].columns)
    players = players or available
    idx = [available.index(p) for p in players]
    k = np.clip(np.searchsorted(grid, want), 1, len(grid) - 1)
    k = np.where(np.abs(grid[k] - want) < np.abs(grid[k - 1] - want), k, k - 1)
    return {"players": np.array(players),
            **{c: P[c].to_numpy()[np.ix_(k, idx)] for c in LINEAR_COLS if c in P.columns.get_level_values(0)}}


# ---------------------------------------------------------------------------
# Validation: sweep the anchor and look for a minimum.
# ---------------------------------------------------------------------------
#
# The check that makes this trustworthy: perturb the anchor by +-N frames,
# measure how well the 22 known player<->helmet correspondences fit a
# homography at each, and plot the curve. If the anchor is right the curve
# bottoms out at zero shift. If it bottoms out somewhere else, that offset IS
# the bug, handed to you for free.

def _clip_arrays(play_tracking: pd.DataFrame, labels: pd.DataFrame):
    """Pivot tracking and labels once so a sweep can reuse them."""
    P = play_tracking.pivot_table(index="time", columns="player", values=["x", "y"])
    grid = (P.index - P.index[0]).total_seconds().to_numpy()
    X, Y = P["x"].to_numpy(), P["y"].to_numpy()
    tracked = list(P["x"].columns)

    B = labels.pivot_table(index="frame", columns="label", values=["cx", "cy"])
    frames = B.index.to_numpy()
    boxed = list(B["cx"].columns)

    shared = [p for p in tracked if p in boxed]
    ti = [tracked.index(p) for p in shared]
    bi = [boxed.index(p) for p in shared]
    return dict(grid=grid, X=X[:, ti], Y=Y[:, ti],
                frames=frames, CX=B["cx"].to_numpy()[:, bi], CY=B["cy"].to_numpy()[:, bi],
                t0=P.index[0], players=shared)


def reprojection_error(
    arrays: dict, align: ClipAlignment, *, n_probe: int = 15,
    ransac_px: float = 10.0, nearest: bool = False,
) -> float:
    """Median reprojection error (px) of a field->image homography.

    Valid because the plane through every helmet is parallel to the field, and
    the image of a plane maps to the image of a parallel plane by a homography.
    Player height variation is the noise floor, not a modelling error.
    """
    import cv2

    snap_rel = (align.snap_time - arrays["t0"]).total_seconds()
    probe = np.unique(np.linspace(0, len(arrays["frames"]) - 1, n_probe).astype(int))
    errs = []
    for f in probe:
        t = snap_rel + align.frame_to_seconds(arrays["frames"][f])
        if not (arrays["grid"][0] <= t <= arrays["grid"][-1]):
            continue
        if nearest:
            k = int(np.abs(arrays["grid"] - t).argmin())
            fx, fy = arrays["X"][k], arrays["Y"][k]
        else:
            fx = np.array([np.interp(t, arrays["grid"], arrays["X"][:, j])
                           for j in range(arrays["X"].shape[1])])
            fy = np.array([np.interp(t, arrays["grid"], arrays["Y"][:, j])
                           for j in range(arrays["Y"].shape[1])])
        ix, iy = arrays["CX"][f], arrays["CY"][f]
        ok = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(ix) & np.isfinite(iy)
        if ok.sum() < 8:
            continue
        src = np.c_[fx, fy][ok].astype(np.float64)
        dst = np.c_[ix, iy][ok].astype(np.float64)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
        if H is None:
            continue
        pr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        errs.append(float(np.median(np.hypot(*(pr - dst).T))))
    return float(np.median(errs)) if errs else np.inf


def sweep_snap_frame(arrays: dict, snap_time, candidates, **kw) -> np.ndarray:
    """Reprojection error for each candidate snap frame. Minimum wins."""
    return np.array([
        reprojection_error(arrays, ClipAlignment(snap_time=snap_time, snap_frame=c), **kw)
        for c in candidates
    ])


def parabolic_min(xs: np.ndarray, ys: np.ndarray) -> float:
    """Sub-sample minimum by fitting a parabola through the best 3 points."""
    i = int(np.nanargmin(ys))
    if i in (0, len(ys) - 1):
        return float(xs[i])
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(xs[i])
    return float(xs[i] + 0.5 * (y0 - y2) / denom * (xs[i + 1] - xs[i]))


def refine_alignment(
    arrays: dict, snap_time, *, search: np.ndarray | None = None, n_probe: int = 21,
) -> ClipAlignment:
    """Per-clip refinement of the snap frame, by minimising reprojection error.

    Worth doing, but know what it is and what it needs.

    WHAT IT CORRECTS. `ball_snap` is stamped on a 10 Hz tick, so it is up to
    +-50 ms (+-3 frames) from the true snap. Most of what this recovers is that
    quantisation error. Because the tick is shared by both cameras filming a
    play, the correction transfers between them - which is what makes it real
    rather than overfitting.

    MEASURED, held out across camera views (fit on Endzone, scored on Sideline
    and vice versa):
        global constant          11.84 px
        offset from OTHER view   11.17 px   <- honest gain, ~6%
        offset from THIS view     9.89 px   <- the overfitting ceiling

    IT NEEDS GROUND-TRUTH IDENTITIES. The objective is built from known
    player<->helmet correspondences, so this is for CONSTRUCTING TRAINING PAIRS
    from the 60 labelled plays, not for inference. At inference use the
    constant `SNAP_FRAME`.
    """
    if search is None:
        search = np.arange(-12, 13, 1.0) + SNAP_FRAME
    errs = sweep_snap_frame(arrays, snap_time, search, n_probe=n_probe)
    return ClipAlignment(snap_time=pd.Timestamp(snap_time),
                         snap_frame=parabolic_min(search, errs))
