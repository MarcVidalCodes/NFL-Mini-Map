from __future__ import annotations

import json
from dataclasses import asdict
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from align import SNAP_FRAME, ClipAlignment, sample_tracking
from camera import (ROTATIONS_BY_VIEW, CameraSide, resolve_camera_side,
                    rotate_field)
from canonical import canonicalize_tracking, infer_play_direction

DATA = Path("data/raw")
CACHE = Path("data/cache")
TRACK_COLS = ("x", "y", "s", "a", "dis", "o", "dir")


class Play:
    """One (play, view) clip: video + labels + oriented, aligned tracking."""

    def __init__(
        self,
        play_id: str,
        view: str = "Sideline",
        *,
        split: str = "train",
        data_root: Path | str = DATA,
        canonical: bool = True,
        snap_frame: float = SNAP_FRAME,
    ):
        self.play_id, self.view, self.split = play_id, view, split
        self.root = Path(data_root)
        self.canonical = canonical
        self._snap_frame = snap_frame

        game, pid = play_id.split("_")
        self.game_key, self.play_num = int(game), int(pid)
        self.video_path = self.root / split / f"{play_id}_{view}.mp4"

    # ---------------------------------------------------------------- loading

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_cached(path: str, parse_time: bool) -> pd.DataFrame:
        """train_labels.csv is 99 MB - read each table once per process."""
        kw = {"parse_dates": ["time"]} if parse_time else {}
        df = pd.read_csv(path, **kw)
        df["play"] = df.gameKey.astype(str) + "_" + df.playID.astype(str).str.zfill(6)
        return df

    @classmethod
    def _load(cls, path: Path, parse_time: bool = False) -> pd.DataFrame:
        return cls._load_cached(str(path), parse_time)

    @cached_property
    def tracking(self) -> pd.DataFrame:
        """Raw tracking for this play, canonicalised if requested."""
        src = self.root / f"{self.split}_player_tracking.csv"
        T = self._load(src, parse_time=True)
        T = T[T.play == self.play_id].copy()
        if T.empty:
            raise FileNotFoundError(f"no tracking rows for {self.play_id}")
        if self.canonical:
            T = canonicalize_tracking(T, self.direction.direction)
        return T

    @cached_property
    def labels(self) -> pd.DataFrame:
        """Labelled helmet boxes for this clip (sideline players excluded)."""
        src = self.root / f"{self.split}_labels.csv"
        if not src.exists():
            return pd.DataFrame()
        L = self._load(src)
        L = L[(L.play == self.play_id) & (L.view == self.view) & (~L.isSidelinePlayer)].copy()
        L["cx"] = L.left + L.width / 2
        L["cy"] = L.top + L.height / 2
        return L

    # ------------------------------------------------------- derived geometry

    @cached_property
    def direction(self):
        """Canonical play direction, inferred from the snap formation."""
        raw = self._load(self.root / f"{self.split}_player_tracking.csv", parse_time=True)
        raw = raw[raw.play == self.play_id]
        snap_t = raw.loc[raw.event == "ball_snap", "time"]
        if not len(snap_t):
            raise ValueError(f"{self.play_id} has no ball_snap event")
        return infer_play_direction(raw[raw.time == snap_t.iloc[0]])

    @cached_property
    def alignment(self) -> ClipAlignment:
        """Frame -> tracking-time mapping, anchored on the snap."""
        snap = self.tracking.loc[self.tracking.event == "ball_snap", "time"]
        return ClipAlignment(snap_time=pd.Timestamp(snap.iloc[0]), snap_frame=self._snap_frame)

    @cached_property
    def camera(self) -> CameraSide:
        """Which side the camera is on. Brute-forced once, then cached to disk."""
        key = f"{self.play_id}_{self.view}_{'canon' if self.canonical else 'raw'}"
        cache_file = CACHE / "camera_side.json"
        store = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        if key in store:
            return CameraSide(**store[key])

        cs = self._resolve_camera()
        store[key] = asdict(cs)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(store, indent=1, sort_keys=True))
        return cs

    def _resolve_camera(self) -> CameraSide:
        if self.labels.empty:
            raise ValueError("camera side needs labelled boxes to score against")
        L = self.labels
        frames = np.unique(np.linspace(L.frame.min(), L.frame.max(), 30).astype(int))
        samp = sample_tracking(self.tracking, self.alignment.frame_to_time(frames))
        pidx = {p: i for i, p in enumerate(samp["players"])}

        fx, fy, u, v = [], [], [], []
        for i, f in enumerate(frames):
            for _, r in L[L.frame == f].iterrows():
                j = pidx.get(r.label)
                if j is None:
                    continue
                fx.append(samp["x"][i, j]); fy.append(samp["y"][i, j])
                u.append(r.cx); v.append(r.cy)
        return resolve_camera_side(np.array(fx), np.array(fy), np.array(u), np.array(v),
                                   allowed=ROTATIONS_BY_VIEW.get(self.view))

    # ------------------------------------------------------------- the payload

    @cached_property
    def n_frames(self) -> int:
        import cv2

        if not self.video_path.exists():
            return int(self.labels.frame.max()) if not self.labels.empty else 0
        cap = cv2.VideoCapture(str(self.video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def frame_time(self, frame) -> pd.Timestamp:
        """Tracking timestamp for a video frame index."""
        return self.alignment.frame_to_time(frame)

    def tracking_at(self, frame, *, oriented: bool = True) -> pd.DataFrame:
        """Tracking rows for exactly this video frame.

        Linearly interpolated to the frame's timestamp (angles through
        sin/cos), canonicalised for play direction, and - when `oriented` -
        rotated into the camera's frame of reference.

        Returns one row per player with columns
        ``player, x, y, s, a, dis, o, dir`` and, if oriented, ``cam_x, cam_y``.
        """
        times = pd.DatetimeIndex([self.frame_time(frame)])
        samp = sample_tracking(self.tracking, times)
        out = pd.DataFrame({"player": samp["players"]})
        for c in TRACK_COLS:
            if c in samp:
                out[c] = samp[c][0]
        out["frame"] = frame
        out["time"] = times[0]

        if oriented:
            cx, cy = rotate_field(out.x.to_numpy(), out.y.to_numpy(), self.camera.rotation)
            out["cam_x"], out["cam_y"] = cx, cy
        return out

    def boxes(self, frame) -> pd.DataFrame:
        """Labelled helmet boxes on this frame."""
        return self.labels[self.labels.frame == frame].copy()

    def image(self, frame) -> np.ndarray:
        """Decode one video frame (BGR). Frame indices are 1-based."""
        import cv2

        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame) - 1))
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise IndexError(f"frame {frame} unreadable in {self.video_path.name}")
        return img

    def homography(self, frame, *, ransac_px: float = 10.0):
        """Field -> image homography for this frame, from known correspondences.

        Needs ground-truth labels, so this is for building supervision, not
        inference.
        """
        import cv2

        trk = self.tracking_at(frame).set_index("player")
        box = self.boxes(frame).set_index("label")
        shared = [p for p in trk.index if p in box.index]
        if len(shared) < 8:
            return None
        src = trk.loc[shared, ["x", "y"]].to_numpy(np.float64)
        dst = box.loc[shared, ["cx", "cy"]].to_numpy(np.float64)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
        return H

    def __repr__(self) -> str:
        return (f"Play({self.play_id!r}, {self.view!r}, frames={self.n_frames}, "
                f"offense={self.direction.offense}, rot={self.camera.rotation})")
