# Resolve which side of the field a clip's camera is on.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIELD_CENTRE = (60.0, 160.0 / 6)        # (120/2, (160/3)/2)
CANDIDATE_ROTATIONS = (0, 90, 180, 270)


def rotate_field(x, y, degrees: float, centre=FIELD_CENTRE):
    """Rotate field coordinates about the field centre.

    Standard 2D rotation matrix, applied about the centre rather than the
    origin so the field maps onto itself for 0/180 (and onto its transpose
    for 90/270):

        [x']   [cos t  -sin t] [x - cx]   [cx]
        [y'] = [sin t   cos t] [y - cy] + [cy]
    """
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    dx, dy = np.asarray(x) - centre[0], np.asarray(y) - centre[1]
    return c * dx - s * dy + centre[0], s * dx + c * dy + centre[1]


@dataclass(frozen=True)
class CameraSide:
    rotation: int          # degrees, one of CANDIDATE_ROTATIONS
    score: float           # winning score
    margin: float          # gap to the runner-up; small margin = unreliable
    corr_xu: float
    corr_yv: float

    @property
    def confident(self) -> bool:
        return self.margin > 0.25


def _score(fx, fy, u, v) -> tuple[float, float, float]:
    """Score one candidate. Wants +corr(x,u) and -corr(y,v)."""
    ok = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 20:
        return -np.inf, np.nan, np.nan
    cxu = float(np.corrcoef(fx[ok], u[ok])[0, 1])
    cyv = float(np.corrcoef(fy[ok], v[ok])[0, 1])
    return cxu - cyv, cxu, cyv


#: The declared view constrains the answer: a sideline camera looks across the
#: field's short axis, an endzone camera down its long axis. Measured over 120
#: clips, the unconstrained search recovers the declared view on 119 of them -
#: so the constraint costs nothing and repairs the one low-margin failure.
ROTATIONS_BY_VIEW = {"Sideline": (0, 180), "Endzone": (90, 270)}


def resolve_camera_side(field_x, field_y, img_u, img_v, *, allowed=None) -> CameraSide:
    """Brute-force the rotations and return the best.

    Inputs are flat arrays of matched field/image positions pooled over frames.
    `allowed` restricts the candidate set - pass ROTATIONS_BY_VIEW[view] when
    the view is known, which it always is in this dataset.
    """
    results = []
    for deg in (allowed or CANDIDATE_ROTATIONS):
        rx, ry = rotate_field(field_x, field_y, deg)
        s, cxu, cyv = _score(rx, ry, img_u, img_v)
        results.append((s, deg, cxu, cyv))
    results.sort(reverse=True)
    (s0, d0, cxu, cyv), (s1, *_) = results[0], results[1]
    return CameraSide(rotation=d0, score=s0, margin=float(s0 - s1),
                      corr_xu=cxu, corr_yv=cyv)
