"""Draw an NFL field in matplotlib and plot tracking data on it.

Coordinate system (this is the whole point of the module - get it wrong once
and every downstream number is quietly off by 10 yards):

    x: 0 .. 120 yards, along the field's long axis.
       x=0    back of the left end zone
       x=10   left goal line
       x=60   the 50 yard line          <-- NOT x=50
       x=110  right goal line
       x=120  back of the right end zone

    y: 0 .. 53.3 yards, across the field. y=0 is one sideline.

The 100 yards of playing field are the *interior*, x in [10, 110]. A yard
line's painted number is therefore `50 - abs(x - 60)`, not `x`. The tracking
CSVs use exactly this convention: observed x runs 2.94..116.39, which only
makes sense if the end zones are included.

y is allowed outside [0, 53.3]: players run out of bounds, and ~0.1% of
tracking rows are legitimately off the field. Nothing here clamps.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

FIELD_LENGTH = 120.0        # including both end zones
FIELD_WIDTH = 53.3          # 160 ft; the real value is 53.33, the data uses 53.3
END_ZONE = 10.0
GOAL_LEFT = END_ZONE        # x = 10
GOAL_RIGHT = FIELD_LENGTH - END_ZONE   # x = 110
MIDFIELD = FIELD_LENGTH / 2            # x = 60, the 50 yard line

# NFL inbound (hash) lines sit 70 ft 9 in from each sideline.
HASH_FROM_SIDELINE = 70.75 / 3
# Painted numbers sit 12 yards in from each sideline.
NUMBER_FROM_SIDELINE = 12.0

TURF = "#3b7a4a"
ENDZONE_FILL = "#2f6340"
PAINT = "white"
TEAM_COLORS = {"H": "#e8453c", "V": "#3f6fd1"}


def yard_number(x: float) -> int:
    """Painted number for the yard line at field coordinate `x`.

    >>> yard_number(60)   # midfield
    50
    >>> yard_number(20)   # 10 yard line, left side
    10
    """
    return int(round(50 - abs(x - MIDFIELD)))


def draw_field(
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] = (14, 6.5),
    show_numbers: bool = True,
    show_hashes: bool = True,
) -> Axes:
    """Render an empty, correctly proportioned NFL field. Returns the Axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.add_patch(Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, fc=TURF, zorder=0))
    for x0 in (0.0, GOAL_RIGHT):
        ax.add_patch(Rectangle((x0, 0), END_ZONE, FIELD_WIDTH, fc=ENDZONE_FILL, zorder=0.1))

    # Every 5 yards between the goal lines, plus the goal lines themselves.
    for x in np.arange(GOAL_LEFT, GOAL_RIGHT + 0.1, 5.0):
        heavy = x in (GOAL_LEFT, GOAL_RIGHT)
        ax.plot([x, x], [0, FIELD_WIDTH], color=PAINT,
                lw=2.0 if heavy else 1.0, alpha=1.0 if heavy else 0.75, zorder=1)

    # Sidelines and end lines.
    ax.add_patch(Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                           fill=False, ec=PAINT, lw=2.0, zorder=2))

    if show_hashes:
        for x in np.arange(GOAL_LEFT + 1, GOAL_RIGHT, 1.0):
            if x % 5 == 0:
                continue
            for y in (HASH_FROM_SIDELINE, FIELD_WIDTH - HASH_FROM_SIDELINE):
                ax.plot([x, x], [y - 0.33, y + 0.33], color=PAINT, lw=0.9, alpha=0.7, zorder=1)
            for y in (0, FIELD_WIDTH):                      # sideline ticks
                sign = 1 if y == 0 else -1
                ax.plot([x, x], [y, y + sign * 0.5], color=PAINT, lw=0.7, alpha=0.6, zorder=1)

    if show_numbers:
        for x in np.arange(GOAL_LEFT + 10, GOAL_RIGHT - 9, 10.0):
            label = f"{yard_number(x)}"
            ax.text(x, NUMBER_FROM_SIDELINE, label, color=PAINT, fontsize=15,
                    ha="center", va="center", alpha=0.85, zorder=1, fontweight="bold")
            ax.text(x, FIELD_WIDTH - NUMBER_FROM_SIDELINE, label, color=PAINT, fontsize=15,
                    ha="center", va="center", alpha=0.85, zorder=1, fontweight="bold",
                    rotation=180)

    ax.set_xlim(-3, FIELD_LENGTH + 3)
    ax.set_ylim(-3, FIELD_WIDTH + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_tracking_frame(
    frame,
    ax: Axes | None = None,
    *,
    label_players: bool = True,
    show_direction: bool = True,
    title: str | None = None,
) -> Axes:
    """Plot one tracking tick (<=22 rows) on a field.

    `frame` needs columns `player`, `x`, `y`; `dir` is used for heading arrows
    if present. Team comes from the first character of `player` (H/V).
    """
    ax = draw_field(ax) if ax is None else ax

    for _, r in frame.iterrows():
        team = str(r["player"])[0]
        color = TEAM_COLORS.get(team, "#dddddd")
        ax.scatter(r["x"], r["y"], s=190, c=color, ec="white", lw=1.4, zorder=4)

        if show_direction and "dir" in frame.columns and np.isfinite(r.get("dir", np.nan)):
            # `dir` is a compass bearing in degrees; convert to a field vector.
            th = np.deg2rad(r["dir"])
            ax.arrow(r["x"], r["y"], 2.2 * np.sin(th), 2.2 * np.cos(th),
                     head_width=0.7, head_length=0.7, fc=color, ec="white",
                     lw=0.6, length_includes_head=True, zorder=3)

        if label_players:
            ax.text(r["x"], r["y"], str(r["player"])[1:], color="white",
                    fontsize=6.5, ha="center", va="center", zorder=5, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=11)
    return ax


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
    # The assertion that pays for this whole module.
    assert yard_number(60) == 50, "midfield must be x=60"
    assert yard_number(50) == 40, "x=50 is the 40, not the 50"
    assert yard_number(GOAL_LEFT) == 0 and yard_number(GOAL_RIGHT) == 0
    print("field.py coordinate checks passed")
