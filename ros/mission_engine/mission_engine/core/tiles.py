"""Shared frame-tiling grid.

Training materialization (models/yolo/datagen/materialize.py) and onboard
tiled inference must cut frames identically, so the grid geometry lives here
and both sides import it. Everything about a grid follows from three numbers:
frame extent, tile size, and minimum overlap.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def tile_grid(
    width: int, height: int, tile: int = 640, overlap: int = 192
) -> List[Tuple[int, int]]:
    """Tile origins (x0, y0), row-major, covering the frame edge to edge.

    Per axis: the fewest tile-sized windows whose pairwise overlap is at
    least `overlap`, spaced evenly with the first and last flush to the
    frame edges. Any object smaller than `overlap` along both axes is
    guaranteed to appear whole in at least one tile (the even spacing keeps
    every stride <= tile - overlap).

    An axis not longer than the tile yields a single flush window; callers
    with smaller frames pad or accept the short tile.
    """
    if not (0 <= overlap < tile):
        raise ValueError(f"need 0 <= overlap < tile, got {overlap} vs {tile}")
    if width < 1 or height < 1:
        raise ValueError(f"bad frame {width}x{height}")

    def axis(extent: int) -> List[int]:
        if extent <= tile:
            return [0]
        n = 1 + math.ceil((extent - tile) / (tile - overlap))
        span = extent - tile
        return [round(i * span / (n - 1)) for i in range(n)]

    return [(x, y) for y in axis(height) for x in axis(width)]
