"""
IARC Mission 10 - Minefield Pathfinder & Simulator
===================================================
Finds the best safe path through a minefield by optimizing width and length
jointly. For each safe corridor width, it finds the global shortest path and
then selects the candidate with the highest official score.

Score = 150000 * W / ( (1+B) * L * (1 + 7*A + 100*N) )

Grid: 40 cols x 150 rows, each cell = 2x2 feet (80ft x 300ft field)
Path: bottom (y=0) to top (y=149), commands = S,U,D,L,R

USAGE:
  python iarc_pathfinder.py                  # Random minefield, find best path
  python iarc_pathfinder.py --seed 42        # Specific seed
  python iarc_pathfinder.py --mines 300      # More mines
  python iarc_pathfinder.py --batch 20       # Test 20 random seeds
  python iarc_pathfinder.py --no-plot        # Skip visualization window
"""

import argparse
from collections import deque
from pathlib import Path
import random
import statistics
import time

instructions_output = "iarc_steps.txt"

# ====================================================================
# Grid constants
# ====================================================================
COLS = 40
ROWS = 150
CELL_FT = 2
FIELD_W = COLS * CELL_FT   # 80 ft
FIELD_H = ROWS * CELL_FT   # 300 ft

constants_path = Path(__file__).parent.parent / "constants/bounding_boxes.txt"

# ====================================================================
# Minefield generation
# ====================================================================

def grid_index(value, lower_bound, cell_size, cell_count):
    """Map a coordinate to a grid index, clamped to the arena boundary."""
    raw_index = int((value - lower_bound) / cell_size)
    return min(max(raw_index, 0), cell_count - 1)


def generate_minefield(mine_locations):
    with open(constants_path, "r") as f:
        bounds = [tuple(map(float, line.split())) for line in f]

    lats = [lat for lat, _ in bounds]
    lons = [lon for _, lon in bounds]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    delta_lat = (max_lat - min_lat) / COLS
    delta_lon = (max_lon - min_lon) / ROWS

    mine_locs = []

    for lat, lon in mine_locations:
        mine_lat = float(lat)
        mine_lon = float(lon)
        idx = grid_index(mine_lat, min_lat, delta_lat, COLS)
        idy = grid_index(mine_lon, min_lon, delta_lon, ROWS)
        mine_locs.append((idx, idy))

    return set(mine_locs)





def fetch_fexl_mines(seed=0.1934, num_mines=135, scale=5):
    """Fetch actual mine positions from fexl.com/iarc/draw/ by parsing the SVG."""
    import urllib.request
    import re

    url = f"https://fexl.com/iarc/draw/?num_danger={num_mines}&seed={seed}&scale={scale}"
    print(f"  Fetching mines from fexl.com (seed={seed})...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8')

    # Parse red rectangles from SVG — mines are fill="red"
    # SVG format: <rect x=360 y=1490 width=10 height=10 fill="red" />
    mines = set()
    cell_px = scale * CELL_FT  # pixels per cell
    for m in re.finditer(r'<rect x=(\d+) y=(\d+) width=\d+ height=\d+ fill="red"', html):
        svg_x = int(m.group(1))
        svg_y = int(m.group(2))
        # Convert SVG coords to grid coords
        # SVG has 30px left border, 10px top border
        # SVG y=10 is top row (y=149), y=1490 is bottom row (y=0)
        gx = (svg_x - 30) // cell_px
        gy = (ROWS - 1) - (svg_y - 10) // cell_px
        if 0 <= gx < COLS and 0 <= gy < ROWS:
            mines.add((gx, gy))

    print(f"  Got {len(mines)} mines from fexl.com")
    return mines


def compute_clearance_map(mines):
    """Compute Chebyshev distance to the nearest mine for every cell.

    Mines must already be grid cells. An empty field has no finite mine
    distance, so every cell receives a value larger than the grid dimensions.
    """
    out_of_bounds = sorted(
        (x, y)
        for x, y in mines
        if not (0 <= x < COLS and 0 <= y < ROWS)
    )
    if out_of_bounds:
        raise ValueError(f"mine cells outside the grid: {out_of_bounds}")

    if not mines:
        no_mine_distance = max(COLS, ROWS) + 1
        return {
            (x, y): no_mine_distance
            for x in range(COLS)
            for y in range(ROWS)
        }

    clearance = {mine: 0 for mine in mines}
    queue = deque(mines)

    while queue:
        cx, cy = queue.popleft()
        next_distance = clearance[(cx, cy)] + 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (cx + dx, cy + dy)
                if not (0 <= neighbor[0] < COLS and 0 <= neighbor[1] < ROWS):
                    continue
                if neighbor in clearance:
                    continue
                clearance[neighbor] = next_distance
                queue.append(neighbor)

    if len(clearance) != COLS * ROWS:
        raise AssertionError("clearance map does not cover the complete grid")

    return clearance


# ====================================================================
# Path utilities
# ====================================================================

def grid_path_to_commands(grid_path, G=0):
    """Encode a contiguous path as Fexl S,U,D,L,R commands.

    Fexl requires the first and last path moves to be upward. Invalid paths are
    rejected instead of being silently changed after scoring.
    """
    if not grid_path:
        return ""
    if len(grid_path) == 1:
        raise ValueError("path must contain at least one move")

    direction_for_delta = {
        (0, 1): "U",
        (0, -1): "D",
        (1, 0): "R",
        (-1, 0): "L",
    }
    moves = []
    for current, following in zip(grid_path, grid_path[1:]):
        delta = (following[0] - current[0], following[1] - current[1])
        direction = direction_for_delta.get(delta)
        if direction is None:
            raise ValueError(f"path contains a non-adjacent move: {current} -> {following}")
        moves.append(direction)

    if moves and (moves[0] != "U" or moves[-1] != "U"):
        raise ValueError("path must begin and end with an upward move")

    runs = []
    for direction in moves:
        if runs and runs[-1][0] == direction:
            runs[-1][1] += 1
        else:
            runs.append([direction, 1])

    start_x, _ = grid_path[0]
    commands = [f"S,{start_x},{G}"]
    commands.extend(f"{direction},{count}" for direction, count in runs)

    return '\n'.join(commands)


def compute_green_zone(path_cells, G):
    """Green zone = cells within G of blue path but not on it."""
    if G == 0:
        return set()
    blue = set(path_cells)
    green = set()
    for (px, py) in path_cells:
        for dx in range(-G, G + 1):
            for dy in range(-G, G + 1):
                c = (px + dx, py + dy)
                if 0 <= c[0] < COLS and 0 <= c[1] < ROWS and c not in blue:
                    green.add(c)
    return green


def official_score(width_ft, length_ft, missed_mines=0, scan_time=7, overweight=0):
    """Calculate the official IARC score for a surviving path."""
    denominator = (
        (1 + missed_mines)
        * length_ft
        * (1 + 7 * scan_time + 100 * overweight)
    )
    return 150000 * width_ft / denominator if denominator > 0 else 0.0


def score_path(path_cells, G, mines, scan_time_A=7, overweight_N=0):
    """Describe and score a path using the official IARC formula."""
    blue = set(path_cells)
    green = compute_green_zone(path_cells, G)
    on_path = blue & mines
    in_green = green & mines

    B = len(in_green)
    L = len(path_cells) * CELL_FT
    W = (1 + 2 * G) * CELL_FT

    if on_path:
        score = 0.0
    else:
        score = official_score(
            W,
            L,
            missed_mines=B,
            scan_time=scan_time_A,
            overweight=overweight_N,
        )

    return {
        'score': score, 'path_length_ft': L, 'path_width_ft': W,
        'path_cells': len(path_cells), 'mines_on_path': len(on_path),
        'mines_in_green': B, 'scan_time': scan_time_A, 'dead': len(on_path) > 0,
    }


# ====================================================================
# Find best path
# ====================================================================

def shortest_safe_path(clearance_map, G):
    """Find the shortest bottom-to-top path for a fixed safe half-width.

    Every path cell must be more than ``G`` cells from a mine, and at least
    ``G`` cells from either side of the field.  All valid entrance cells are
    BFS sources, so the first top-row cell reached is a shortest path for this
    width. The fixed source and neighbor order makes equal-length ties
    deterministic.
    """
    min_x = G
    max_x = COLS - G - 1
    if min_x > max_x:
        return None

    starts = [
        (x, 0)
        for x in range(min_x, max_x + 1)
        if clearance_map[(x, 0)] > G
    ]
    if not starts:
        return None

    queue = deque(starts)
    came_from = {start: None for start in starts}

    while queue:
        current = queue.popleft()
        cx, cy = current
        if cy == ROWS - 1:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in ((0, 1), (-1, 0), (1, 0), (0, -1)):
            neighbor = (cx + dx, cy + dy)
            nx, ny = neighbor
            if not (min_x <= nx <= max_x and 0 <= ny < ROWS):
                continue
            if neighbor in came_from or clearance_map[neighbor] <= G:
                continue
            came_from[neighbor] = current
            queue.append(neighbor)

    return None


def find_best_path(mines, scan_time=7):
    """
    Optimize W/L jointly over all zero-miss corridor widths.

    For each possible ``G``, find the global shortest safe path.  Comparing the
    resulting official scores finds the best width/length tradeoff directly.
    The grid admits only 20 widths, so the search has a fixed upper bound.
    """
    clearance_map = compute_clearance_map(mines)

    best_score = 0
    best_path = None
    best_G = 0
    best_clearance = 0

    for G in range((COLS - 1) // 2 + 1):
        path = shortest_safe_path(clearance_map, G)
        if path is None:
            continue

        path_length_ft = len(path) * CELL_FT
        path_width_ft = (1 + 2 * G) * CELL_FT
        candidate_score = official_score(
            path_width_ft,
            path_length_ft,
            scan_time=scan_time,
        )

        if candidate_score > best_score:
            best_score = candidate_score
            best_path = path
            best_G = G
            best_clearance = min(clearance_map[cell] for cell in path)

    return best_path, best_G, best_clearance


# ====================================================================
# Visualization
# ====================================================================

def visualize(mines, path_cells, G, result, title="IARC Pathfinder", save_path='iarc_result.png'):
    """Draw the minefield grid with path overlay."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 18))
    blue = set(path_cells)
    green = compute_green_zone(path_cells, G)

    for x in range(COLS):
        for y in range(ROWS):
            c = (x, y)
            if c in mines and c in blue:      color = 'orange'
            elif c in blue:                    color = '#4488ff'
            elif c in mines and c in green:    color = 'yellow'
            elif c in green:                   color = 'lightgreen'
            elif c in mines:                   color = 'red'
            else:                              color = 'white'
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color,
                                        edgecolor='gray', linewidth=0.1))

    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect('equal')
    ax.set_xlabel('Column (x)')
    ax.set_ylabel('Row (y)')
    s = "DEAD" if result['dead'] else f"{result['score']:.3f}"
    ax.set_title(f"{title}\nScore: {s} | L={result['path_length_ft']}ft | "
                 f"W={result['path_width_ft']}ft | Clearance={G+1} cells")
    ax.legend(handles=[
        mpatches.Patch(color='#4488ff', label='Path'),
        mpatches.Patch(color='lightgreen', label='Green zone (safe)'),
        mpatches.Patch(color='red', label='Mine'),
        mpatches.Patch(color='orange', label='Mine on path (DEAD)'),
        mpatches.Patch(color='yellow', label='Missed mine'),
    ], loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    return fig


# ====================================================================
# Main
# ====================================================================

def run_iarc_pathfinder(mine_locations):
    parser = argparse.ArgumentParser(description='IARC Mission 10 Pathfinder')
    parser.add_argument('--seed', type=float, default=None, help='Random seed (default: random)')
    parser.add_argument('--mines', type=int, default=135, help='Number of mines (default: 135)')
    parser.add_argument('--scan-time', type=int, default=7, help='Scan time in minutes (default: 7)')
    parser.add_argument('--fexl', action='store_true', default=False,
                        help='Fetch mines from fexl.com instead of local RNG (use with --seed)')
    parser.add_argument('--batch', type=int, metavar='N', help='Run N random seeds')
    parser.add_argument('--no-plot', action='store_true', help='Skip visualization')
    args = parser.parse_args()

    if args.seed is None:
        args.seed = round(random.random() * 10000, 4)

    # --- Batch mode ---
    if args.batch:
        print(f"Batch: {args.batch} seeds | {args.mines} mines | A={args.scan_time}")
        print(f"{'Seed':<10} {'Score':<10} {'Length':<8} {'Width':<8} {'Clear':<6} {'G':<4}")
        print("-" * 50)

        scores = []
        for i in range(args.batch):
            seed = round(random.random() * 10000, 4)
            mines = generate_minefield(mine_locations)

            t0 = time.time()
            path, G, clearance = find_best_path(mines, args.scan_time)
            dt = time.time() - t0

            if path:
                r = score_path(path, G, mines, args.scan_time)
                if not r['dead']:
                    scores.append(r['score'])
                    print(f"{seed:<10} {r['score']:<10.3f} {r['path_length_ft']:<8} "
                          f"{r['path_width_ft']:<8} {clearance:<6} {G:<4}")
                else:
                    print(f"{seed:<10} DEAD")
            else:
                print(f"{seed:<10} NO PATH")

        if scores:
            print(f"\n{'=' * 50}")
            print(f"  Avg score:  {statistics.fmean(scores):.3f}")
            print(f"  Best:       {max(scores):.3f}")
            print(f"  Worst:      {min(scores):.3f}")
            print(f"  Alive:      {len(scores)}/{args.batch}")
        return

    # --- Single run ---
    if args.fexl:
        mines = fetch_fexl_mines(seed=args.seed, num_mines=args.mines)
    else:
        mines = generate_minefield(mine_locations)
    print(f"IARC Pathfinder | {len(mines)} mines | seed={args.seed} | A={args.scan_time}"
          + (" | SOURCE: fexl.com" if args.fexl else ""))

    t0 = time.time()
    path, G, clearance = find_best_path(mines, args.scan_time)
    dt = time.time() - t0

    if path:
        result = score_path(path, G, mines, args.scan_time)
        s = "DEAD" if result['dead'] else f"{result['score']:.3f}"
        print(f"\n  Score:        {s}")
        print(f"  Path length:  {result['path_length_ft']} ft ({result['path_cells']} cells)")
        print(f"  Path width:   {result['path_width_ft']} ft (G={G})")
        print(f"  Clearance:    {clearance} cells ({clearance * CELL_FT} ft)")
        print(f"  Mines on path:{result['mines_on_path']}")
        print(f"  Missed mines: {result['mines_in_green']}")
        print(f"  Compute time: {dt:.2f}s")

        commands = grid_path_to_commands(path, G)
        # print(f"\nPath commands (paste into https://fexl.com/iarc/draw/):\n{commands}")
        with open(instructions_output, "w") as f:
            f.write(f"{commands}")

        if not args.no_plot:
            visualize(
                mines,
                path,
                G,
                result,
                f"seed={args.seed} | {args.mines} mines",
                save_path="iarc_result.png",
            )
    else:
        print("  No path found!")