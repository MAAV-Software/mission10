"""
IARC Mission 10 - Minefield Pathfinder & Simulator
===================================================
Finds the WIDEST safe path through a minefield using max-clearance pathfinding.
Maximizes minimum distance from mines along the entire path, then uses that
clearance as the green zone width for maximum score.

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

import numpy as np
import heapq
import time
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

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
mines_path = "all_results.csv"

MIN_LON = None
MIN_LAT = None
MAX_LON = None
MAX_LAT = None

lats = []
lons = []


# ====================================================================
# Minefield generation
# ====================================================================

def generate_minefield():
    with open(constants_path, "r") as f:
        for line in f:
            line_contents = line.strip().split()
            lats.append(float(line_contents[0]))
            lons.append(float(line_contents[1]))
    
    MIN_LON = min(lons)
    MAX_LON = max(lons)
    MIN_LAT = min(lats)
    MAX_LAT = max(lats)

    LAT_DIST = MAX_LAT - MIN_LAT
    LON_DIST = MAX_LON - MIN_LON

    delta_lat = LAT_DIST / COLS
    delta_lon = LON_DIST / ROWS

    mine_locs = []

    with open(mines_path, "r") as f:
        for line in f:
            line_contents = line.strip().split(',')
            mine_lat = float(line_contents[0])
            mine_lon = float(line_contents[1])

            idx = int((mine_lat - MIN_LAT) / delta_lat)
            idy = int((mine_lon - MIN_LON) / delta_lon)
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
    """Compute Chebyshev distance to nearest mine for every cell."""
    clearance = {}
    for x in range(COLS):
        for y in range(ROWS):
            if (x, y) in mines:
                clearance[(x, y)] = 0
            else:
                min_d = min(max(abs(x - mx), abs(y - my)) for mx, my in mines)
                clearance[(x, y)] = min_d
    return clearance


# ====================================================================
# Max-clearance pathfinder
# ====================================================================

def max_clearance_path(mines, start, end, clearance_map):
    """
    Find the path that maximizes the MINIMUM clearance from any mine.
    Among paths with equal clearance, picks the shortest.
    This gives the widest possible safe corridor through the minefield.
    """
    if clearance_map[start] == 0 or clearance_map[end] == 0:
        return None, 0

    # Priority: (-min_clearance_along_path, path_length, counter, cell)
    # Negative because heapq is a min-heap but we want max clearance
    open_set = [(-clearance_map[start], 0, 0, start)]
    best = {}  # cell -> (min_clearance, path_length)
    came_from = {}
    counter = 1

    while open_set:
        neg_mc, plen, _, current = heapq.heappop(open_set)
        mc = -neg_mc

        if current in best:
            bmc, blen = best[current]
            if mc < bmc or (mc == bmc and plen >= blen):
                continue
        best[current] = (mc, plen)

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, mc

        cx, cy = current
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            nb = (nx, ny)
            if not (0 <= nx < COLS and 0 <= ny < ROWS):
                continue
            if clearance_map[nb] == 0:
                continue

            new_mc = min(mc, clearance_map[nb])
            new_len = plen + 1

            if nb not in best or new_mc > best[nb][0] or \
               (new_mc == best[nb][0] and new_len < best[nb][1]):
                came_from[nb] = current
                heapq.heappush(open_set, (-new_mc, new_len, counter, nb))
                counter += 1

    return None, 0


# ====================================================================
# Path utilities
# ====================================================================

def grid_path_to_commands(grid_path, G=0):
    """Convert grid cell list to S,U,D,L,R command string.
    Fexl requires: first move must be U, last move must cross finish line (U to y=150)."""
    if not grid_path:
        return ""
    sx, sy = grid_path[0]
    commands = [f"S,{sx},{G}"]

    # Build direction commands
    i = 1
    while i < len(grid_path):
        cx, cy = grid_path[i - 1]
        nx, ny = grid_path[i]
        dx, dy = nx - cx, ny - cy
        direction = {(0, 1): 'U', (0, -1): 'D', (1, 0): 'R', (-1, 0): 'L'}.get((dx, dy))
        if not direction:
            i += 1
            continue
        count = 1
        while i + count < len(grid_path):
            px, py = grid_path[i + count - 1]
            qx, qy = grid_path[i + count]
            if (qx - px, qy - py) == (dx, dy):
                count += 1
            else:
                break
        commands.append(f"{direction},{count}")
        i += count

    # Fexl requires: first move after S must be U (enter from south edge)
    if len(commands) > 1 and not commands[1].startswith('U,'):
        commands.insert(1, "U,1")

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


def score_path(path_cells, G, mines, scan_time_A=7, overweight_N=0):
    """Score using official IARC formula."""
    blue = set(path_cells)
    green = compute_green_zone(path_cells, G)
    on_path = blue & mines
    in_green = green & mines

    B = len(in_green)
    L = len(path_cells) * CELL_FT
    W = (1 + 2 * G) * CELL_FT

    if len(on_path) > 0:
        score = 0.0
    else:
        denom = (1 + B) * L * (1 + 7 * scan_time_A + 100 * overweight_N)
        score = 150000 * W / denom if denom > 0 else 0.0

    return {
        'score': score, 'path_length_ft': L, 'path_width_ft': W,
        'path_cells': len(path_cells), 'mines_on_path': len(on_path),
        'mines_in_green': B, 'scan_time': scan_time_A, 'dead': len(on_path) > 0,
    }


# ====================================================================
# Find best path
# ====================================================================

def path_respects_edge_margin(path, G):
    """Check that every cell on the path is at least G cells from left/right edges.
    Fexl allows green zone to extend past top/bottom, only x matters."""
    for (px, py) in path:
        if px - G < 0 or px + G >= COLS:
            return False
    return True


def find_best_path(mines, scan_time=7):
    """
    Try all start/end column combos with max-clearance pathfinding.
    For each path, set G = clearance - 1 so green zone has 0 missed mines.
    Ensures the green zone stays within the grid (path stays G cells from edges).
    Returns the path with the highest score.
    """
    clearance_map = compute_clearance_map(mines)

    best_score = 0
    best_path = None
    best_G = 0
    best_clearance = 0

    for sc in range(COLS):
        if (sc, 0) in mines:
            continue
        for ec in range(max(0, sc - 10), min(COLS, sc + 11)):
            if (ec, ROWS - 1) in mines:
                continue

            path, mc = max_clearance_path(mines, (sc, 0), (ec, ROWS - 1), clearance_map)
            if path is None:
                continue

            # Set G to max safe value (clearance - 1 = no mines in green zone)
            safe_G = max(0, mc - 1)

            # Try G values from largest to smallest
            for try_G in range(safe_G, -1, -1):
                # Green zone must not go past left/right edges
                if not path_respects_edge_margin(path, try_G):
                    continue
                r = score_path(path, try_G, mines, scan_time_A=scan_time)
                if not r['dead'] and r['score'] > best_score:
                    best_score = r['score']
                    best_path = path
                    best_G = try_G
                    best_clearance = mc
                break  # Largest valid G is best, no need to try smaller

    return best_path, best_G, best_clearance


# ====================================================================
# Visualization
# ====================================================================

def visualize(mines, path_cells, G, result, title="IARC Pathfinder", save_path='iarc_result.png'):
    """Draw the minefield grid with path overlay."""
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

def main():
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
        args.seed = round(np.random.random() * 10000, 4)

    # --- Batch mode ---
    if args.batch:
        print(f"Batch: {args.batch} seeds | {args.mines} mines | A={args.scan_time}")
        print(f"{'Seed':<10} {'Score':<10} {'Length':<8} {'Width':<8} {'Clear':<6} {'G':<4}")
        print("-" * 50)

        scores = []
        for i in range(args.batch):
            seed = round(np.random.random() * 10000, 4)
            mines = generate_minefield()

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
            print(f"  Avg score:  {np.mean(scores):.3f}")
            print(f"  Best:       {max(scores):.3f}")
            print(f"  Worst:      {min(scores):.3f}")
            print(f"  Alive:      {len(scores)}/{args.batch}")
        return

    # --- Single run ---
    if args.fexl:
        mines = fetch_fexl_mines(seed=args.seed, num_mines=args.mines)
    else:
        mines = generate_minefield()
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
            maav_IARCM10_compweb_dir = Path(__file__).parent.parent.parent
            visualize(mines, path, G, result, f"seed={args.seed} | {args.mines} mines", save_path=f"{maav_IARCM10_compweb_dir}/maav-IARCM10-compweb/compweb/static/images/iarc_result.png")
    else:
        print("  No path found!")


if __name__ == "__main__":
    main()
