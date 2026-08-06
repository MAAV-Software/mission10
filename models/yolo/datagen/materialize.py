"""Materialize training sets: analytic labels + measured occlusion + tiling.

The renderer measures per-mine occlusion (ray-cast visible fraction) into
out/occlusion/*.json; the label files stay purely analytic. This step
recomputes each station's geometry through the deterministic pipeline —
which also recovers box-to-mine identity, since label lines carry none —
and applies the visibility policy:

  A box survives when occlusion_frac x clip_frac >= --min-frac. The production
  default is 40%: severely cropped or grass-hidden mines are not useful
  supervision, and the overlapping tile grid supplies a better view. A crop
  containing 15--40% is skipped rather than emitted as a false negative.

Two products, both under out/:

  labels_filtered/   full-frame labels with buried boxes dropped (default)
  train/             with --tiles: 640px tile crops + labels cut on the
                     shared mission_engine tile grid — the same grid onboard
                     tiled inference uses, jittered per frame for training
                     diversity. A tile a visible mine crosses without
                     keeping a label is poisoned (unlabeled mine pixels
                     teach suppression) and is skipped; empty tiles are
                     subsampled. Production emits only tiles because onboard
                     inference always uses the same tiled geometry.

Thresholds and tile knobs live here rather than at generation, so retuning
never needs a re-render. Tile image cropping needs Pillow (RunPod has it);
--no-images materializes labels + the tile index only.

    python3 -m datagen.materialize --out /path/to/ds --tiles
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

from mission_engine.core.tiles import tile_grid

from .config import GenConfig
from .labels import box_from_extents, raw_extents, yolo_box
from .manifest import OCCLUSION_SCHEMA, SCHEMA
from .scene import build_scene, image_stem

FULLFRAME_MODEL_PX = 640  # untiled inference letterboxes the frame to this
DEFAULT_MIN_FRAC = 0.40
POISON_MIN_FRAC = 0.15


@dataclass(frozen=True)
class TileParams:
    tile: int = 640
    overlap: int = 192  # > the largest projected mine (~165 px at 1 m alt)
    empty_keep: float = 0.03
    fullframe_frac: float = 0.0
    images: bool = True

    def __post_init__(self) -> None:
        if self.tile < 1 or not 0 <= self.overlap < self.tile:
            raise ValueError(
                f"bad tile geometry tile={self.tile}, overlap={self.overlap}"
            )
        for name in ("empty_keep", "fullframe_frac"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"bad {name} {value}")


def _scene_context(out: Path, occ: dict):
    """Load the manifest-authoritative config and validate sparse stations."""
    if occ.get("schema") != OCCLUSION_SCHEMA:
        raise ValueError(
            f"expected {OCCLUSION_SCHEMA}, got {occ.get('schema')!r}"
        )
    seed = occ.get("seed")
    scene_index = occ.get("scene")
    manifest_path = out / f"{seed}_s{scene_index:04d}.manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing scene manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}, got {manifest.get('schema')!r}")
    if manifest.get("seed") != seed or manifest.get("scene") != scene_index:
        raise ValueError("occlusion sidecar and manifest identify different scenes")
    cfg = GenConfig.from_dict(manifest["config"])
    if cfg.seed != seed:
        raise ValueError("manifest config seed does not match dataset identity")
    scene = build_scene(cfg, scene_index)
    station_indices = occ.get("station_indices")
    if not isinstance(station_indices, list) or any(
        not isinstance(k, int) for k in station_indices
    ):
        raise ValueError("occlusion station_indices must be a list of integers")
    if station_indices != sorted(set(station_indices)):
        raise ValueError("occlusion station_indices must be sorted and unique")
    if any(k < 0 or k >= len(scene.stations) for k in station_indices):
        raise ValueError("occlusion station index outside the generated path")
    manifest_indices = [
        station.get("station_index") for station in manifest.get("stations", [])
    ]
    if station_indices != manifest_indices:
        raise ValueError("occlusion and manifest station selections differ")
    expected_stems = {
        image_stem(cfg, scene, k) for k in station_indices
    }
    visible = occ.get("visible_frac")
    if not isinstance(visible, dict) or set(visible) != expected_stems:
        raise ValueError("occlusion entries do not exactly match selected stations")
    expected_mines = {str(i) for i in range(len(scene.mines))}
    for stem, fracs in visible.items():
        if not isinstance(fracs, dict) or set(fracs) != expected_mines:
            raise ValueError(f"occlusion mine entries are incomplete for {stem}")
        if any(
            not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0
            for value in fracs.values()
        ):
            raise ValueError(f"invalid occlusion fraction for {stem}")
    return cfg, scene, station_indices


def materialize_scene(out: Path, occ: dict, min_frac: float) -> Tuple[int, int]:
    """Write filtered full-frame label files; returns (kept, dropped)."""
    cfg, scene, station_indices = _scene_context(out, occ)
    cam = replace(cfg.camera, tilt_deg=scene.tilt)
    dest = out / "labels_filtered"
    dest.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    for k in station_indices:
        st = scene.stations[k]
        stem = image_stem(cfg, scene, k)
        fracs = occ["visible_frac"][stem]
        lines = []
        for i, mine in enumerate(scene.mines):
            box = yolo_box(
                cam,
                st.pos,
                st.q,
                mine,
                cfg.mine_dims_m,
                min_visible_frac=cfg.min_visible_frac,
                min_box_px=cfg.min_box_px,
            )
            if box is None:
                continue
            if fracs.get(str(i), 1.0) * box.visible_frac < min_frac:
                dropped += 1
                continue
            kept += 1
            lines.append(box.line() + "\n")
        (dest / f"{stem}.txt").write_text("".join(lines))
    return kept, dropped


def _jittered(
    grid: List[Tuple[int, int]], w: int, h: int, tile: int, rng: random.Random
) -> List[Tuple[int, int]]:
    """Shift the shared grid by a per-frame offset, clamped in-bounds.

    Clamping can push the penultimate origin almost onto the edge anchor
    (for example y=591 and y=592). Coalesce adjacent origins closer than half
    the unshifted axis spacing; the edge anchor wins. Training tiles need
    correct labels, not complete frame coverage, and inference uses the
    unshifted grid.
    """
    jx, jy = rng.randrange(tile // 2), rng.randrange(tile // 2)

    def axis(values: List[int], offset: int, limit: int) -> List[int]:
        if len(values) < 2:
            return [min(values[0] + offset, limit)]
        spacing = min(b - a for a, b in zip(values, values[1:]))
        shifted: List[int] = []
        for value in values:
            candidate = min(value + offset, limit)
            if shifted and 2 * (candidate - shifted[-1]) < spacing:
                shifted[-1] = candidate
            elif not shifted or candidate != shifted[-1]:
                shifted.append(candidate)
        return shifted

    xs = axis(sorted({x for x, _ in grid}), jx, w - tile)
    ys = axis(sorted({y for _, y in grid}), jy, h - tile)
    return [(x, y) for y in ys for x in xs]


def _station_tiles(
    cfg: GenConfig,
    cam,
    st,
    fracs: dict,
    scene_mines,
    windows: List[Tuple[float, float, float, float]],
    min_frac: float,
) -> List[Optional[List[str]]]:
    """Label lines per window; None marks a poisoned window (a visible mine
    crosses it without keeping a label)."""
    extents = [
        raw_extents(cam, st.pos, st.q, mine, cfg.mine_dims_m)
        for mine in scene_mines
    ]
    poison_min_frac = min(min_frac, POISON_MIN_FRAC)
    out: List[Optional[List[str]]] = []
    for win in windows:
        lines: Optional[List[str]] = []
        for i, ext in enumerate(extents):
            if ext is None:
                continue
            box = box_from_extents(
                ext,
                win,
                min_visible_frac=cfg.min_visible_frac,
                min_box_px=cfg.min_box_px,
            )
            occ_f = fracs.get(str(i), 1.0)
            if box is not None and occ_f * box.visible_frac >= min_frac:
                lines.append(box.line() + "\n")
                continue
            # no surviving label: poison check on the unclipped overlap
            u0, u1, v0, v1 = ext
            x0, y0, x1, y1 = win
            inter = max(0.0, min(u1, x1) - max(u0, x0)) * max(
                0.0, min(v1, y1) - max(v0, y0)
            )
            raw = (u1 - u0) * (v1 - v0)
            if raw > 0.0 and occ_f * inter / raw >= poison_min_frac:
                lines = None  # learnable mine pixels with no label
                break
        out.append(lines)
    return out


def _crop(src_img, dest: Path, x0: int, y0: int, tile: int) -> None:
    src_img.crop((x0, y0, x0 + tile, y0 + tile)).save(dest)


def tile_scene(
    out: Path, occ: dict, min_frac: float, tp: TileParams
) -> Tuple[int, int, int]:
    """Emit training tiles for one scene; returns (tiles, poisoned, boxes)."""
    cfg, scene, station_indices = _scene_context(out, occ)
    cam = replace(cfg.camera, tilt_deg=scene.tilt)
    w, h = cam.width_px, cam.height_px
    grid = tile_grid(w, h, tp.tile, tp.overlap)
    img_dir = out / "train" / "images"
    lbl_dir = out / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if tp.images:
        try:
            from PIL import Image
        except ImportError as e:  # host python has no Pillow; RunPod does
            raise SystemExit(
                "tile image cropping needs Pillow — rerun with --no-images "
                "for labels + index only"
            ) from e
    index = []
    n_tiles = n_poisoned = n_boxes = 0
    for k in station_indices:
        st = scene.stations[k]
        stem = image_stem(cfg, scene, k)
        fracs = occ["visible_frac"][stem]
        rng = random.Random(f"{cfg.seed}:{stem}:tiles")
        origins = _jittered(grid, w, h, tp.tile, rng)
        windows = [
            (float(x), float(y), float(x + tp.tile), float(y + tp.tile))
            for x, y in origins
        ]
        per_win = _station_tiles(
            cfg, cam, st, fracs, scene.mines, windows, min_frac
        )
        src_img = None
        for (x0, y0), lines in zip(origins, per_win):
            if lines is None:
                n_poisoned += 1
                continue
            if not lines and rng.random() >= tp.empty_keep:
                continue
            name = f"{stem}_x{x0:04d}_y{y0:04d}"
            (lbl_dir / f"{name}.txt").write_text("".join(lines))
            index.append({"tile": name, "src": stem, "x0": x0, "y0": y0})
            n_tiles += 1
            n_boxes += len(lines)
            if tp.images:
                if src_img is None:
                    src_img = Image.open(out / "images" / f"{stem}.png")
                _crop(src_img, img_dir / f"{name}.png", x0, y0, tp.tile)
        # untiled-mode slice: the whole frame, letterboxed by the trainer to
        # FULLFRAME_MODEL_PX — so min_box_px must hold at that scale, not
        # native, or sub-pixel mines survive as labels
        if rng.random() < tp.fullframe_frac:
            ff_lines = []
            for i, mine in enumerate(scene.mines):
                box = yolo_box(
                    cam,
                    st.pos,
                    st.q,
                    mine,
                    cfg.mine_dims_m,
                    min_visible_frac=cfg.min_visible_frac,
                    min_box_px=cfg.min_box_px * w / FULLFRAME_MODEL_PX,
                )
                if box is not None and (
                    fracs.get(str(i), 1.0) * box.visible_frac >= min_frac
                ):
                    ff_lines.append(box.line() + "\n")
            name = f"{stem}_full"
            (lbl_dir / f"{name}.txt").write_text("".join(ff_lines))
            index.append({"tile": name, "src": stem, "x0": 0, "y0": 0, "full": True})
            n_tiles += 1
            n_boxes += len(ff_lines)
            if tp.images:
                shutil.copyfile(
                    out / "images" / f"{stem}.png", img_dir / f"{name}.png"
                )
    index_file = out / "train" / "tiles.json"
    existing = (
        json.loads(index_file.read_text()) if index_file.exists() else {}
    )
    existing[f"{cfg.seed}_s{scene.index:04d}"] = {
        "tile_px": tp.tile,
        "overlap_px": tp.overlap,
        "tiles": index,
    }
    index_file.write_text(json.dumps(existing, indent=2))
    return n_tiles, n_poisoned, n_boxes


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--min-frac",
        type=float,
        default=DEFAULT_MIN_FRAC,
        help="minimum occlusion x crop fraction for a labeled mine",
    )
    p.add_argument("--tiles", action="store_true", help="emit train/ tile set")
    p.add_argument("--tile-px", type=int, default=TileParams.tile)
    p.add_argument("--overlap-px", type=int, default=TileParams.overlap)
    p.add_argument("--empty-keep", type=float, default=TileParams.empty_keep)
    p.add_argument(
        "--fullframe-frac", type=float, default=TileParams.fullframe_frac
    )
    p.add_argument("--no-images", action="store_true")
    ns = p.parse_args(argv)
    out = Path(ns.out)
    occ_files = sorted((out / "occlusion").glob("*.json"))
    if not occ_files:
        raise SystemExit(f"no occlusion sidecars under {out / 'occlusion'}")
    kept = dropped = tiles = poisoned = boxes = 0
    tp = TileParams(
        tile=ns.tile_px,
        overlap=ns.overlap_px,
        empty_keep=ns.empty_keep,
        fullframe_frac=ns.fullframe_frac,
        images=not ns.no_images,
    )
    for f in occ_files:
        occ = json.loads(f.read_text())
        k, d = materialize_scene(out, occ, ns.min_frac)
        kept += k
        dropped += d
        if ns.tiles:
            t, px, b = tile_scene(out, occ, ns.min_frac, tp)
            tiles += t
            poisoned += px
            boxes += b
    print(
        f"full-frame: kept {kept} boxes, dropped {dropped} below {ns.min_frac}"
    )
    if ns.tiles:
        print(
            f"train/: {tiles} tiles ({boxes} boxes), "
            f"{poisoned} poisoned tiles skipped"
        )


if __name__ == "__main__":
    main()
