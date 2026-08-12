"""Render combined Qwen and paired-YOLO CM2 review overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "appearance": (255, 48, 48),
    "production": (48, 200, 255),
    "qwen": (255, 220, 32),
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    label: str,
    color: tuple[int, int, int],
    label_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), outline=color, width=5)
    text_box = draw.textbbox((0, 0), label, font=label_font)
    label_y = max(3, y0 - text_box[3] - 6)
    draw.rectangle(
        (x0, label_y, x0 + text_box[2] + 8, label_y + text_box[3] + 5),
        fill=(0, 0, 0),
    )
    draw.text((x0 + 4, label_y + 2), label, fill=color, font=label_font)


def render(
    image: Image.Image,
    selection: dict[str, Any],
    qwen: dict[str, Any],
    threshold: float,
    yolo_threshold: float,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    label_font = font(22)
    for model in ("production", "appearance"):
        for item in selection["detections"].get(
            f"{model}_{yolo_threshold:.2f}", []
        ):
            detection = item["box"]
            draw_label(
                draw,
                [detection[key] for key in ("x0", "y0", "x1", "y1")],
                f"{model[0].upper()} {detection['confidence']:.2f}",
                COLORS[model],
                label_font,
            )
    for prediction in qwen["predictions"]:
        if prediction["confidence"] >= threshold:
            draw_label(
                draw,
                prediction["xyxy"],
                f"Q {prediction['confidence']:.2f}",
                COLORS["qwen"],
                label_font,
            )
    hud = (
        f"{qwen['id']}  range={selection['range_m']:.2f}m  "
        f"speed={selection['speed_mps']:.2f}m/s  A=red P=cyan Q=yellow"
    )
    hud_box = draw.textbbox((0, 0), hud, font=label_font)
    draw.rectangle((8, 8, hud_box[2] + 24, hud_box[3] + 22), fill=(0, 0, 0))
    draw.text((16, 12), hud, fill=(255, 255, 255), font=label_font)
    return output


def contact_sheets(rows: list[dict[str, Any]], output: Path) -> None:
    columns, sheet_rows = 4, 3
    thumb = (410, 308)
    label_height = 26
    per_sheet = columns * sheet_rows
    for start in range(0, len(rows), per_sheet):
        subset = rows[start : start + per_sheet]
        sheet = Image.new(
            "RGB", (columns * thumb[0], sheet_rows * (thumb[1] + label_height))
        )
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(subset):
            with Image.open(output / "overlays" / f"{row['id']}_overlay.jpg") as source:
                image = source.copy()
            image.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = (offset % columns) * thumb[0]
            y = (offset // columns) * (thumb[1] + label_height)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + thumb[1] + 4), row["id"], fill="white")
        number = start // per_sheet + 1
        sheet.save(output / f"qwen_contact_sheet_{number:02d}.jpg", quality=93)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--yolo-threshold", type=float, default=0.60)
    args = parser.parse_args()

    selections = {
        f"{row['bag']}_f{row['frame']:04d}": row
        for row in json.loads((args.review / "selection.json").read_text())
    }
    manifest = json.loads((args.qwen / "manifest.json").read_text())
    if manifest["completed_count"] != len(selections):
        raise ValueError("Qwen manifest does not cover the complete review selection")
    (args.qwen / "overlays").mkdir(exist_ok=True)
    for row in manifest["rows"]:
        source = args.review / row["image"]
        with Image.open(source) as image:
            rendered = render(
                image,
                selections[row["id"]],
                row,
                args.threshold,
                args.yolo_threshold,
            )
        rendered.save(args.qwen / "overlays" / f"{row['id']}_overlay.jpg", quality=94)
    contact_sheets(manifest["rows"], args.qwen)
    print(f"rendered {len(manifest['rows'])} combined overlays into {args.qwen}")


if __name__ == "__main__":
    main()
