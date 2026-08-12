"""Extract exact indexed CM2 frames and seed editable Qwen label proposals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from audit.labels import freeze_roles, sha256, validate_labels, write_labels
from tools.extract_cm2_review import image_from_message, stamp_ns

BAG_DIRECTORIES = {
    "manual": "manual_survey",
    "return": "return_failure",
    "petal": "petal_qual",
}


def extract_bag(
    bag: str,
    records: list[dict[str, Any]],
    archive: Path,
    output: Path,
) -> list[dict[str, Any]]:
    prepared = archive / "bags" / BAG_DIRECTORIES[bag]
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    with (prepared / "frames.csv").open(newline="") as stream:
        frames = {int(row["frame"]): row for row in csv.DictReader(stream)}
    selected = {int(record["frame"]): record for record in records}
    extracted = []
    frame_index = 0
    for source in prepared_manifest["sources"]:
        source_path = Path(source["path"])
        if sha256(source_path) != source["sha256"]:
            raise ValueError(f"source hash changed: {source_path}")
        with source_path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, _, message, msg in reader.iter_decoded_messages(
                topics=["/camera_down/image_raw"]
            ):
                if frame_index in selected:
                    expected = frames[frame_index]
                    timestamp_ns = stamp_ns(msg.header)
                    if timestamp_ns != int(expected["camera_timestamp_ns"]):
                        raise ValueError(
                            f"{bag} frame {frame_index}: timestamp does not match prepared index"
                        )
                    sample_id = f"{bag}_f{frame_index:04d}"
                    destination = output / "raw" / f"{sample_id}.png"
                    image = image_from_message(msg)
                    image.save(destination, compress_level=3)
                    extracted.append(
                        {
                            "id": sample_id,
                            "bag": bag,
                            "frame": frame_index,
                            "purpose": selected[frame_index]["purpose"],
                            "source_mcap": str(source_path),
                            "source_mcap_sha256": source["sha256"],
                            "source_split_log_time_ns": int(message.log_time),
                            "camera_timestamp_ns": timestamp_ns,
                            "camera_time_s": float(expected["camera_time_s"]),
                            "range_m": float(expected["range_current_distance"]),
                            "file": str(destination.relative_to(output)),
                            "sha256": sha256(destination),
                            "width": int(msg.width),
                            "height": int(msg.height),
                            "encoding": str(msg.encoding).lower(),
                        }
                    )
                frame_index += 1
    missing = sorted(set(selected) - {record["frame"] for record in extracted})
    if missing:
        raise ValueError(f"{bag}: selected frames were not found: {missing}")
    return extracted


def qwen_rows(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text())
    return {row["id"]: row for row in manifest["rows"]}


def label_document(
    records: list[dict[str, Any]],
    qwen: dict[str, dict[str, Any]],
    output: Path,
    threshold: float,
) -> dict[str, Any]:
    images = []
    for record in records:
        objects = [
            {"xyxy": prediction["xyxy"], "visibility": "unknown"}
            for prediction in qwen[record["id"]]["predictions"]
            if prediction["confidence"] >= threshold
        ]
        images.append(
            {
                "source": record["file"],
                "source_sha256": record["sha256"],
                "width": record["width"],
                "height": record["height"],
                "capture_group": f"20260809-{record['bag']}-development-v1",
                "role": "development_eval",
                "review_state": "in_progress",
                "objects": objects,
                "ignore_regions": [],
            }
        )
    document = {"schema": "mission10-yolo-real-labels/1", "images": images}
    validate_labels(document)
    return freeze_roles(document, "codex")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("qwen_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--qwen-threshold", type=float, default=0.60)
    args = parser.parse_args()
    if (args.output / "labels.json").exists():
        raise ValueError("refusing to replace an existing certification label document")
    (args.output / "raw").mkdir(parents=True, exist_ok=True)

    selections = json.loads(args.selection.read_text())
    by_bag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selections:
        by_bag[record["bag"]].append(record)
    extracted = []
    for bag in BAG_DIRECTORIES:
        extracted.extend(extract_bag(bag, by_bag[bag], args.archive, args.output))
    extracted.sort(key=lambda record: (list(BAG_DIRECTORIES).index(record["bag"]), record["frame"]))
    manifest = {
        "schema": "mission10-cm2-certification-extract/1",
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "qwen_manifest_sha256": hashlib.sha256(args.qwen_manifest.read_bytes()).hexdigest(),
        "qwen_proposal_threshold": args.qwen_threshold,
        "records": extracted,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_labels(
        args.output / "labels.json",
        label_document(extracted, qwen_rows(args.qwen_manifest), args.output, args.qwen_threshold),
    )
    print(f"extracted and proposed {len(extracted)} exact frames into {args.output}")


if __name__ == "__main__":
    main()
