from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from detect_grid import detect_cells
from detect_handwriting import classify_cells
from preprocess import build_binary_mask, deskew_image, load_image, to_grayscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect handwritten move cells in chess scoresheets.")
    parser.add_argument(
        "--input-dir",
        default="dataset",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save annotated images, crops, and JSON metadata.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "annotated": output_dir / "annotated",
        "cells": output_dir / "cells",
        "json": output_dir / "json",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def draw_detections(
    image,
    detections: list[dict],
    table_bbox: tuple[int, int, int, int],
):
    annotated = image.copy()
    x, y, w, h = table_bbox
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for index, detection in enumerate(detections, start=1):
        x0, y0, x1, y1 = detection["bbox"]
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            str(index),
            (x0 + 4, max(y0 + 18, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 128, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def save_crops(image, detections: list[dict], output_dir: Path, stem: str) -> None:
    for index, detection in enumerate(detections, start=1):
        x0, y0, x1, y1 = detection["bbox"]
        crop = image[y0:y1, x0:x1]
        crop_name = f"{stem}_cell_{index:03d}_r{detection['row']}_c{detection['col']}.png"
        cv2.imwrite(str(output_dir / crop_name), crop)


def analyze_image_variant(image, skew_angle: float) -> dict:
    gray = to_grayscale(image)
    binary = build_binary_mask(gray)
    table_bbox, xs, ys, cells = detect_cells(binary)
    detections = classify_cells(binary, cells, ys)
    return {
        "image_data": image,
        "skew_angle": skew_angle,
        "table_bbox": table_bbox,
        "vertical_lines": xs,
        "horizontal_lines": ys,
        "detections": detections,
    }


def variant_score(result: dict) -> tuple[int, int, int, int]:
    vertical_score = -abs(len(result["vertical_lines"]) - 5)
    horizontal_score = -abs(len(result["horizontal_lines"]) - 31)
    detection_score = min(len(result["detections"]), 120)
    skew_penalty = -int(round(abs(result["skew_angle"]) * 10))
    return (vertical_score, horizontal_score, detection_score, skew_penalty)


def process_image(image_path: Path, output_paths: dict[str, Path]) -> dict:
    original_image = load_image(str(image_path))
    candidates = [analyze_image_variant(original_image, 0.0)]

    deskewed_image, skew_angle = deskew_image(original_image)
    if abs(skew_angle) >= 0.3:
        candidates.append(analyze_image_variant(deskewed_image, skew_angle))

    best = max(candidates, key=variant_score)
    image = best["image_data"]
    table_bbox = best["table_bbox"]
    xs = best["vertical_lines"]
    ys = best["horizontal_lines"]
    detections = best["detections"]

    annotated = draw_detections(image, detections, table_bbox)
    cv2.imwrite(str(output_paths["annotated"] / image_path.name), annotated)
    save_crops(image, detections, output_paths["cells"], image_path.stem)

    result = {
        "image": image_path.name,
        "skew_angle_degrees": round(best["skew_angle"], 4),
        "table_bbox": [table_bbox[0], table_bbox[1], table_bbox[0] + table_bbox[2], table_bbox[1] + table_bbox[3]],
        "vertical_lines": xs,
        "horizontal_lines": ys,
        "detections": detections,
    }
    json_path = output_paths["json"] / f"{image_path.stem}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_paths = ensure_output_dirs(output_dir)

    image_paths = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    summary = []
    for image_path in image_paths:
        result = process_image(image_path, output_paths)
        summary.append({"image": result["image"], "detections": len(result["detections"])})

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
