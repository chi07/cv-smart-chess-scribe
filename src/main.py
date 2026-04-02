from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detect_grid import _extract_lines, detect_cells
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
    horizontal_models: list[tuple[float, float]],
):
    annotated = image.copy()
    x, y, w, h = table_bbox
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for index, detection in enumerate(detections, start=1):
        x0, y0, x1, y1 = detection["bbox"]
        row = int(detection["row"])
        if 0 <= row < len(horizontal_models) - 1:
            top_a, top_b = horizontal_models[row]
            bot_a, bot_b = horizontal_models[row + 1]
            polygon = np.array(
                [
                    [x0, int(round(top_a * x0 + top_b))],
                    [x1, int(round(top_a * x1 + top_b))],
                    [x1, int(round(bot_a * x1 + bot_b))],
                    [x0, int(round(bot_a * x0 + bot_b))],
                ],
                dtype=np.int32,
            )
            cv2.polylines(annotated, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
            label_y = max(int(round(top_a * x0 + top_b)) + 18, 18)
        else:
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
            label_y = max(y0 + 18, 18)
        cv2.putText(
            annotated,
            str(index),
            (x0 + 4, label_y),
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
    horizontal_models = fit_horizontal_models(binary, table_bbox, ys)
    return {
        "image_data": image,
        "skew_angle": skew_angle,
        "table_bbox": table_bbox,
        "vertical_lines": xs,
        "horizontal_lines": ys,
        "horizontal_models": horizontal_models,
        "detections": detections,
    }


def fit_horizontal_models(
    binary: np.ndarray,
    table_bbox: tuple[int, int, int, int],
    ys: list[int],
) -> list[tuple[float, float]]:
    x, y, w, h = table_bbox
    roi = binary[y : y + h, x : x + w]
    horizontal_mask = _extract_lines(roi, "horizontal")
    models: list[tuple[float, float]] = []

    for yy in ys:
        center = yy - y
        top = max(center - 6, 0)
        bottom = min(center + 7, h)
        band = horizontal_mask[top:bottom, :]
        points = np.column_stack(np.where(band > 0))

        if len(points) < max(w // 8, 60):
            models.append((0.0, float(yy)))
            continue

        xs_band = points[:, 1].astype(float) + x
        ys_band = points[:, 0].astype(float) + top + y
        slope, intercept = np.polyfit(xs_band, ys_band, 1)
        models.append((float(slope), float(intercept)))

    return models


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
    horizontal_models = best["horizontal_models"]
    detections = best["detections"]

    annotated = draw_detections(image, detections, table_bbox, horizontal_models)
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
