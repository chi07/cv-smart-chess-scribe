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


def remove_grid_lines(binary: np.ndarray) -> np.ndarray:
    """Trả về binary mask đã xóa đường kẻ ngang/dọc của bảng.

    _extract_lines dùng MORPH_OPEN với kernel dài (width//30 px), nên chỉ giữ
    lại những nét liên tục đủ dài — đường kẻ bảng. Nét chữ viết tay ngắn hơn
    nhiều nên không bị ảnh hưởng.
    """
    h_lines = _extract_lines(binary, "horizontal")
    v_lines = _extract_lines(binary, "vertical")
    grid_mask = cv2.bitwise_or(h_lines, v_lines)
    return cv2.bitwise_and(binary, cv2.bitwise_not(grid_mask))


def analyze_image_variant(image, skew_angle: float) -> dict:
    gray = to_grayscale(image)
    binary = build_binary_mask(gray)
    table_bbox, xs, ys, cells = detect_cells(binary)
    # Xóa đường kẻ bảng khỏi mask trước khi phân tích nội dung ô,
    # tránh grid line chảy vào padded ROI gây false positive.
    content_binary = remove_grid_lines(binary)
    detections = classify_cells(content_binary, cells, ys)
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
        top = max(center - 12, 0)
        bottom = min(center + 13, h)
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

def detect_line(grey, line_type="vertical"):
    if line_type == "vertical":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, grey.shape[0] // 40))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, grey.shape[0] // 5))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, grey.shape[0] // 10))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grey.shape[1] // 40, 1))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (grey.shape[1] // 5, 1))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (grey.shape[1] // 10, 1))

    lines = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, kernel_dilate)
    lines = cv2.erode(lines, kernel_erode)
    return lines

import cv2
import numpy as np

def extract_line_segments(line_mask, line_type="vertical", threshold_count=4000):
    """
    Extract line segments using best-fit line (least squares).

    Returns:
        list of (x1, y1, x2, y2)
    """
    contours, _ = cv2.findContours(
        line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    segments = []

    for contour in contours:
        if cv2.contourArea(contour) < threshold_count:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Shape filtering
        if line_type == "vertical" and h <= w:
            continue
        if line_type == "horizontal" and w <= h:
            continue

        # Fit line (vx, vy, x0, y0)
        vx, vy, x0, y0 = cv2.fitLine(
            contour, cv2.DIST_L2, 0, 0.01, 0.01
        )

        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])

        # 🔥 Project contour points onto line to find endpoints
        points = contour[:, 0, :]  # (N, 2)

        # Parametric t for projection
        t = (points[:, 0] - x0) * vx + (points[:, 1] - y0) * vy

        t_min = t.min()
        t_max = t.max()

        # Endpoints
        x1 = int(x0 + t_min * vx)
        y1 = int(y0 + t_min * vy)
        x2 = int(x0 + t_max * vx)
        y2 = int(y0 + t_max * vy)

        segments.append((x1, y1, x2, y2))

    print(f"Detected {len(segments)} line segments.")

    # Visualization
    # vis = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
    # for (x1, y1, x2, y2) in segments:
    #     cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # cv2.imshow('Line Segments', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return segments

def line_intersection(seg1, seg2):
    """
    Find intersection of two lines (not limited to segment bounds)

    seg = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2

    # Line 1: A1x + B1y = C1
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    # Line 2: A2x + B2y = C2
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1

    if abs(det) < 1e-10:
        return None  # parallel lines

    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det

    return int(x), int(y)

def get_cell_from_lines(horizontal_lines, vertical_lines):
    # Sort horizontal lines from up to down, vertical lines from left to right
    horizontal_lines = sorted(horizontal_lines, key=lambda seg: (seg[1] + seg[3]) / 2)
    vertical_lines = sorted(vertical_lines, key=lambda seg: (seg[0] + seg[2]) / 2)

    intersection_point = []
    # From left to right then up to down

    for h in horizontal_lines:
        for v in vertical_lines:
            point = line_intersection(h, v)
            if point is not None:
                intersection_point.append(point)
    
    cells_left = []
    cells_right = []
    # 4 intersection points form a cell. Top left index is i bottom right is i + 5 in intersection_point list 
    for i in range(len(intersection_point) - 6):
        if i % 5 == 4:
            continue
        x1, y1 = intersection_point[i]
        x2, y2 = intersection_point[i + 6]
        if i % 5 <= 1:
            cells_left.append((x1, y1, x2, y2))
        else:
            cells_right.append((x1, y1, x2, y2))


    return cells_left + cells_right

def process_02(image_path, output_paths):
    original_image = load_image(image_path)
    img_grey = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)

    # Apply sobel
    sobel_horizontal = np.array([[ -1, 0, 1],
                                  [ -2, 0, 2],
                                  [ -1, 0, 1]]) * 0.25

    sobel_vertical = np.array([[ -1, -2, -1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]]) * 0.25
    
    def get_kernel_from_size(size, size_y=None):
        if size_y is None:
            return np.ones((size, size), np.uint8)
        return  np.ones((size, size_y), np.uint8)
    
    kernel_size_dilate = 5
    kernel_dilate = get_kernel_from_size(kernel_size_dilate)

    kernel_size_erode = 5
    kernel_erode = get_kernel_from_size(kernel_size_erode)
    
    # Transform to get horizontal and vertical
    sobel_horizontal_result = np.abs(cv2.filter2D(img, cv2.CV_32F, sobel_horizontal)).astype(np.uint8)
    _, sobel_horizontal_edge = cv2.threshold(sobel_horizontal_result, 250, 255, cv2.THRESH_BINARY)
    sobel_horizontal_edge = cv2.dilate(sobel_horizontal_edge, kernel_dilate)
    sobel_horizontal_edge = cv2.erode(sobel_horizontal_edge, kernel_erode)
    sobel_horizontal_edge = detect_line(sobel_horizontal_edge, "vertical")
    lines_vertical = extract_line_segments(sobel_horizontal_edge, "vertical")

    sobel_vertical_result = np.abs(cv2.filter2D(img, cv2.CV_32F, sobel_vertical)).astype(np.uint8)
    _, sobel_vertical_edge = cv2.threshold(sobel_vertical_result, 250, 255, cv2.THRESH_BINARY)
    sobel_vertical_edge = cv2.dilate(sobel_vertical_edge, kernel_dilate)
    sobel_vertical_edge = cv2.erode(sobel_vertical_edge, kernel_erode)
    sobel_vertical_edge = detect_line(sobel_vertical_edge, "horizontal")
    lines_horizontal = extract_line_segments(sobel_vertical_edge, "horizontal")


    cells = get_cell_from_lines(lines_horizontal, lines_vertical)

    img_grey_horizontal = np.abs(cv2.filter2D(img_grey, cv2.CV_32F, sobel_horizontal)).astype(np.uint8)
    img_grey_vertical = np.abs(cv2.filter2D(img_grey, cv2.CV_32F, sobel_vertical)).astype(np.uint8)
    img_grey_magnitude = cv2.magnitude(img_grey_horizontal.astype(np.float32), img_grey_vertical.astype(np.float32)).astype(np.uint8)
    img_grey_magnitude = img_grey_magnitude & (255 - sobel_horizontal_edge) & (255 - sobel_vertical_edge)
    
    img_grey_edge = np.zeros(img_grey_magnitude.shape, dtype=np.uint8)
    
    # OTSU for each cell
    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = cell
        cell_img = img_grey_magnitude[y1:y2, x1:x2]
        
        # Remove 25% of the width
        if i % 2 == 0:
            cell_img = cell_img[:, int((x2 - x1) * 0.25):]
            img_grey_edge[y1:y2, x1 + int((x2 - x1) * 0.25):x2] = cell_img
        else:
            img_grey_edge[y1:y2, x1:x2] = cell_img
    
    _, img_grey_edge = cv2.threshold(img_grey_edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('img_grey_edge_org', img_grey_edge)

    size = 3
    img_grey_edge = cv2.dilate(img_grey_edge, get_kernel_from_size(size * 2))
    img_grey_edge = cv2.erode(img_grey_edge, get_kernel_from_size(size * 2))
    img_grey_edge = cv2.erode(img_grey_edge, get_kernel_from_size(3))
    # Remove vertical and horizontal big line
    # img_grey_edge = img_grey_edge & (~sobel_horizontal_edge) & (~)

    threshold = 1
    detections = []

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = cell
        cell_img = img_grey_edge[y1:y2, x1:x2]
        # Remove 25% of the width
        if i % 2 == 0:
            cell_img = cell_img[:, int((x2 - x1) * 0.25):]
        white_percent = (cell_img == 255).sum() / (cell_img.shape[0] * cell_img.shape[1]) * 100

        if white_percent > threshold:
            detections.append((x1, y1, x2, y2))

    # Visualization
    vis = original_image.copy()
    for i, (x1, y1, x2, y2) in enumerate(detections, start=1):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, str(i), (x1 + 10, y1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 3)

    # Save vis to annotate
    cv2.imwrite(str(output_paths["annotated"] / image_path.name), vis)

    # Save cell images to cells (folder name is the name of the image)
    for i, (x1, y1, x2, y2) in enumerate(detections, start=1):
        cell_img = original_image[y1:y2, x1:x2]
        cv2.imwrite(str(output_paths["cells"] / f"{image_path.stem}_cell_{i}.png"), cell_img)

    # Save cell to json
    result = {
        "image": image_path.name,
        "table_bbox": [0, 0, original_image.shape[1], original_image.shape[0]],
        "vertical_lines": lines_vertical,
        "horizontal_lines": lines_horizontal,
        "detections": [{"bbox": [x1, y1, x2, y2]} for (x1, y1, x2, y2) in detections],
    }
    json_path = output_paths["json"] / f"{image_path.stem}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def main_02():
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
    
    for image_path in image_paths:
        process_02(image_path, output_paths)


if __name__ == "__main__":
    main_02()