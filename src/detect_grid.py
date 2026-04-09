from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Cell:
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int


def _extract_lines(binary: np.ndarray, axis: str) -> np.ndarray:
    height, width = binary.shape[:2]
    if axis == "horizontal":
        kernel_width = max(width // 30, 25)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    else:
        kernel_height = max(height // 30, 25)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    repaired = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    return cv2.dilate(repaired, kernel, iterations=1)


def detect_table_bbox(binary: np.ndarray) -> tuple[int, int, int, int]:
    horizontal = _extract_lines(binary, "horizontal")
    vertical = _extract_lines(binary, "vertical")
    grid_mask = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("Unable to detect any table contour.")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Expand bbox to capture faint/broken lines at the bottom (and edges) that
    # may have been missed by the morphological line extraction.
    img_h, img_w = binary.shape[:2]
    margin_x = int(w * 0.02)
    margin_y = int(h * 0.08)  # ~8 % ≈ 2-3 row heights of extra search room
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(img_w - x, w + 2 * margin_x)
    h = min(img_h - y, h + 2 * margin_y)
    return x, y, w, h


def _group_positions(indices: np.ndarray, max_gap: int = 6) -> list[int]:
    if indices.size == 0:
        return []

    groups: list[list[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        value = int(idx)
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])

    return [int(round(sum(group) / len(group))) for group in groups]


def _line_positions(mask: np.ndarray, axis: str) -> list[int]:
    if axis == "horizontal":
        projection = np.count_nonzero(mask, axis=1)
        # Lowered from 0.2 → 0.10: faint/broken lines near the bottom often
        # span only ~10-15 % of the row width after morphological extraction.
        threshold = max(int(mask.shape[1] * 0.10), 15)
    else:
        projection = np.count_nonzero(mask, axis=0)
        threshold = max(int(mask.shape[0] * 0.3), 20)

    indices = np.where(projection >= threshold)[0]
    return _group_positions(indices)


def _dominant_step(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    diffs = [b - a for a, b in zip(values, values[1:])]
    candidates = [diff for diff in diffs if 35 <= diff <= 80]
    if not candidates:
        candidates = [diff for diff in diffs if diff > 0]
    if not candidates:
        return 0
    return int(round(float(np.median(candidates))))


def _fill_missing_lines(positions: list[int], max_position: int | None = None) -> list[int]:
    if len(positions) < 2:
        return positions

    step = _dominant_step(positions)
    if not step:
        return positions

    filled = [positions[0]]
    if positions[0] > int(step * 1.2):
        filled = [positions[0] - step, positions[0]]

    for current in positions[1:]:
        previous = filled[-1]
        gap = current - previous
        if gap > int(step * 1.6):
            missing = max(int(round(gap / step)) - 1, 1)
            for index in range(missing):
                filled.append(previous + step * (index + 1))
        filled.append(current)

    # Extrapolate forward: if the last detected line is well above the table
    # bottom (max_position), keep adding lines at the same step until we reach
    # it.  This covers rows whose grid lines were too faint to be detected but
    # whose position is predictable from the regular grid spacing.
    if max_position is not None:
        while filled[-1] + int(step * 0.8) <= max_position:
            filled.append(filled[-1] + step)

    return filled


def _regularize_main_rows(positions: list[int], row_count: int = 30) -> list[int]:
    if len(positions) < 8:
        return positions

    diffs = [b - a for a, b in zip(positions, positions[1:]) if 40 <= (b - a) <= 70]
    if not diffs:
        return positions

    base_step = float(np.median(diffs))
    step_candidates = [base_step - 1.0, base_step, base_step + 1.0]
    anchor_index = 0
    if len(positions) >= 3 and (positions[1] - positions[0]) < base_step * 0.6:
        anchor_index = 1
    anchor_position = float(positions[anchor_index])
    start_candidates = [anchor_position - 1.0, anchor_position, anchor_position + 1.0]
    tolerance = 8.0

    best_score = -1
    best_model: tuple[float, float] | None = None

    for start in start_candidates:
        for step in step_candidates:
            generated = np.array([start + step * idx for idx in range(row_count + 1)], dtype=float)
            score = 0
            for position in positions:
                if np.min(np.abs(generated - position)) <= tolerance:
                    score += 1
            if score > best_score or (
                score == best_score and best_model is not None and abs(float(start) - anchor_position) < abs(best_model[0] - anchor_position)
            ):
                best_score = score
                best_model = (float(start), float(step))

    if best_model is None:
        return positions

    start, step = best_model
    matched_indices: list[int] = []
    matched_positions: list[int] = []
    for position in positions:
        index = int(round((position - start) / step))
        if 0 <= index <= row_count:
            target = start + step * index
            if abs(position - target) <= tolerance:
                matched_indices.append(index)
                matched_positions.append(position)

    if len(matched_indices) >= 6:
        indices = np.array(matched_indices, dtype=float)
        observed = np.array(matched_positions, dtype=float)
        slope, intercept = np.polyfit(indices, observed, 1)
        start = float(intercept)
        step = float(slope)

    # Build lookup: regularised index → actual detected position (where available).
    # This preserves real line positions instead of replacing them with the ideal model,
    # preventing accumulated drift at the bottom of the grid.
    actual_by_idx: dict[int, int] = dict(zip(matched_indices, matched_positions))

    result: list[int] = []
    for idx in range(row_count + 1):
        if idx in actual_by_idx:
            result.append(actual_by_idx[idx])
        else:
            result.append(int(round(start + step * idx)))
    return result


def detect_grid_lines(binary: np.ndarray, table_bbox: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    x, y, w, h = table_bbox
    roi = binary[y : y + h, x : x + w]

    horizontal = _extract_lines(roi, "horizontal")
    vertical = _extract_lines(roi, "vertical")

    ys = _line_positions(horizontal, "horizontal")
    xs = _line_positions(vertical, "vertical")

    ys = _fill_missing_lines(ys, max_position=h)
    ys = _regularize_main_rows(ys)

    xs = [x + value for value in xs]
    ys = [y + value for value in ys]
    return xs, ys


def build_cells(xs: list[int], ys: list[int]) -> list[Cell]:
    cells: list[Cell] = []
    if len(xs) < 2 or len(ys) < 2:
        return cells

    for row_idx in range(len(ys) - 1):
        for col_idx in range(len(xs) - 1):
            x0 = xs[col_idx]
            y0 = ys[row_idx]
            x1 = xs[col_idx + 1]
            y1 = ys[row_idx + 1]
            width = x1 - x0
            height = y1 - y0
            if width < 20 or height < 12:
                continue
            cells.append(Cell(row=row_idx, col=col_idx, x=x0, y=y0, w=width, h=height))
    return cells


def detect_cells(binary: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int], list[Cell]]:
    table_bbox = detect_table_bbox(binary)
    xs, ys = detect_grid_lines(binary, table_bbox)
    cells = build_cells(xs, ys)
    return table_bbox, xs, ys, cells
