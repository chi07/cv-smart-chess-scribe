from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from detect_grid import Cell


def _dominant_step(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    diffs = [b - a for a, b in zip(values, values[1:]) if b - a > 0]
    if not diffs:
        return 0
    rounded = [int(round(diff / 5) * 5) for diff in diffs]
    return Counter(rounded).most_common(1)[0][0]


def _wide_column_indices(cells: list[Cell]) -> set[int]:
    column_widths: dict[int, list[int]] = {}
    for cell in cells:
        column_widths.setdefault(cell.col, []).append(cell.w)

    if not column_widths:
        return set()

    medians = {col: int(np.median(widths)) for col, widths in column_widths.items()}
    typical_width = int(np.median(list(medians.values())))
    wide_columns = {col for col, width in medians.items() if width >= int(typical_width * 1.15)}
    if wide_columns:
        return wide_columns
    return set(column_widths.keys())


def _ink_ratio(binary_roi: np.ndarray) -> float:
    if binary_roi.size == 0:
        return 0.0
    return float(np.count_nonzero(binary_roi)) / float(binary_roi.size)


def _count_components(binary_roi: np.ndarray) -> tuple[int, float]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)
    if component_count <= 1:
        return 0, 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    useful = areas[(areas >= 8) & (areas <= binary_roi.size * 0.6)]
    largest = float(useful.max()) if useful.size else 0.0
    return int(useful.size), largest


def classify_cells(binary: np.ndarray, cells: list[Cell], ys: list[int]) -> list[dict]:
    detections: list[dict] = []
    if not cells:
        return detections

    wide_columns = _wide_column_indices(cells)
    row_step = _dominant_step(ys)
    header_cutoff = ys[1] if len(ys) > 1 else 0

    for cell in cells:
        if cell.col not in wide_columns:
            continue
        if cell.y < header_cutoff:
            continue
        if row_step and cell.h > row_step * 1.6:
            continue

        pad_x = max(int(cell.w * 0.08), 2)
        pad_y = max(int(cell.h * 0.18), 2)
        x0 = cell.x + pad_x
        y0 = cell.y + pad_y
        x1 = cell.x + cell.w - pad_x
        y1 = cell.y + cell.h - pad_y
        roi = binary[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        ink_ratio = _ink_ratio(roi)
        component_count, largest_component = _count_components(roi)

        if ink_ratio < 0.015:
            continue
        if component_count < 2 and largest_component < 25:
            continue

        detections.append(
            {
                "row": cell.row,
                "col": cell.col,
                "bbox": [cell.x, cell.y, cell.x + cell.w, cell.y + cell.h],
                "ink_ratio": round(ink_ratio, 4),
                "component_count": component_count,
            }
        )

    return detections
