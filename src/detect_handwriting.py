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

    areas   = stats[1:, cv2.CC_STAT_AREA]
    widths  = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    aspect  = widths / np.maximum(heights, 1)

    # Loại nhiễu quá nhỏ, component quá lớn (nền), và đường kẻ ngang chảy vào
    # ROI (aspect ratio w/h > 5 → dải ngang mỏng, không phải nét chữ).
    mask  = (areas >= 8) & (areas <= binary_roi.size * 0.6) & (aspect <= 5.0)
    useful = areas[mask]
    largest = float(useful.max()) if useful.size else 0.0
    return int(useful.size), largest


def _left_ink_ratio(binary_roi: np.ndarray, left_fraction: float = 1 / 4) -> float:
    """Tỉ lệ mực nằm trong vùng left_fraction bên trái ROI.

    Số in sẵn (số thứ tự nước đi) luôn nằm sát cạnh trái ô → left_ratio ≈ 0.9–1.0.
    Chữ viết tay (ký hiệu nước đi) trải đều hoặc lệch phải → left_ratio ≈ 0.3–0.5.
    Ngưỡng phân biệt thực nghiệm: 0.75.
    """
    if binary_roi.size == 0:
        return 0.5
    split = max(1, int(binary_roi.shape[1] * left_fraction))
    left_ink = int(np.count_nonzero(binary_roi[:, :split]))
    total_ink = int(np.count_nonzero(binary_roi))
    if total_ink == 0:
        return 0.5
    return left_ink / total_ink


def classify_cells(binary: np.ndarray, cells: list[Cell], ys: list[int]) -> list[dict]:
    detections: list[dict] = []
    if not cells:
        return detections

    wide_columns = _wide_column_indices(cells)
    row_step = _dominant_step(ys)
    first_band_height = ys[1] - ys[0] if len(ys) > 1 else 0
    has_header_band = bool(row_step and first_band_height and first_band_height < row_step * 0.75)
    header_cutoff = ys[1] if has_header_band and len(ys) > 1 else ys[0] if ys else 0

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
        left_ratio = _left_ink_ratio(roi)

        if ink_ratio < 0.006:
            continue
        if component_count < 2 and largest_component < 60:
            continue
        # Loại ô có mực tập trung ở cạnh trái VÀ tổng mực ít → số thứ tự in sẵn.
        # Chữ viết tay đôi khi cũng lệch trái nhưng có ink_ratio cao hơn (nét dày hơn).
        if left_ratio > 0.75 and ink_ratio < 0.035:
            continue

        detections.append(
            {
                "row": cell.row,
                "col": cell.col,
                "bbox": [cell.x, cell.y, cell.x + cell.w, cell.y + cell.h],
                "ink_ratio": round(ink_ratio, 4),
                "component_count": component_count,
                "left_ratio": round(left_ratio, 4),
            }
        )

    return detections
