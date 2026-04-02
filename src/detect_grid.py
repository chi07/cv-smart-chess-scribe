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
    return cv2.boundingRect(largest)


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
        threshold = max(int(mask.shape[1] * 0.2), 20)
    else:
        projection = np.count_nonzero(mask, axis=0)
        threshold = max(int(mask.shape[0] * 0.3), 20)

    indices = np.where(projection >= threshold)[0]
    return _group_positions(indices)


def detect_grid_lines(binary: np.ndarray, table_bbox: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    x, y, w, h = table_bbox
    roi = binary[y : y + h, x : x + w]

    horizontal = _extract_lines(roi, "horizontal")
    vertical = _extract_lines(roi, "vertical")

    ys = _line_positions(horizontal, "horizontal")
    xs = _line_positions(vertical, "vertical")

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
