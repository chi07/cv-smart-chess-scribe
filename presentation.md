# Pipeline: Phát hiện ô chữ viết tay trên phiếu ghi nước đi cờ vua

---

## Stage 0 — Entry Point (`main.py`)

```
Input image (PNG/JPG)
        │
        ▼
  load_image()  ──►  deskew_image()  ──►  analyze_image_variant()
        │                                         │
        └──────────────────────────────────────►  variant_score()
                                                  │
                                              best variant
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                       ▼                       ▼
                   annotated image           cell crops              JSON metadata
```

Với mỗi ảnh, pipeline thử **2 variants**: ảnh gốc và ảnh đã deskew. Variant nào có điểm cao hơn (dựa trên số đường dọc/ngang phát hiện được và số ô có chữ) sẽ được chọn làm kết quả cuối.

---

## Stage 1 — Tiền xử lý (`preprocess.py`)

**Mục tiêu:** Chuyển ảnh màu thành binary mask, làm nổi bật nét mực.

```
BGR image
    │
    ▼
GaussianBlur(5×5)        ← giảm nhiễu nhỏ
    │
    ▼
adaptiveThreshold(GAUSSIAN_C, THRESH_BINARY_INV, blockSize=35, C=11)
    │
    ▼
Binary mask (nét mực = trắng, nền = đen)
```

Tại sao dùng **adaptive** thay vì global threshold? Vì ảnh chụp tay thường có ánh sáng không đều — adaptive tính ngưỡng theo từng vùng nhỏ, tránh mất nét ở vùng tối hoặc cháy trắng ở vùng sáng.

**Deskew:** Nếu ảnh bị nghiêng, dùng `HoughLinesP` để phát hiện các đường ngang dài, tính median góc nghiêng, rồi xoay ảnh về thẳng bằng `warpAffine`.

---

## Stage 2 — Phát hiện bảng và lưới (`detect_grid.py`)

### 2a. Tách đường ngang / dọc bằng morphology

```
Binary mask
    │
    ├──► MORPH_OPEN với kernel ngang (width//30 × 1)  ──► đường ngang
    │
    └──► MORPH_OPEN với kernel dọc  (1 × height//30)  ──► đường dọc
```

`MORPH_OPEN` = erosion rồi dilation. Kernel hình que ngang chỉ giữ lại những pixel nằm thành hàng dài liên tục → loại bỏ nét chữ, chỉ còn đường kẻ bảng.

### 2b. Tìm bounding box bảng lớn nhất

OR hai mask lại → tìm contour lớn nhất → `boundingRect` → đây là vùng bảng.

### 2c. Suy ra vị trí các đường lưới

Trong ROI của bảng:
- **Projection**: đếm số pixel trắng theo từng hàng (hoặc cột).
- Hàng nào có đủ pixel trắng (≥ 20% chiều rộng) → đó là đường kẻ ngang.
- Gom các hàng liền kề thành 1 đường (clustering), lấy trung bình.

### 2d. Regularize — điền đường bị thiếu

Đây là bước quan trọng nhất của stage này:

```
Detected positions: [50, 100, 200, 250, 300]  ← thiếu ~150
                                ↑
                         gap = 100 ≈ 2× step

_fill_missing_lines() → [50, 100, 150, 200, 250, 300]
```

Sau đó `_regularize_main_rows()` fit một grid đều đặn 30 hàng bằng cách:
1. Tính `base_step` = median của các khoảng cách hợp lệ (40–70px).
2. Thử các tổ hợp `(start, step)` để tìm model khớp nhiều điểm nhất.
3. Dùng `polyfit` để tinh chỉnh lần cuối.

Kết quả: 31 đường ngang đều nhau, bất kể đường kẻ gốc có bị mờ hay thiếu.

---

## Stage 3 — Phân loại ô có chữ viết tay (`detect_handwriting.py`)

### 3a. Lọc cột

Chỉ xét các cột "rộng" (width ≥ 115% median width). Cột hẹp thường là cột số thứ tự in sẵn → bỏ qua.

### 3b. Lọc header

Hàng đầu tiên nếu cao bất thường (< 75% row_step) → đó là header → bỏ qua.

### 3c. Tính đặc trưng cho từng ô

Với mỗi ô hợp lệ, crop ROI có padding (8% ngang, 18% dọc) để tránh lấy nhầm đường kẻ:

```
ink_ratio = số pixel trắng / tổng pixel ROI

component_count, largest_component = connectedComponentsWithStats()
    → chỉ đếm component có area ∈ [8, 60% ROI]  ← lọc nhiễu nhỏ và đường kẻ lớn
```

### 3d. Quyết định

```
if ink_ratio < 0.015:          → bỏ (ô trống)
if component_count < 2
   AND largest_component < 25: → bỏ (chỉ có 1 nét nhỏ, có thể là nhiễu)
→ còn lại: ô có chữ viết tay ✓
```

Hai điều kiện kết hợp giúp tránh cả 2 loại lỗi:
- **False positive**: ô trống nhưng có vài pixel nhiễu → ink_ratio thấp → loại.
- **False positive**: đường kẻ bảng lọt vào ROI → component lớn nhưng area > 60% ROI → không đếm.

---

## Stage 4 — Xuất kết quả

| Output | Nội dung |
|---|---|
| `output/annotated/` | Ảnh gốc + khung đỏ quanh ô có chữ, khung xanh quanh bảng |
| `output/cells/` | Crop từng ô, tên file ghi rõ row/col |
| `output/json/` | Metadata: bbox bảng, danh sách đường lưới, ink_ratio, component_count |

---

## Tóm tắt luồng dữ liệu

```
PNG
 │
 ├─[preprocess]──► binary mask
 │
 ├─[detect_grid]──► table_bbox → xs, ys → cells[]
 │
 ├─[detect_handwriting]──► detections[] (filtered cells)
 │
 └─[main]──► annotated.png + crops + .json
```

---

## Điểm mạnh / hạn chế

**Mạnh:**
- Không cần training data, chạy được ngay.
- Regularization grid bù được đường kẻ mờ hoặc thiếu.
- Deskew tự động xử lý ảnh chụp nghiêng nhẹ.

**Hạn chế:**
- Giả định bảng có ~30 hàng, ~4 cột nội dung — nếu form khác cấu trúc thì cần điều chỉnh tham số.
- Chữ viết rất mờ (ink_ratio < 0.015) sẽ bị bỏ sót.
- Không nhận dạng nội dung chữ, chỉ phát hiện có/không.
