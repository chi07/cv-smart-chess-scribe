# cv-smart-chess-scribe

## Bài toán

Từ các ảnh phiếu ghi lịch sử nước đi trong thư mục `dataset/`, cần phát hiện các ô có chứa chữ viết tay và xuất ra:

- Ảnh gốc đã được khoanh vùng các ô có chữ viết tay.
- Các ảnh crop tương ứng với từng ô được phát hiện.
- File JSON chứa metadata của các ô được phát hiện.

Lưu ý: phần mô tả ban đầu ghi là "cờ vây", nhưng ảnh mẫu hiện tại giống phiếu ghi nước đi của cờ vua. Pipeline xử lý ảnh vẫn giữ nguyên ý tưởng: tìm bảng, tách ô, rồi lọc các ô có nét bút viết tay.

## Cấu trúc thư mục

```text
cv-smart-chess-scribe/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── dataset/
│   ├── 001_0.png
│   ├── 009_0.png
│   ├── 011_0.png
│   └── 012_0.png
├── output/
│   ├── annotated/
│   ├── cells/
│   └── json/
└── src/
    ├── detect_grid.py
    ├── detect_handwriting.py
    ├── main.py
    └── preprocess.py
```

## Ý tưởng giải quyết

Pipeline hiện tại là bản đầu theo hướng rule-based với OpenCV:

1. Đọc ảnh và chuyển sang grayscale.
2. Tạo ảnh nhị phân bằng adaptive threshold để làm nổi bật đường kẻ và nét bút.
3. Dùng morphology để tách các đường ngang và dọc của bảng.
4. Tìm bounding box của bảng lớn nhất.
5. Suy ra vị trí các đường lưới ngang và dọc.
6. Tạo danh sách các ô từ giao điểm của các đường lưới.
7. Lọc các cột quá hẹp để bỏ qua cột số thứ tự in sẵn.
8. Tính mật độ nét mực và số connected component trong từng ô.
9. Ô nào vượt ngưỡng sẽ được xem là ô có chữ viết tay.
10. Xuất ảnh annotated, ảnh crop, và file JSON.

Ưu điểm của hướng này:

- Nhanh để triển khai.
- Phù hợp với form có cấu trúc bảng rõ ràng.
- Dễ giải thích trong báo cáo môn học.

## Cài đặt

Tạo môi trường ảo rồi cài dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cách chạy

Chạy với đúng một ảnh đầu vào:

```bash
python3 src/main.py 001_0.png
```

Hoặc truyền đường dẫn ảnh đầy đủ/ tương đối:

```bash
python3 src/main.py dataset/001_0.png
```

Sau khi chạy xong, kết quả sẽ nằm trong:

- `output/annotated/`: ảnh gốc đã được vẽ khung.
- `output/cells/`: từng ô được crop ra.
- `output/json/`: metadata cho từng ảnh.

## Giải thích từng file trong `src/`

- `preprocess.py`: đọc ảnh, grayscale, adaptive threshold.
- `detect_grid.py`: phát hiện bảng, đường ngang, đường dọc, và tạo danh sách các ô.
- `detect_handwriting.py`: tính đặc trưng đơn giản để quyết định ô nào có chữ viết tay.
- `main.py`: chạy toàn pipeline, lưu output và in summary.

## Step-by-step nên làm tiếp theo

### Bước 1: Chạy bản baseline hiện tại

Mục tiêu:

- Xem pipeline có phát hiện đúng phần lớn các ô có chữ hay không.
- Kiểm tra nhanh output để biết đang thiếu ở đâu.

Việc cần làm:

- Cài dependency.
- Chạy `python3 src/main.py 001_0.png`.
- Mở thư mục `output/annotated/` để xem kết quả.

### Bước 2: Đánh giá thủ công trên từng ảnh

Mục tiêu:

- Đếm số ô đúng, số ô bị sót, số ô bị nhận nhầm.

Việc cần làm:

- So sánh ảnh gốc với ảnh annotated.
- Ghi chú ảnh nào bị lệch lưới.
- Ghi chú ô trống nào bị bắt nhầm.

### Bước 3: Tinh chỉnh ngưỡng

Vị trí cần tinh chỉnh chính:

- Kernel morphology trong `detect_grid.py`.
- Ngưỡng `ink_ratio`.
- Điều kiện `component_count`.
- Tỷ lệ padding khi crop ROI trong ô.

Nếu mô hình bắt nhầm nhiều ô trống:

- Tăng `ink_ratio`.
- Tăng diện tích tối thiểu của connected components.

Nếu mô hình bỏ sót ô có chữ mờ:

- Giảm `ink_ratio`.
- Giảm điều kiện `largest_component`.

### Bước 4: Cải thiện tiền xử lý

Nếu ảnh bị nghiêng hoặc ánh sáng không đều:

- Bổ sung deskew hoặc perspective correction.
- Thử CLAHE trước khi threshold.
- Thử adaptive threshold với tham số khác.

### Bước 5: Chuẩn hóa output để làm báo cáo

Nên có:

- 1 hình minh họa pipeline.
- 1 bảng kết quả cho từng ảnh.
- 1 phần mô tả lỗi thường gặp.
- 1 phần hướng cải tiến bằng machine learning nếu dữ liệu tăng lên.

## Hướng nâng cấp sau baseline

Nếu dữ liệu lớn hơn hoặc ảnh đa dạng hơn, có thể mở rộng theo 2 hướng:

### Hướng 1: Phân loại từng ô

Sau khi đã tách ô bằng lưới, train classifier:

- `empty`
- `handwritten`

Hướng này nhẹ hơn object detection vì lưới đã có sẵn.

### Hướng 2: OCR hoặc nhận dạng ký tự

Sau khi tách đúng ô có chữ, có thể làm thêm:

- nhận dạng ký hiệu nước đi
- trích xuất nội dung chữ viết tay

## Định dạng JSON đầu ra

Ví dụ:

```json
{
  "image": "001_0.png",
  "table_bbox": [0, 0, 100, 100],
  "vertical_lines": [10, 40, 80],
  "horizontal_lines": [5, 25, 45],
  "detections": [
    {
      "row": 1,
      "col": 2,
      "bbox": [20, 30, 90, 55],
      "ink_ratio": 0.0421,
      "component_count": 5
    }
  ]
}
```

## Ghi chú

- Đây là bản baseline để repo có thể chạy end-to-end.
- Bộ rule hiện tại phù hợp để khởi động nhanh, nhưng chưa đảm bảo tối ưu cho mọi ảnh.
- Nếu cần, bước tiếp theo nên là tinh chỉnh tham số trực tiếp trên các ảnh trong `dataset`.
