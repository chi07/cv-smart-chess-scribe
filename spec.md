"Yêu cầu chung:

Dựa trên những kiến thức đã học, bạn cần tạo một chuỗi xử lý ảnh sao cho cho kết quả tốt nhất khi áp dụng trên các ảnh đầu vào. Trong chuỗi xử lý này, ở mỗi bước bạn phải lựa chọn cẩn thận phương pháp và các tham số phù hợp. Bạn phải xác định một chuỗi duy nhất với các giá trị tham số giống nhau để áp dụng cho tất cả các ảnh.
Bạn có thể sử dụng tất cả các hàm trong OpenCV hoặc cả trong các thư viện khác.
Ở mỗi bước, ngay cả khi bạn chưa tìm được phương pháp/tham số hoàn hảo để giải quyết bài toán, bạn vẫn cần phải phân tích ảnh hưởng của chúng lên kết quả trung gian và kết quả cuối cùng.
Gợi ý cho chuỗi xử lý:
•	Tiền xử lý: giảm nhiễu, cải thiện chất lượng ảnh để dễ dàng cho các bước sau (các đối tượng rõ ràng hơn, được tách biệt tốt hơn). Bạn có thể kết hợp nhiều kỹ thuật theo thứ tự hợp lý, với tham số thích hợp.
•	Phân đoạn: tách ảnh thành các vùng riêng biệt, có thể sử dụng một hoặc nhiều phương pháp (ngưỡng tự động, watershed, phát triển vùng, …).
•	Hậu xử lý: sửa lỗi của bước phân đoạn trước bằng các phép toán hình thái học (erosion, dilation, opening, closing, …), gán nhãn vùng/đường biên và đếm đối tượng.
⚠️ Lưu ý (nhắc lại): Cần áp dụng cùng một chuỗi xử lý cho tất cả các ảnh. Chương trình của bạn chỉ nhận vào duy nhất tên một ảnh và thực hiện các xử lý giống như các ảnh khác. Ngưỡng có thể được xác định trước hoặc chọn tự động, nhưng không được phép điều chỉnh thủ công bởi người dùng. Chương trình bạn nộp không có đối số nào khác ngoài tên ảnh đầu vào.

4)	Báo cáo: Bạn cần mô tả một chuỗi xử lý duy nhất hoạt động cho tất cả các ảnh.
Ở mỗi bước, cần nêu rõ:
•	phương pháp đã chọn,
•	các giá trị tham số,
•	lý do lựa chọn.
Bạn cần trình bày rõ ràng các kết quả trung gian quan trọng trong chuỗi xử lý, phân tích cả kết quả tốt và chưa tốt. Giải thích kết quả cuối cùng có đạt yêu cầu hay không, và nhất là phải nói rõ lý do.
Tiêu chí đánh giá:
•	chất lượng kết quả,
•	phần giải thích,
•	và khả năng thiết kế một chuỗi xử lý ảnh hoàn chỉnh cho ứng dụng."