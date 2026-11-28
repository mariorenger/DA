# Câu hỏi cần hỏi thầy hướng dẫn (chuẩn bị cho buổi gặp)

Ngữ cảnh một câu: luận văn đo xem mô hình ngôn ngữ cỡ nhỏ và vừa có dùng chuỗi suy luận một cách nhân quả không. Kết quả đến Round 40 (REVIEW.md) cho thấy: mô hình chép số sai khi đọc đáp án ngay sau chuỗi (đến 32B), nhưng một câu nhắc tính lại hoặc chưng cất làm hành vi đó giảm gần hết; cơ chế kích hoạt chỉ nhận ra loại lỗi được chèn, không nhận ra lỗi tự nhiên (câu nhắc chỉ sửa 7 đến 14% chuỗi sai tự nhiên của chính mô hình).

## 1. Chốt nghĩa của chữ causal
- Thầy muốn causal theo nghĩa nào: (a) suy luận nhân quả kiểu Pearl (CLadder, Corr2Cause, do-calculus), (b) khám phá đồ thị nhân quả từ dữ liệu, hay (c) mô hình có dùng chuỗi suy luận của chính nó một cách nhân quả (hướng hiện tại)?
- Nếu (a) hoặc (b), thầy chấp nhận để phần chẩn đoán hiện tại làm nền và thêm một chương mới không?

## 2. Phát biểu giả thuyết của thầy cho kiểm chứng được
Bản em đang dùng: mô hình không tự kiểm tra được suy luận của chính nó, cần thông tin ngoài (bảng luật, đồ thị, engine).
- Bằng chứng ủng hộ: copy test (TN3, TN13), tỷ lệ sửa lỗi tự nhiên 7 đến 14% (Round 40), cơ chế kích hoạt chỉ nhận ra lỗi chèn (Round 37 đến 39).
- Bằng chứng chưa ủng hộ toàn phần: fine-tuning trên chuỗi dạy được composition (TN7); một câu nhắc làm mô hình bớt chép số sai (Round 33).
- Câu hỏi: phát biểu này có đúng ý thầy không, và thầy cần bằng chứng nào nữa để coi là đã kiểm chứng?

## 3. Loại tri thức ngoài và miền
- Loại nào: bảng luật composition (đang có ở CLUTRR), đồ thị tri thức, hay engine do-calculus (DoWhy)?
- Miền: CLUTRR (luật đóng, ground truth chính xác) hay paper IT (miền mở, luật không bắc cầu chặt, khó nhãn)? Em đề xuất CLUTRR làm lõi và paper IT làm hướng phát triển.
- Nếu thầy muốn miền thật, thầy có nguồn dữ liệu có nhãn chuyên gia hoặc cho em hướng chọn miền không?

## 4. Phạm vi và tiêu chuẩn luận văn
- Thầy chấp nhận kết quả âm có kiểm soát (bác bỏ các thước đo bước, cơ chế kích hoạt không nhận ra lỗi tự nhiên) làm đóng góp chính, hay thầy cần một phương pháp cải thiện kết quả?
- Số seed tối thiểu thầy yêu cầu cho các so sánh chính (hiện nhiều kết quả một seed).
- Có cần chạy mô hình rất lớn qua API (ước tính 50 đến 100 USD) không, hay 14B đến 32B bản 4-bit là đủ?

## 5. Hướng thực nghiệm tiếp theo em đề xuất (cần thầy chọn)
- Round 41: dùng bảng luật làm bộ kiểm tra từng bước ngoài, đo xem lỗi tự nhiên của mô hình có được sửa nhiều hơn mức 7 đến 14% của tự tính lại không.
- Thêm seed cho Round 33, 36, 37.
- Pilot CLadder: đưa đồ thị vào đề, can thiệp lên đồ thị, đo nhạy đúng và không nhạy sai.

## 6. Hội nghị, thời hạn, cách báo cáo
- Mục tiêu nộp (Findings, COLM, TMLR, workshop) và mốc thời gian.
- Định dạng thầy muốn xem định kỳ: slide, báo cáo chi tiết, hay REVIEW.md.
