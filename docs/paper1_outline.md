# Bài 1: bản nháp cấu trúc (11/11/2026)

Tên làm việc: **Mỗi bước viết ra gộp được bao nhiêu bước tính? Ngưỡng ngân sách token của chuỗi suy luận phụ thuộc phép tính và mô hình, và các điểm quan trọng của bước không dự đoán được nó.**

Tình trạng: khung đã đủ để viết bản thảo thô; ba khoảng trống còn lại ghi ở mục 5. Số liệu lấy từ `REVIEW.md` (mỗi vòng đăng ký giả thuyết trước khi chạy).

## 1. Mở đầu
Chuỗi suy luận dài tốn token, nên nhiều công trình chấm điểm từng bước để cắt (PNS của Yu, step entropy, IGPO, LIFT). Giả định ngầm: điểm quan trọng của bước cho biết bước đó cần thiết ra sao. Ta kiểm tra giả định này bằng ground truth và các đối chứng, rồi đo trực tiếp giới hạn của việc bỏ bước trên tác vụ có chuỗi tối thiểu biết trước.

## 2. Các đóng góp (theo mức chắc chắn)
1. **Điểm quan trọng của bước không dự đoán giá trị huấn luyện của bước** (Round 14 đến 16). AUROC với bước thừa cài sẵn: LOO 0.66–0.81, PNS 0.54–0.68, entropy dưới 0.4. Xóa bước ít quan trọng nhất không tốt hơn xóa bước quan trọng nhất ở 1.5B và 3B (+9.3 điểm ở 7B). Độ giảm độ đúng sau SFT gần như tuyến tính theo tỷ lệ token (Δ ≈ 63·tỷ lệ − 64, r = 0.94, 42 cách cắt, phân tích sau).
2. **Giữ phép tính có ích nhưng không đủ** (Round 17, 19): chuỗi chỉ giữ phép tính nằm trên đường xóa bước 10–12 điểm nhưng vẫn kém chuỗi đầy đủ 4–17 điểm; giữ mọi số không giúp thêm.
3. **Ngưỡng ngân sách token là điểm gãy, có chuỗi tối thiểu biết trước** (Round 20 đến 24): trên tra con trỏ, mỗi bước viết ra gộp được khoảng 3 đến 4 phép tra; độ đúng ở bước nhảy 4 là 0.36, 0.70, 0.92 (1.5B, 3B, 7B, trung bình theo seed); kết quả của 3B rất phụ thuộc seed (0.57–0.91). Qwen3-1.7B gãy ở 4→5. Loss cuối dự đoán độ đúng (Spearman −0.96, 13 lần chạy).
4. **Sức chứa phụ thuộc phép tính** (Round 22–23): cộng dồn số nhỏ không có điểm gãy (≥ 12 phép cộng mỗi bước ở 79–85%). Cộng modulo 10 không hội tụ nên không dùng được.
5. **Bias của PNS chỉ ở mô hình nhỏ nhất** (Round 14, 14c): 0.28 ở 1.5B, ≈ 0 từ 3B trở lên.
6. **Chép kết quả sai giảm theo quy mô nhưng không biến mất** (Round 18): 99.5 → 92 → 85.5 → 66 → 76.7% (1.5B đến 32B; 14B và 32B ở 4-bit, nguồn lượng tử hóa khác nhau).

## 3. Phương pháp
Bước thừa cài sẵn và nhãn "mang phép tính" (đồ thị phụ thuộc số) làm ground truth; cắt cùng số bước theo thứ tự ít quan trọng nhất, nhiều nhất, ngẫu nhiên; SFT LoRA; tác vụ tổng hợp con trỏ và cộng dồn với bước nhảy m; đăng ký trước mọi giả thuyết; bootstrap theo câu hỏi.

## 4. Giới hạn đã biết
Một benchmark thật (GSM8K, bão hòa ở 7B); nhiễu SFT 3–4 điểm và nhiều seed chỉ ở một vài ô; 14B, 32B ở 4-bit; tác vụ tổng hợp; Qwen3 ở chế độ không thinking; nhiều kiểm định.

## 5. Khoảng trống phải lấp trước khi viết bản thảo
1. **Phép tính thứ ba** để khẳng định "phụ thuộc phép tính" (hiện có hai: tra bảng và cộng). Round 26 (theo dõi hoán vị) đã đăng ký và chờ hạn mức GPU.
2. **Mô hình ≥ 14B trên tra con trỏ**: đã có 14B ở bước nhảy 4 (0.97, hội tụ); bước nhảy 6 chưa hội tụ (0.41) nên vị trí gãy của 14B vẫn nằm giữa 4 và 6.
3. **Liên kết điểm gãy với đường token GSM8K**: đã có phân tích thăm dò dùng thống kê trước huấn luyện (bước nhảy hiệu dụng r = −0.86 so với tỷ lệ token r = +0.76, 59 cách cắt); cần đối chứng sạch hơn (chuỗi GSM8K có bước đồng nhất hơn).
4. Một mô hình frontier qua API cho phần chép kết quả sai (cần ngân sách).
5. ~~Đọc toàn văn~~ Đã đọc (xem `docs/related_work_positioning.md`): điểm gãy theo bước nhảy được BAPO dự đoán về lý thuyết; phải đổi cách phát biểu đóng góp thành phép thử thực nghiệm của dự đoán đó.

## 6. Hội nghị
ACL 2027 qua ARR (04/01/2027) nếu 1, 2 và 5 xong kịp; dự phòng ICML 2027 (khoảng 22/01/2027), COLM 2027 (khoảng 30/03/2027, tạm thời), TMLR.

## 7. Hình đã có (`tools/make_figs.py`, `docs/figs/`)
- `fig1_stride_pointer.png`: độ đúng theo bước nhảy trên tác vụ con trỏ cho sáu mô hình. Chú thích cần ghi: 14B chỉ có bước 4 và 6 (QLoRA 4-bit, một seed); dải bóng là khoảng min–max giữa các seed ở bước có nhiều seed và được nội suy giữa các điểm.
- `fig2_seeds_and_cumsum.png`: trái, độ đúng ở bước 4 theo từng seed (1.5B: 5 seed, 3B: 5, 7B: 3, 14B: 1); phải, cộng dồn không có điểm gãy (các lần chạy chưa hội tụ ở Qwen3-1.7B bước 4 và 6 và Qwen2.5-3B bước 4 vẫn nằm trong hình, cần đánh dấu).
- `fig3_copy_scale.png`: tỷ lệ chép kết quả sai theo quy mô, họ Qwen2.5 (14B và 32B ở 4-bit, nguồn lượng tử hóa khác nhau).
