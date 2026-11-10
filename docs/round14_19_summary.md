# Tổng hợp Round 14 đến 19 (cập nhật 10/11/2026)

Các Round 18b (32B), Round 13 ở 7B/8B (hall) còn đang chạy. Mọi số liệu dưới đây lấy từ `REVIEW.md` (mỗi vòng đều đăng ký giả thuyết trước khi chạy).

## Câu hỏi
Các phép đo tầm quan trọng của bước trong chuỗi suy luận (Yu PNS, leave-one-out, step entropy, và hai ý mới của chúng ta) có đo đúng thứ chúng tuyên bố không, và dùng chúng để nén chuỗi thì được gì.

## Kết quả đã chốt
| Vòng | Kết quả | Kết luận |
|---|---|---|
| 14 (1.5B–4B) | Mọi cách cắt 40% bước làm SFT mất 5 đến 46 điểm; LOO tốt nhất về AUROC (0.66–0.81), entropy dưới ngẫu nhiên | Không cách chấm bước nào cho nén |
| 14c (7B/8B) | Bias của Yu biến mất (0.003 và 0.018); SFT cắt chuỗi vẫn mất 13–26 điểm ở 7B | Bias chỉ có ở 1.5B; âm tính không do mô hình nhỏ |
| 15 | LOO che đáp án và cắt theo placebo: H-Q1..H-Q4 đều 0/3 | Ý "che đáp án" sai; placebo chỉ là cơ chế an toàn (xóa rất ít) |
| 16 | Cắt bước ít quan trọng nhất không tốt hơn cắt bước quan trọng nhất (2/3), nhưng ở 7B hơn 9.3 điểm | Thứ tự điểm LOO không mang thông tin ở mô hình nhỏ, bắt đầu mang thông tin ở 7B (thăm dò) |
| Phân tích sau | Δ độ đúng ≈ 63 × tỷ lệ token − 64 (r = 0.94, 42 cách cắt) | Mỗi 10% token bớt tốn khoảng 6 điểm bất kể cách chọn |
| 17 và 19 | Chuỗi chỉ giữ phép tính hơn xóa ngẫu nhiên 19–30 điểm và nằm trên đường 10–12 điểm; vẫn kém chuỗi đầy đủ 4–17 điểm; giữ mọi số không giúp thêm | Giữ phép tính có ích, nhưng thay đổi phong cách chuỗi vẫn có giá |
| 18 (0.5B–14B) | Chép kết quả sai: 99 → 92 → 85.5 → 66% (1.5B, 3B, 7B, 14B 4-bit) | Giảm theo quy mô nhưng ở 14B vẫn chép 66% |
| Độ lệch chọn (dữ liệu Round 8) | Δ = −12.6 [−21.8, −2.7] ở 7B; ≈ 0 ở ≤ 3B | "7B chỉ chép 10%" không phân biệt được bác bỏ thật với khó–dễ |

## Điều mạnh nhất có thể nói
1. Các đối chứng chuẩn (cùng số bước, đảo thứ tự, token-matched random) cho thấy các điểm quan trọng của bước không dự đoán giá trị huấn luyện của bước ở 1.5B–3B; ở 7B bắt đầu có (+9.3).
2. Độ đúng sau SFT trên chuỗi đã rút gọn phụ thuộc chủ yếu vào lượng token giữ lại.
3. Việc chép kết quả sai giảm theo quy mô nhưng không biến mất ở 14B.

## Giới hạn
Một benchmark (GSM8K, đã bão hòa ở 7B), một seed mỗi cách (cùng cấu hình lặp lại sai khác ≤ 0.4 điểm, nhưng nhiễu giữa các cách huấn luyện khác nhau khoảng 3–4 điểm), nhiều kiểm định, các mô hình 14B và 32B ở dạng 4-bit.

## Việc tiếp theo
Round 20: ngưỡng "số bước nhảy mỗi token" trên tác vụ tổng hợp có độ sâu biết trước (xem REVIEW.md), đo theo quy mô; sau đó thêm AttriCoT/NLDD vào bảng ground truth và chạy phép đối chứng đảo thứ tự trên miền thứ hai.
