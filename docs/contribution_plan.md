# Kế hoạch đóng góp và công bố (viết ngày 10/11/2026)

Trạng thái: các Round 14 đến 16 đã chốt, Round 17 (chuỗi chỉ giữ phép tính) và Round 18 (đường cong chép kết quả theo quy mô, kiểm tra quy mô cho Round 13) đang chạy. Kế hoạch dưới đây có hai phần phụ thuộc kết quả đang chờ, đã đánh dấu.

## 1. Lịch hội nghị (kiểm tra ngày 10/11/2026)

| Hội nghị | Hạn nộp | Mức độ chắc chắn của ngày | Còn lại |
|---|---|---|---|
| ICLR 2027 | Đã qua (25/09/2026), kết quả 16/12/2026 | Chính thức (iclr.cc) | Không nộp được. Workshop của ICLR thường nhận bài ngắn vào quý 1/2027, chưa có thông báo |
| **ACL 2027** (Kyoto, 08/2027) qua ARR | **04/01/2027** | Chính thức (trang ACL 2027) | 8 tuần |
| **ICML 2027** | Dự kiến 16 đến 28/01/2027 (tóm tắt trước, bài đầy đủ sau vài ngày) | Chưa công bố, các trang tổng hợp không thống nhất | 10 đến 11 tuần |
| COLM 2027 | Dự kiến 30/03/2027 | Tạm thời | 20 tuần |
| NeurIPS 2027, EMNLP 2027 | Theo quy luật các năm trước khoảng tháng 5 | Chưa kiểm tra | 26 tuần |
| TMLR | Nộp quanh năm | Đã biết | Phương án dự phòng |

Cần kiểm tra lại trực tiếp trên trang chính thức trước khi chốt ngày: ICML 2027, COLM 2027, NeurIPS 2027.

## 2. Đóng góp đề xuất

Câu hỏi nghiên cứu: các phép đo được dùng để chọn hoặc đánh giá bước của chuỗi suy luận (masking, resampling/PNS, leave-one-out, step entropy, độ trung thực dựa vào tắt bước) có đo đúng thứ chúng tuyên bố không, và hệ quả cho việc nén và huấn luyện chuỗi suy luận.

Đóng góp, theo thứ tự độ chắc chắn:

1. **Kiểm chứng bằng ground truth** (đã có): bước thừa cài sẵn và bảng luật CLUTRR làm ground truth. Kết quả: PNS của Yu chỉ lệch ở mô hình 1.5B, entropy dưới ngẫu nhiên, LOO tốt nhất nhưng chỉ 0.66 đến 0.81 AUROC.
2. **Phép đối chứng chuẩn** (đã có, cần thêm miền thứ hai): xóa cùng số bước theo thứ tự ít quan trọng nhất, quan trọng nhất và ngẫu nhiên. Kết quả: các điểm tầm quan trọng không dự đoán giá trị huấn luyện của bước; độ giảm độ đúng sau SFT tuyến tính theo tỷ lệ token (r = 0.94 trên 42 cách cắt, 1.5B đến 7B).
3. **Cơ chế** (Round 17, chờ kết quả): nếu chuỗi chỉ giữ phép tính không kém chuỗi đầy đủ ở khoảng một nửa số token thì độ đúng gắn với lượng phép tính tường minh, không phải với số token. Hướng dẫn nén dựa trên đó.
4. **Quy mô** (Round 18, chờ kết quả): tỷ lệ mô hình chép kết quả sai theo quy mô từ 0.5B đến 32B và một mô hình frontier qua API.
5. **Bộ công cụ và checklist**: mã nguồn, dữ liệu ground truth, danh sách các đối chứng bắt buộc (bước đối chứng, token-matched random, đảo thứ tự, đọc đáp án).

Kết nối với phần luận văn đang có (R1, bias của resampling, copy test, TN9 độ tin cậy RAG): bài 1 là phần chẩn đoán, bài 2 là phần cải thiện độ tin cậy.

### Hai bài, không gộp một

- **Bài 1 (chính):** kiểm chứng và đối chứng phép đo bước CoT, 2 đến 5 ở trên. Nhắm ACL 2027 qua ARR (04/01/2027). Dự phòng: ICML 2027, rồi COLM 2027 (30/03/2027), cuối cùng TMLR.
- **Bài 2:** độ tin cậy đầu ra trong RAG ở 7B đến 14B (từ chối khi thiếu bằng chứng, bám ngữ cảnh mâu thuẫn, thưởng nhiều tầng kiểu GRACE, so với GRACE và SABER). Nhắm EMNLP 2027 hoặc NeurIPS 2027. Cần kết quả Round 18 (kiểm tra quy mô) và một thí nghiệm RL ở 7B.

## 3. Vị trí so với tài liệu (đã xác minh abstract, chưa đọc toàn văn)

| Bài | Chạm vào đóng góp nào | Cách phân biệt |
|---|---|---|
| Breaking the Chain (2603.16475) | Can thiệp cấu trúc trung gian, mỗi chỉnh sửa có một đáp án đúng | Họ đo độ nhất quán sau can thiệp; ta đo chất lượng của các phép đo tầm quan trọng và hệ quả cho huấn luyện |
| NLDD / Reasoning Horizon (2602.11201), Performative traces (2605.11746) | Làm hỏng bước và đo; đáp án ổn định trước khi chuỗi kết thúc | Họ đề xuất và dùng phép đo; ta kiểm chứng phép đo với ground truth |
| Readout Shortcut (2605.22870) | Chép số cuối ở 1 đến 3B | Họ làm với số học, 1 đến 3B; ta thêm điều kiện hóa "tự giải được", đường cong đến 32B |
| Yu (PNS), Step Entropy, IGPO | Chấm bước để nén | Ta chỉ ra điều kiện bias, và đối chứng chuẩn cho thấy không hơn ngẫu nhiên khi token ngang nhau |
| Chain of Draft (2502.18600) | Chuỗi ngắn gọn | Định dạng không mới; ta kiểm tra lời giải thích dưới SFT |

Rủi ro mới: có nhiều bài cùng thời. Trước khi viết cần đọc toàn văn Breaking the Chain, NLDD, Performative traces, Readout Shortcut.

## 4. Việc cần làm trước 04/01/2027 (8 tuần)

| Tuần | Việc | Phụ thuộc |
|---|---|---|
| 1 (từ 10/11) | Đọc toàn văn 4 bài gần nhất; chốt khung bài 1 theo kết quả Round 17 và 18 | Kết quả đang chạy |
| 1 đến 2 | Thêm NLDD và ước lượng kiểu FACT-E vào cùng bảng ground truth | Mã mới |
| 2 đến 4 | Lặp đối chứng đảo thứ tự trên miền thứ hai (CLUTRR chuỗi dài hoặc MuSiQue) vì GSM8K đã bão hòa ở 7B | Kaggle |
| 2 đến 4 | Copy test ở 14B, 32B (4-bit) và một mô hình frontier qua API | **Cần ngân sách API** |
| 3 đến 6 | Chạy lại các so sánh SFT chính với 3 seed (nhiễu hiện tại 3 đến 4 điểm) | Kaggle |
| 5 đến 7 | Viết bài, hình, bảng; phát hành mã và dữ liệu | |
| 7 đến 8 | Duyệt nội bộ với người hướng dẫn; nộp ARR | |

## 5. Tài nguyên cần xin người hướng dẫn

- Ngân sách API khoảng 50 đến 100 USD cho copy test ở mô hình frontier.
- Nếu có thể: giờ A100 80GB (khoảng 100 giờ) cho SFT/QLoRA ở 14B và RL ở 7B (bài 2).
- Kaggle (4 tài khoản, mỗi tài khoản 2 notebook) đủ cho phần còn lại; hạn mức GPU tuần là điểm nghẽn.

## 6. Đánh giá khả thi (thẳng thắn)

- Bài 1 có đóng góp phương pháp luận rõ và dữ liệu đã chạy, nhưng bị cạnh tranh bởi các bài cùng thời. Khả năng tốt nhất là Findings hoặc main track nếu có miền thứ hai, ba seed và ít nhất một mô hình lớn. Không nên kỳ vọng main track của ACL nếu chỉ có GSM8K.
- Kết quả Round 17 quyết định tiêu đề: nếu chuỗi chỉ giữ phép tính hoạt động thì bài có một phát hiện xây dựng; nếu không, đây là bài phân tích thuần túy, mạnh ở kiểm chứng nhưng khó được nhận hơn.
- Bài 2 chưa có phương pháp mới; cần kết quả thưởng nhiều tầng ở 7B mới có đóng góp.
