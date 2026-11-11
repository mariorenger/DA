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

## 7. Phụ lục: ý tưởng phát triển từ các paper công bố mới nhất (khảo sát ngày 10/11/2026, đã xác minh abstract, chưa đọc toàn văn)

| # | Paper | Ý của họ | Phát triển được gì với tài sản hiện có | Rủi ro |
|---|---|---|---|---|
| 1 | **BAPO bounds on CoT token complexity** (2602.02909) | Chứng minh một số tác vụ (đa số, ghép bộ ba, đồ thị liên thông) cần Ω(n) token CoT; thí nghiệm chỉ trên mô hình frontier | Đường tuyến tính độ đúng theo tỷ lệ token (r = 0.94) của ta là kết quả thực nghiệm cùng chủ đề. Thí nghiệm mới: tác vụ tổng hợp có độ phức tạp token tối thiểu đã biết (liên thông trên đồ thị, hợp thành quan hệ kiểu CLUTRR), SFT mô hình nhỏ ở nhiều ngân sách token, tìm ngưỡng gãy và so với cận lý thuyết | Trung bình. Cần sinh dữ liệu tổng hợp; kết quả có thể không khớp lý thuyết |
| 2 | **LIFT** (2509.13334) | Chấm bước bằng thay bằng sự kiện không liên quan, đo giảm log-prob, rồi fine-tune trên chuỗi điểm cao; báo tăng faithfulness | Chọn chuỗi theo điểm LIFT so với chọn ngẫu nhiên cùng kích thước (đối chứng ta đã dùng cho bước); đánh giá bằng copy test độc lập thay vì cùng loại can thiệp lúc huấn luyện | Thấp đến trung bình |
| 3 | **Correctness Conditioning as Selection** (2610.37070) | Chỉ đánh giá faithfulness trên câu đúng làm lệch quần thể; có phân rã hiệp phương sai | Copy test của ta điều kiện hóa trên "tự giải được" nên chịu đúng lỗi này. Việc cần làm: báo cáo cả có và không điều kiện, áp phân rã của họ vào dữ liệu Round 8 và Round 18 | Thấp (phân tích trên dữ liệu có sẵn) |
| 4 | **AttriCoT** (ICML 2026, 2606.21821), **CMI** (2602.03994) | Ước lượng quan hệ nhân quả giữa các đơn vị của một chuỗi (O(U) lần forward); kiểm toán theo tầng bằng activation patching | Thêm vào bảng ground truth (bước thừa cài sẵn, bước mang phép tính, bảng luật CLUTRR) cùng NLDD và FACT-E | Thấp; CMI nặng hơn vì cần patching |
| 5 | SPRING (2609.34660), VeriGate (2605.30451) | Thưởng bước có solver kiểm chứng cho mô hình nhỏ | Bảng luật CLUTRR là bộ kiểm chứng từng hop; dùng làm baseline hoặc phần mở rộng bài 2 | Cao về độ mới (đã đông) |
| 6 | AWA-RL, ACA-RL (EMNLP 2026), Abstain-R1 (ACL 2026 Findings) | RL từ chối khi thiếu bằng chứng cho mô hình nhỏ | Baseline bắt buộc cho bài 2 (Round 13); không đủ mới để làm trục chính | Cao |
| 7 | LoRi (2606.05315), Do Latent-CoT think step by step (2602.00449) | Suy luận ngầm; "fusion muộn" ở bước đọc đáp án | Cùng hiện tượng với phát hiện "đọc đáp án"; chưa có mô hình để thử | Cao, bỏ qua |

Đề xuất thứ tự: (3) làm ngay vì rẻ và sửa lỗi của chính ta; (4) đưa vào bài 1; (1) là hướng có thể thành đóng góp chính vì nối lý thuyết với đường token của ta; (2) làm nếu còn thời gian.

## 8. Cập nhật 11/11/2026: Round 20 là ứng viên đóng góp chính

Round 20 (con trỏ tổng hợp với chuỗi tối thiểu biết trước) cho kết quả sạch ở ba mô hình nhỏ: mọi mô hình đạt 1.00 khi viết mọi bước, có điểm gãy rõ khi mỗi bước viết ra phải gộp 4 phép tra ở Qwen2.5-1.5B (0.99 → 0.35) và Qwen2.5-3B (1.00 → 0.59), còn Qwen3-1.7B giữ 0.95. Điều nó hỗ trợ cho bài 1:
- Ngưỡng ngân sách token là hiện tượng thật, có điểm gãy, khác với suy giảm từ từ mà đường GSM8K gợi ý (r = 0.94 là trung bình trên nhiều tác vụ).
- Sức chứa mỗi bước viết ra không là hàm của số tham số: Qwen3-1.7B hơn Qwen2.5-3B. Điều này cảnh báo không kết luận quy mô từ một họ mô hình.
- Còn chờ: Qwen2.5-7B (H-T3, sức chứa có tăng theo quy mô không) và phần mở rộng Round 21 (điểm gãy của Qwen3).
Việc cần làm để nâng thành đóng góp chính: (1) nhiều seed cho điểm gãy, (2) một tác vụ thứ hai có chuỗi tối thiểu biết trước (hợp thành quan hệ), (3) liên hệ với đường token GSM8K, (4) mô hình 14B trở lên (4-bit).
Các kết quả âm của Round 13 ở 7B/8B nên chuyển bài 2 thành bài đánh giá (huấn luyện từ chối không hơn prompt ở quy mô này) thay vì bài phương pháp.

## 9. Cập nhật 11/11/2026 (chiều): Round 20 và 21 đã chốt

Điểm gãy theo bước nhảy trên tác vụ tra con trỏ (chuỗi tối thiểu biết trước), độ đúng trung bình:
| | bước 1 | 3 | 4 | 5 | 6 | 8 | chỉ đáp án |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1.00 | 0.99 | 0.35 | | | | 0.24 |
| Qwen2.5-3B | 1.00 | 1.00 | 0.58 | | | | 0.29 |
| Qwen2.5-7B | 1.00 | 1.00 | 0.96 | | | | 0.30 |
| Qwen3-1.7B | 1.00 | 1.00 | 0.95 | 0.46 | 0.32 | 0.31 | 0.29 |
| Qwen3-4B | 1.00 | | 1.00 | | 0.59 | 0.39 | 0.36 |
H-T1, H-T2 (2/3), H-T3 và hai giả thuyết thăm dò của Round 21 đều đạt. Điều bài 1 có thể khẳng định: (i) ngưỡng ngân sách token là một điểm gãy chứ không phải suy giảm từ từ; (ii) vị trí điểm gãy khoảng 3 đến 4 phép tra mỗi bước viết ra cho 1.5B đến 7B; (iii) quy mô làm sự sụp đổ nhẹ đi (độ đúng ở bước 4: 0.35, 0.58, 0.96 cho 1.5B, 3B, 7B); (iv) họ mô hình quan trọng hơn kích thước (Qwen3-1.7B bằng Qwen2.5-7B ở bước 4).
Phần chưa có: tác vụ cộng modulo (Round 22) không đo được vì không hội tụ; Round 23 (cộng dồn không modulo, có tiêu chí hội tụ) đang chạy; kiểm tra seed (Round 22 H-G1) đang chạy. Chưa có mô hình ≥ 14B và chưa có liên kết trực tiếp giữa điểm gãy này và đường token của GSM8K.

## 10. Cập nhật 11/11/2026 (tối): hai điều chỉnh quan trọng cho bài 1

1. **Điểm gãy không đúng cho mọi phép tính.** Với phép tra con trỏ, mỗi bước viết ra chỉ gộp được khoảng 3 đến 4 phép tra; với phép cộng dồn các số 1 đến 5, ngay cả chuỗi chỉ có đáp án vẫn cộng đúng 12 số ở 79 đến 85% (không có điểm gãy; độ đúng giảm theo độ sâu k chứ không theo bước nhảy). Vậy khẳng định đóng góp phải là: **sức chứa mỗi bước viết ra phụ thuộc phép tính**, chứ không phải một hằng số "số bước". (Tác vụ cộng modulo 10 không dùng được vì không hội tụ.)
2. **Xu hướng theo quy mô yếu hơn đã nói.** Điểm gãy của Qwen2.5-1.5B lặp lại qua 3 seed (độ đúng ở bước 4: 0.35, 0.33, 0.37) nhưng Qwen2.5-3B thì không (0.58, 0.91, 0.57). Xu hướng 0.35 → 0.58 → 0.96 cho 1.5B, 3B, 7B chỉ dựa trên một seed mỗi cỡ, và điểm 3B không đáng tin. Round 24 (đang chạy) thêm seed cho 3B, 1.5B và 7B.
Còn thiếu cho một bài hoàn chỉnh: ít nhất một phép tính thứ ba (để nói về "phụ thuộc phép tính" cần hơn hai điểm), mô hình từ 14B trở lên, liên kết với đường token GSM8K, và bản nháp cấu trúc bài.

## 11. Cập nhật 11/11/2026 (đêm): Round 24 và bản nháp cấu trúc bài

Round 24 (thêm seed) xác nhận cả hai giả thuyết: 3B dao động mạnh (độ đúng ở bước 4: 0.57 đến 0.91, độ rộng 0.35), 1.5B ổn định (0.33 đến 0.39), 7B luôn ≥ 0.84. Xu hướng theo quy mô ở bước 4 giữ được ở trung bình (0.36 → 0.70 → 0.92). Loss cuối dự đoán độ đúng (Spearman −0.96, 13 lần chạy). Round 25 (14B 4-bit, bước 4 và 6) đang chạy. Bản nháp cấu trúc bài 1 ở `docs/paper1_outline.md`, kèm danh sách khoảng trống.
