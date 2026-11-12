# Định vị so với tài liệu (đã đọc toàn văn: 12/11/2026)

Đã đọc toàn văn (không chỉ abstract) các bài: BAPO bounds (2602.02909), Breaking the Chain (2603.16475), Readout Shortcut (2605.22870), Performative traces (2605.11746), NLDD (2602.11201, phần kết luận và giới hạn). Chưa đọc toàn văn: Lee và cộng sự 2025 (2503.01141, chỉ abstract), CST, LIFT, FACT-E.

## Điều cần sửa trong cách phát biểu đóng góp
1. **"Ngưỡng ngân sách token" không phải ý mới.** Lee và cộng sự (2025) đề xuất giả thuyết độ phức tạp token (mỗi bài toán có độ dài chuỗi tối thiểu; mô hình thành công nếu chuỗi dài hơn độ dài tới hạn), và BAPO (2026) chứng minh cận dưới Ω(n) cho ba bài toán. **Bài toán đi theo con trỏ nằm ngay trong BAPO (Phụ lục D):** với số tầng và băng thông cố định, κ-pointer chasing là BAPO-hard khi κ lớn (Định lý D.2), còn κ token CoT đủ để giải (Định lý 5.10); κ tầng đủ để giải trong một lần forward (Mệnh đề D.1). Họ chỉ kiểm tra bằng mô hình frontier trên ba bài toán khác, không kiểm tra con trỏ bằng thực nghiệm.
2. **Phần của chúng ta là phép thử thực nghiệm có kiểm soát của dự đoán đó:** SFT mô hình mở ở nhiều quy mô trên chuỗi viết với bước nhảy m biết trước, định lượng điểm gãy (3 đến 4 phép tra mỗi lần forward), độ dao động theo seed, và sự phụ thuộc vào phép tính (cộng ≥ 12). Trung thực: điểm gãy được lý thuyết dự đoán, cái mới là độ lớn thực nghiệm, sự phụ thuộc quy mô, họ mô hình, seed và phép tính.
3. **Số tầng không giải thích sức chứa.** Qwen2.5-7B có 28 tầng như Qwen2.5-1.5B nhưng độ đúng ở bước nhảy 4 là 0.92 so với 0.35; Qwen3-1.7B (28 tầng) đạt 0.95. Trong Qwen2.5 độ đúng tăng cùng độ rộng (hidden 1536 → 2048 → 3584 → 5120: 0.35 → 0.70 → 0.92 → 0.97); giữa các họ thì không (Qwen3-1.7B hidden 2048 đạt 0.95, Qwen2.5-3B cùng độ rộng đạt 0.70). Điều này khớp với việc cận BAPO là về băng thông, nhưng chúng ta không có thí nghiệm tách băng thông khỏi dữ liệu huấn luyện.

## Từng bài
| Bài | Họ làm | Chúng ta thêm | Rủi ro trùng |
|---|---|---|---|
| BAPO (Tomlinson và cộng sự 2026) | Lý thuyết cận token; kiểm tra frontier trên đa số, ghép bộ ba, đồ thị liên thông; con trỏ chỉ lý thuyết | SFT mô hình mở, 0.5B đến 14B, con trỏ và cộng, seed | Cao: câu hỏi "cần bao nhiêu token" là của họ |
| Lee và cộng sự 2025 | Mối quan hệ độ dài chuỗi và độ đúng khi nén bằng prompt; độ dài tới hạn | Huấn luyện thay vì prompt, chuỗi tối thiểu biết trước | Trung bình |
| Breaking the Chain (Somov và cộng sự 2026) | Can thiệp lên cấu trúc trung gian; mô hình vừa tự nhất quán vừa không cập nhật; cấu trúc trung gian là "tín hiệu ngữ cảnh có ảnh hưởng" chứ không phải nút thắt nhân quả; giới hạn: mô hình cỡ vừa | Đường cong chép kết quả sai 0.5B đến 32B (họ ghi giới hạn kích thước) | Trung bình |
| Readout Shortcut (Liu 2026) | Mô hình 1–3B chép số cuối; ngoại suy 7–8B chỉ N = 2; chỉ số học | Đường cong đến 32B, tác vụ con trỏ và cộng, điều kiện hóa tự giải được và độ lệch chọn mẫu | Trung bình |
| Performative traces (Li và cộng sự 2026) | Cam kết đáp án sớm so với đáp án hiện ra (61.9%); 7B–32B | Không cùng thiết kế; liên hệ với kết quả đảo thứ tự cắt | Thấp |
| NLDD (Ye và cộng sự 2026) | Làm hỏng bước, đo giảm logit, "reasoning horizon"; chỉ 6.7B–9B | Ground truth cài sẵn để kiểm chứng các phép đo kiểu này | Thấp |

## Cách đặt bài sau khi đọc
Tiêu đề và đóng góp đề xuất nên chuyển từ "ngưỡng token là điểm gãy" sang **"kiểm chứng thực nghiệm dự đoán lý thuyết về con trỏ và chuỗi ngắn, theo quy mô, họ mô hình và phép tính, kèm kiểm chứng các điểm quan trọng của bước bằng ground truth"**. Đây là bài thực nghiệm mở rộng một lý thuyết có sẵn; khả năng được nhận ở mức cao nhất vì thế thấp hơn so với bài khẳng định hiện tượng mới.
