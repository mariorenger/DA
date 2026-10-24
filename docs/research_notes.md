# Ghi chú nghiên cứu (bản tra cứu nhanh)

Cập nhật: 2026-10-24. Nhánh `claude/funny-cori-rt5wrv` của `mariorenger/DA`. Nhật ký đầy đủ của các cuộc trao đổi:
`docs/conversation_log.md` (xuất tự động, đã che token; tìm bằng `grep -n "từ khoá" docs/conversation_log.md`).
Tạo lại: `python tools/export_transcript.py <session.jsonl> docs/conversation_log.md`.

## 1. Mục tiêu của người dùng (không đổi)
- Luận văn có **kết quả thực nghiệm tốt và tính ứng dụng**, không chỉ chứng minh đúng/sai; chấp nhận mọi phương pháp cải thiện miễn là có framework hoàn chỉnh, áp dụng được.
- Ràng buộc của giáo sư: biến can thiệp nhị phân, đồ thị nhân quả kiểm chứng được, miền hẹp, dữ liệu cẩn thận, tài nguyên hạn chế (Kaggle 2×T4, tuỳ chọn 1 ngày A100).
- Trả lời bằng tiếng Việt, giữ thuật ngữ tiếng Anh. Không tạo PR nếu không được yêu cầu. Commit kèm dòng Co-Authored-By / Claude-Session.

## 2. Dòng thời gian các quyết định
| Mốc | Nội dung | Kết luận |
|---|---|---|
| Đầu | Phản biện đề xuất CF-DAPO (`LLM.pdf`) | Có 3 lỗi thiết kế (R_inv bị triệt tiêu bởi chuẩn hoá advantage, R_nec dùng che một hop, v.v.) |
| R1–R5 | Giao thức ground truth, chẩn đoán | R1 do_naive mù trên bridge; R2 bias 1−r; R3 chuỗi được đọc không được suy luận; R4 distractor hại qua chuỗi; R5 HGCD = CoT |
| Round 9 | Decoding có thông tin cấu trúc | MuSiQue lọc bằng attention: NO-GO; CLUTRR extract+rules: GO (+21..+37 so với SC cùng compute); ablation: bảng luật hơn LLM 13–35 điểm |
| Sau R9 | Hỏi "decode có tốt không" | Decode token: không. Cái tốt là thực thi phép hợp thành ngoài mô hình (rule-based, miền đóng) |
| Round 10 | Pilot: R_nec của CF-DAPO (do_naive) ≈ 0; do_total bị hack bằng chép đáp án | RL chưa nên làm; thử SFT cấu trúc trước |
| Round 10 (xong) | Structured-chain SFT trên CLUTRR (S_plain vs S_struct vs P1; 2 seed) | Tiêu chí chính NO-GO: cấu trúc không hơn fine-tune thường. Phát hiện phụ: fine-tune thường 3B đạt 94.7/77.8/59.0 (val/mới ngắn/mới dài), hơn D2 +15..+25 |
| Round 11 (xong phần CLUTRR) | Chuỗi vs chỉ đáp án; DPO phản thực; GRPO; 7B | Chuỗi hơn chỉ-đáp-án +9..+23 (đề lạ/dài): GO. DPO: chép sai −22..−25 nhưng mất 5–13 điểm chuỗi dài: NO-GO. GRPO: không giúp chuỗi dài (+0.0): NO-GO. 7B = 3B. Sau fine-tune hiệu ứng che bước ~40 điểm (so với 3–10 trước đó) |
| Round 10b (2Wiki xong; MuSiQue chưa chạy do hết hạn mức GPU tuần của tài khoản nguyenpn) | Tổng quát hoá: fine-tune trên chuỗi dựng tự động | 2Wiki 3B: few-shot 28.8, chỉ đáp án 50.0, chuỗi thường 56.0, chuỗi+nguồn 54.4. Tiêu chí chính NO-GO (−1.6 [−4.6, 1.4]); hiệu ứng chính là fine-tune (+21..+30) |
| Round 11 (phần 2) | MuSiQue và 2Wiki chuyển loại câu hỏi (Qwen2.5 1.5B/3B) | MuSiQue: cấu trúc không hơn chuỗi thường (NO-GO), hiệu ứng chính là fine-tune (+20 so với few-shot). 2Wiki train compositional→test inference: chuỗi hơn chỉ-đáp-án +18.0 EM (3B), +8.0 (1.5B): GO. Mất dữ liệu từng câu của MuSiQue do ghi đè thư mục |
| Round 12 (Qwen3 1.7B/4B/8B) | Mô hình mới, thinking mode, distillation từ teacher 8B, GRPO lọc prompt khó | Fine-tune chuỗi vẫn hơn few-shot +33..+49 và hơn thinking +19..+39 (4B). Chuỗi hơn chỉ-đáp-án +21 (C5, 4B). Distillation không chuỗi gold: −25..−32 (NO-GO). GRPO lọc khó: C5 −8.5 (NO-GO) |

## 3. Số liệu chính (xem paper/main.tex, paper/baocao_vi.pdf)
- CLUTRR extract+rules SC5 so với baseline tốt nhất: validation k2–4 +32.0/+32.0/+36.7 (1.5B/3B/7B); task chưa thấy k2–4 +20.9/+27.2/+33.5; k5–8 −1.0/+9.5/+24.0 (7B là kiểm định chính đăng ký trước, GO).
- Ablation C7 (cùng fact, đổi bên hợp thành): luật − LLM = +13.3/+25.9/+35.4 (ngắn), +4.0/+5.0/+13.0 (dài; chỉ 7B có ý nghĩa).
- MuSiQue (n=500): mọi cách chọn bằng chứng đều NO-GO; SC5 +2.6..+4.0; trần gold-only +9..+14.
- R3: ≤3B chép đáp án sai 46–67% kể cả khi tự giải được; 7B chỉ 10% khi tự giải được.
- Pilot R_nec: do_naive 1.0–8.9 điểm mọi nơi; do_total 42–77 (MuSiQue, CLUTRR+copy), 3–11 (CLUTRR thường).

## 4. Trạng thái việc đang chạy / chờ
- (Đã xong) Kaggle: `nguyenpn/hgcd-eg2-musique-ood` (MuSiQue + 2Wiki chuyển loại câu hỏi) đang chạy; các notebook vòng 11 khác đã xong (kết quả `results/cfrl/`, phân tích `tools/analyze_cfrl.py`). `notebooks/run_EG1_musique.ipynb` chưa đẩy được vì tài khoản `nguyenpn` hết hạn mức GPU 30h/tuần. Kết quả 2Wiki: `results/egen/wiki2/`, phân tích `tools/analyze_egen.py`.
- Kết quả round 10 (CLUTRR): `results/cftrain/seed{0,1}/`, phân tích `results/cftrain/analysis.csv`, `tools/analyze_cf.py`.
- Việc tiếp: đo mô hình đã fine-tune bằng giao thức R3/R4 (cần lưu adapter), thêm nhánh s_ans (chỉ đáp án).
- Code: `hgcd/cftrain.py` (SFT cấu trúc CLUTRR), `hgcd/egen.py` (MuSiQue/2Wiki), `tools/build_notebook.py`.
- Chưa làm: DPO/GRPO với cặp phản thực (chỉ khi SFT cấu trúc có tín hiệu), mô hình 7B huấn luyện, miền ứng dụng thật của người dùng.

## 5. Câu hỏi mở / hạn chế đã thừa nhận
- Bảng luật dùng nhãn train; CoT thường không thấy nhãn → cần đối chứng fine-tune (S_plain) và rules-in-prompt (chưa chạy).
- CLUTRR là miền đóng, tổng hợp; tính ứng dụng cần E-gen và dữ liệu thật.
- Bảng luật chỉ phủ ~88% chuỗi train, 106/200 chuỗi bộ k5–8.
- extract+rules gần Faithful CoT / Logic-LM: đóng góp mới nằm ở chẩn đoán + ablation nhân quả.
- Mọi kết quả: một seed (round 10 có 2 seed), n = 158–500.

## 6. Bảo mật
- Token Kaggle chỉ truyền qua biến môi trường `KAGGLE_API_TOKEN` trên dòng lệnh; không ghi vào file/commit/prompt lịch. Đã kiểm tra bằng grep: không có `KGAT_` trong repo ngoài dạng `[REDACTED]`.
- Người dùng nên thu hồi (revoke) mọi token đã gửi trong hội thoại sau khi chạy xong.

## 7. Bản đồ file
| Nội dung | Đường dẫn |
|---|---|
| Paper tiếng Anh / báo cáo tiếng Việt | `paper/main.tex|pdf`, `paper/baocao_vi.tex|pdf` |
| Khảo sát hướng decode | `reports/Giải mã nhân quả đa bước.md` |
| Giả thuyết đăng ký trước và go/no-go | `REVIEW.md` |
| Slide (artifact) | https://claude.ai/artifact/BKvigUqkT5YxXQSSur1azL |
| Kết quả | `results/reliance_v2`, `results/mediation_n500`, `results/decoding`, `results/causaldec` |
| Notebook Kaggle | `notebooks/` |
