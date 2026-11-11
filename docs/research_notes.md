# Ghi chú nghiên cứu (bản tra cứu nhanh)

Cập nhật: 2026-10-28. Nhánh `claude/funny-cori-rt5wrv` của `mariorenger/DA`. Nhật ký đầy đủ của các cuộc trao đổi:
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
| Round 13 (hallucination kiểu RAG, MuSiQue) | Điều kiện can thiệp ngữ cảnh: ans / unans (bỏ đoạn bằng chứng) / conflict (thay đáp án); SFT từ chối, SFT mâu thuẫn, DPO, GRPO ba mức, probe trạng thái ẩn (Qwen2.5-3B 2 seed, Qwen3-4B) | Từ chối khi thiếu bằng chứng 0→60–71% (3/3 lần chạy) nhưng độ đúng giảm 4–8 điểm; bám ngữ cảnh mâu thuẫn +21..+24 điểm (3/3); DPO làm xấu (tiện ích −0.2..−0.3); GRPO tăng tiện ích +0.13..+0.26 bằng cách từ chối nhiều hơn (EM −7..−11); probe AUROC 0.66–0.77 (<0.85). Độ đúng gốc 44–61% vẫn là nguồn hallucination chính |
| Round 14 (CoT compression, GSM8K; Qwen2.5-1.5B/3B, Qwen3-1.7B/4B) | Yu PNS vs calibrated PNS vs leave-one-out vs step entropy; SFT on pruned chains | H-P1 NO-GO (Yu bias only at 1.5B), H-P2 GO, H-P3 NO-GO, H-P4 GO but confounded (LOO keeps the answer step, benchmark rewards it), H-P5 NO-GO 0/3: every pruning rule loses 5–46 points in SFT; Round 14c (7B/8B) running |
| Round 14c (7B/8B) | Same estimators at Qwen2.5-7B / Qwen3-8B | H-P6 GO: Yu bias ≈ 0 (0.003 / 0.018); H-P7 GO: SFT on pruned chains loses 13–26 points at 7B |
| Round 15 (answer-blind attribution, placebo calibration) | New estimator + decision rule, 3 small models + 7B audit | H-Q1..H-Q4 all NO-GO (0/3): bloo AUROC 0.61–0.71 < loo in 2/3; bloo_cal −10…−13 vs full. Placebo-calibrated plain LOO is the only arm near full (−3…−4.4) but removes only 4–6% of steps |
| Round 16 (order-reversal control) | Delete least vs most important vs random steps, same count | H-R2 NO-GO, H-R1 GO 2/3: LOO order carries no information about training value; across 42 arms Δacc ≈ 63·token_ratio − 64 (r = 0.94) |
| Round 17 (chuỗi chỉ giữ phép tính, GSM8K) | Giữ phép tính nhưng giảm token vs xóa bước | H-C2 GO 3/3 (hơn xóa ngẫu nhiên +19…+30), H-C1 NO-GO 0/3 (vẫn kém chuỗi đầy đủ 3…17 điểm); nằm trên đường xóa bước ~10–12 điểm. Bộ rút gọn làm mất số ở 15% chuỗi → Round 19 |
| Round 18 (đường cong chép kết quả sai, GSM8K, 0.5B–14B) | Tỷ lệ chép theo kết quả cuối sai khi mô hình tự giải được | 99 → 92 → 85.5 → 66% (1.5B, 3B, 7B, 14B 4-bit); H-S1 NO-GO (14B/3B = 0.72), H-S2 GO (14B vẫn 66%), H-S3 GO; 32B chạy lại |

| Round 20-21 (con trỏ tổng hợp, bước nhảy) | Số phép tra gộp mỗi bước viết ra; chuỗi tối thiểu biết trước | H-T1..H-T3 GO: điểm gãy ở bước 4 (độ đúng 0.35, 0.58, 0.96 cho Qwen2.5 1.5B/3B/7B); Qwen3-1.7B gãy giữa bước 4 và 5; sức chứa không là hàm của số tham số |
| Round 22 (cộng modulo 10; seed) | Tác vụ thứ hai và độ vững qua seed | Cộng modulo không hội tụ (H-G2 không đánh giá được); H-G1 (seed) đang chạy; Round 23 thay bằng cộng dồn không modulo |

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

## 4b. Hướng tiếp theo đã ghi nhận (chưa chạy, 2026-11)
1. **Calibrated PNS để cắt gọn CoT cho mô hình nhỏ** (ưu tiên 1). Yu et al. (NeurIPS 2025) ước lượng PN bằng rollout và giả định mô hình rollout luôn viết tiếp đúng (Lemma 1: P(A=y|do(S))=1). TN2 cho thấy giả định này sai ở mô hình ≤4B (viết tiếp đúng 22–44%, bước không liên quan bị chấm 56–76). Đề xuất: thay baseline cố định bằng baseline theo vị trí b(j), kiểm chứng trên bước không liên quan chèn thêm, rồi dùng để cắt chuỗi thinking của Qwen3-4B. Đo token giảm và độ đúng giữ lại, so với PNS gốc, step entropy (ICLR 2026) và không cắt.
2. **Lọc dữ liệu tự tổng hợp theo từng bước** (cứu Round 12 distillation): giữ chuỗi teacher khi mọi hop đúng theo rule table CLUTRR, kết hợp calibrated PNS.
3. **R_inv của CF-DAPO**: thưởng bất biến đáp án qua đổi tên, đổi thứ tự câu, thêm nhiễu. Chỉ làm sau hướng 1.
4. Từ CausalDANN: thay `[MASKED]` bằng bản viết lại tự nhiên do LLM sinh làm control phân phối; báo cáo hiệu ứng theo nhóm (CATE theo số hop).
5. Liên hệ tài liệu 2026 cần trích dẫn: Liu 2605.22870 (readout shortcut ở 1–3B, cùng hiện tượng với R3 copy), Jia–Benton–Easley 2605.24286 (sufficiency/completeness/necessity qua masked-KL, gần mediation NDE/TIE), Macar–Bogdan et al. ICLR 2026 (resampling on-policy), Li et al. 2610.25876 IGPO (step credit bằng can thiệp tham số trong RL), Hase–Potts 2602.20710 (CST), GRACE 2601.04525 và SABER 2605.18792 (RAG abstention, so với TN9).

## 4c. Ý tưởng phát triển từ các paper 2026 (ghi 10/11/2026; chi tiết ở docs/contribution_plan.md mục 7)
Xếp theo mức liên quan và khả thi với tài sản hiện có (bảng ground truth, đối chứng cùng số bước, đường token, copy test):
1. **BAPO bounds (2602.02909) + đường token của ta:** SFT mô hình nhỏ trên tác vụ tổng hợp có độ phức tạp token tối thiểu đã biết, tìm ngưỡng gãy theo ngân sách token và so với cận lý thuyết. Ứng viên đóng góp chính. Chưa chạy.
2. **Correctness Conditioning as Selection (2610.37070):** đã áp vào copy test Round 8 (làm xong, kết quả ở REVIEW.md): không lệch ≤ 3B, lệch −12.6 điểm ở 7B. Round 18 báo cả hai tỷ lệ.
3. **AttriCoT (ICML 2026, 2606.21821), CMI (2602.03994), NLDD, FACT-E:** thêm vào bảng AUROC ground truth cho bài 1. Chưa làm.
4. **LIFT (2509.13334):** chọn chuỗi theo điểm can thiệp so với ngẫu nhiên cùng kích thước, đánh giá bằng copy test. Chưa làm.
5. Chỉ dùng làm baseline: SPRING, VeriGate, AWA-RL, ACA-RL, Abstain-R1. Bỏ qua: LoRi, latent-CoT.

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
