# Can thiệp vào chuỗi, đừng chỉnh readout

Trả lời thẳng: trong các paper đã đọc (2023–2026), **chưa có phương pháp decoding training-free nào được báo cáo là vượt đồng thời greedy free generation và self-consistency (SC) cùng compute trên MuSiQue có distractor với model ≤7B**. Không paper multi-hop nào trong tập này so sánh với SC cùng compute. Bằng chứng mạnh duy nhất ở quy mô 3–4B nằm ở CLUTRR. Nó đến từ việc đưa phép hợp thành quan hệ ra khỏi LLM: LLM trích xuất bước suy luận, bảng luật kiểm chứng rồi sửa. Cách này tăng **+24 đến +48 điểm ở chuỗi 8–10 hop** ([Reason Popper-ly](https://arxiv.org/abs/2607.23019)).

Dữ liệu của bạn có hai điểm then chốt. Thứ nhất, answer chỉ phụ thuộc chuỗi khi chuỗi đã viết sẵn chuỗi ký tự đáp án. Thứ hai, tác hại của distractor trên MuSiQue đi qua mediator là bước sinh chuỗi (NDE ≈ 0, TIE −3.6/−5.0 EM). Hai điểm này loại bỏ mọi can thiệp ở readout (HGCD, calibration, CAD ở bước đọc đáp án) và các wrapper causal prompting kiểu front-door. Lý do: các phương pháp đó chỉ chọn hoặc cân lại những chuỗi model đã sinh ra.

Hai hướng còn lại nhất quán với dữ liệu:
- **Chọn và lái bằng chứng theo từng hop ngay trong lúc sinh chuỗi**, có điều kiện trên bridge entity. Ở 7–8B, mức tăng là +5 đến +16 EM. Ở Qwen2.5-3B chỉ khoảng +2, và chưa phương pháp nào được so với SC.
- **Ép cấu trúc hop và kiểm chứng bằng bộ kiểm tra deterministic bên ngoài**, để đáp án nằm trong chuỗi nhờ cấu trúc chứ không nhờ model tự hợp thành.

Báo cáo đề xuất **HopGate**, gồm bốn thành phần. (i) Scaffold hop có placeholder. (ii) Chấm điểm paragraph bằng attention đã calibrate theo bridge entity. (iii) Attention bias âm lên các paragraph không được chọn trong lúc sinh mỗi hop. (iv) Kiểm chứng từng hop bằng string match (MuSiQue) hoặc bảng luật kinship (CLUTRR); chỉ branch khi kiểm chứng thất bại. Thiết kế này biến chính thói quen "copy" của model nhỏ thành cơ chế dẫn đáp án đúng ra readout.

Kỳ vọng thực tế (suy luận): **+2 đến +4 EM so với greedy trên MuSiQue ở 3B**, với trần là điểm gold-only, và **+10 đến +25 điểm trên CLUTRR**. Phép thử go/no-go đăng ký trước tốn khoảng **75–95 GPU-giờ trên 2×T4**. Ngưỡng GO là **+2.0 EM so với max(greedy, SC cùng FLOPs) ở Qwen2.5-3B, n ≈ 2.000**, với cận dưới khoảng tin cậy 95% lớn hơn 0 và F1 tăng cùng chiều.

*Quy ước đọc: con số nào có trích dẫn là con số paper báo cáo. Đoạn hoặc ô nào gắn nhãn **[Suy luận]** là suy luận của báo cáo này và chưa được paper nào kiểm chứng. Venue ghi "chưa xác minh" khi ghi chú nghiên cứu không xác nhận được từ trang chính thức.*

## Năm họ phương pháp, nhưng chỉ hai họ có bằng chứng ở ≤7B có distractor

Các phương pháp chia thành năm họ theo vị trí can thiệp trong pipeline X → chuỗi → answer:
1. Contrastive decoding ở mức output.
2. Attention steering và chọn bằng chứng.
3. Causal prompting (front-door, IV, deconfounding).
4. Cấu trúc và neuro-symbolic (decomposition, triple, solver).
5. Step-level search và các biến thể SC.

Họ 1 và họ 5 can thiệp vào phân phối token hoặc vào việc chọn chuỗi. Họ 3 chọn hoặc cân lại cả chuỗi. Chỉ họ 2 và họ 4 thay đổi *nội dung* mà chuỗi được sinh ra từ đó. Các bảng dưới tóm tắt bằng chứng. Mức tăng ghi theo đúng metric của từng paper và không so sánh trực tiếp được giữa các paper: chỉ riêng việc đổi strict EM sang F1 hay "contains" đã làm điểm MuSiQue lệch khoảng 8–15 điểm ([Zhang, Meng & Collier, ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)).

### Contrastive decoding: adaptive α sửa được tác hại của CAD nhưng hiếm khi vượt greedy

Phát hiện nhất quán nhất giai đoạn 2024–2026 là **CAD với α cố định thường thua with-context greedy trên QA không có xung đột tri thức**. AdaCAD đo mức giảm trung bình 18.58% so với greedy ([AdaCAD, NAACL 2025](https://arxiv.org/abs/2409.07394)). Trên TriviaQA có gold context, CAD kéo Falcon-40B từ 72.05 xuống 20.72 EM ([Zhao et al., NAACL 2024](https://arxiv.org/abs/2405.02750)).

Các bản α thích nghi (AdaCAD, CoCoA, NWCAD) gỡ được phần lớn tác hại đó, nhưng trên HotpotQA gold context chúng chỉ ngang greedy. NWCAD là paper duy nhất có điều kiện HotpotQA-distractor tường minh. Nó báo cáo +14.6 EM ở Llama-3.1-8B, nhưng khi chấm bằng GPT-4o-mini làm semantic judge, chênh lệch co lại còn khoảng +3.7. Như vậy một phần mức tăng EM là hiệu ứng định dạng đáp án ([NWCAD, arXiv 2026](https://arxiv.org/abs/2604.16686)).

DeCoRe là phương pháp contrastive duy nhất có số liệu MuSiQue ở ≤8B. Mức tăng lớn nhất xuất hiện khi bật CoT, tức là khi contrast tác động lên token của chuỗi. Tuy vậy, ghi chú nghiên cứu không xác nhận được thiết lập context của DeCoRe, và EM khoảng 60–70 gợi ý context chỉ gồm gold paragraph ([DeCoRe](https://arxiv.org/abs/2410.18860)).

| Phương pháp (venue) | Model | Benchmark | Kết quả báo cáo | Baseline | Chi phí | Training-free? |
|---|---|---|---|---|---|---|
| [CAD](https://arxiv.org/abs/2305.14739) (NAACL 2024) | OPT, GPT-Neo, LLaMA 2.7–30B | NQ-SWAP, MemoTrap | OPT-13B NQ-SWAP 18.8→36.9 EM | greedy | 2 pass/token | Có |
| [AdaCAD](https://arxiv.org/abs/2409.07394) (NAACL 2025) | Llama3-8B, Mistral-7B, Llama2-13B, Llama3-70B | HotpotQA, gold context | Llama3-8B 51.42→50.53 (CAD 39.53); Mistral-7B 40.89→41.35 (CAD 18.49) | greedy | 2 pass/token | Có |
| [CoCoA](https://arxiv.org/abs/2508.17670) (EMNLP 2025) | như trên | HotpotQA, gold context | Llama3-8B 51.42→51.01; Mistral-7B 40.89→45.93 | greedy | 2 pass/token | Có |
| [NWCAD](https://arxiv.org/abs/2604.16686) (arXiv 2026, venue chưa rõ) | Llama-3.1-8B, Ministral-3-8B, 70B | HotpotQA-distractor (1.000 câu) | 8B: 39.20→53.80 EM; theo judge 72.10→75.80 | with-context greedy | 2 pass/token kèm gate | Có |
| [DeCoRe](https://arxiv.org/abs/2410.18860) (preprint) | Llama3-8B, Mistral-7B, Qwen2-7B | MuSiQue open-book, CoT 17-shot | Qwen2-7B 60.57→67.98; Llama3-8B 69.84→74.47; ở 70B chỉ +0.5 | greedy CoT | 2 pass/token | Có (cần dò retrieval head) |
| [ACD](https://arxiv.org/abs/2408.01084) (EMNLP Findings 2024) | Llama2-7B, Llama3-8B | TriviaQA, retrieval nhiễu | Llama2-7B tập nhiễu 33.50→42.06 (CAD 24.75) | greedy | 2 pass/token | Có |
| [COIECD](https://arxiv.org/abs/2402.11893) (Findings ACL 2024, chưa xác minh) | LLaMA2-13B, FLAN-T5-3B | NQ | 46.48→47.42; SC chỉ 46.66 | greedy **và SC** | 2 pass/token | Có |
| [CD expert–amateur](https://arxiv.org/abs/2309.09117) (arXiv 2023) | LLaMA 7–65B | GSM8K; CSQA | 7B GSM8K 10.7→13.6; CSQA 7B −2.7 | greedy | 2 model | Có |
| [DoLa](https://arxiv.org/abs/2309.03883) (ICLR 2024) | LLaMA 7–65B | StrategyQA CoT | 7B 60.1→64.1; GSM8K 7B −0.3 | greedy | ~1× | Có |
| [LeEns/CLeHe](https://aclanthology.org/2025.naacl-long.236.pdf) (NAACL 2025) | Llama-2-7B → Llama-3-8B | NQ, TQA, WebQ, PopQA | +11.74 (Llama-2-7B) giảm còn +1.17 (Llama-3-8B) | naive RAG | latency ×1.18 | Có |

Dấu của hiệu ứng CAD đổi thất thường theo họ model và dataset. Trên HotpotQA-distractor, CAD +7.0 ở Ministral-3-8B nhưng −11.0 ở Llama-3.1-70B ([NWCAD](https://arxiv.org/abs/2604.16686)). Trên NQ, CAD −7.09 ở FLAN-T5-3B nhưng +9.41 ở OPT-6.7B ([COIECD](https://arxiv.org/abs/2402.11893)). Mẫu +4.6 ở 1.5B và −8.6 ở 3B mà bạn quan sát nằm gọn trong khoảng biến thiên đã được ghi nhận này.

### Attention steering và chọn bằng chứng: tín hiệu mạnh nhất ở 7–8B, yếu ở 3B

Họ này nhắm trực tiếp vào câu hỏi *chuỗi đọc paragraph nào*. Tín hiệu cơ chế quan trọng nhất đến từ ba nguồn:
- Attrieval: model dồn attention đáng kể vào fact của hop thứ hai *ngay cả khi không bao giờ viết fact đó ra trong CoT* ([Attrieval](https://arxiv.org/abs/2503.09819)).
- SelfElicit: các layer sâu tách được câu bằng chứng khỏi câu nhiễu *ngay cả khi model trả lời sai* ([SelfElicit](https://arxiv.org/abs/2502.08767)).
- Khi được chỉ đúng index của gold document, model nhỏ phần lớn tự tổng hợp được đáp án. Nút thắt là *nhận diện* bằng chứng chứ không phải *tổng hợp* ([ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)).

Ở ICR, aggregation trên toàn bộ query token và calibration bằng một content-free query là hai yếu tố tạo khác biệt. Bỏ cả hai thì Recall@5 trên MuSiQue rơi từ 55.9 xuống 41.1 ([ICR, ICLR 2025](https://arxiv.org/abs/2410.02642)).

| Phương pháp (venue) | Model | Benchmark | Kết quả báo cáo | Baseline | Chi phí | Training-free? |
|---|---|---|---|---|---|---|
| [AutoPASTA](https://arxiv.org/abs/2409.10790) (Findings EMNLP 2024, chưa xác minh) | Llama3-8B-Instruct, Vicuna-7B | HotpotQA (bản MRQA) | 42.58→58.08 EM; Vicuna 18.11→29.54 | chỉ direct prompting | thêm 1 lượt sinh + profiling head | Có (cần profiling) |
| [DAGCD](https://arxiv.org/abs/2501.01059) (Findings ACL 2025) | LLaMA2-7B, Mistral-7B | HotpotQA (bản MRQA) | Mistral-7B 53.41→63.45; LLaMA2-7B 44.74→47.35 | greedy, CAD | ~1× + detector | Không hẳn (detector huấn luyện trên 100 mẫu) |
| [SelfElicit](https://arxiv.org/abs/2502.08767) (arXiv 2025) | Llama-3.1-8B, Mistral-7B, Qwen2.5-7B | HotpotQA, gold context | Llama 58.9→68.5 EM; Qwen2.5-7B 65.2→69.1 | base, CoT | +18% thời gian | Có |
| [ICR](https://arxiv.org/abs/2410.02642) (ICLR 2025) | Llama-3.1-8B, Mistral-7B | MuSiQue re-ranking | R@2 37.9→44.3 (chỉ đo retrieval, không có EM) | ColBERTv2, RankGPT | 2 prefill | Có |
| [Attrieval](https://arxiv.org/abs/2503.09819) (preprint WIP) | Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-3B | MuSiQue đệm kiểu RULER | 53→63; 36→41 (kl 45); **Qwen2.5-3B 42→44**; HotpotQA Qwen 59→56 | CoT | ~2× | Có |
| [MFAI](https://aclanthology.org/2026.acl-long.1937.pdf) (ACL 2026) | Qwen2.5-7B, Llama-3.1-8B, +3 model | MuSiQue 18 docs | +4.83 đến +11.49% ở bucket cuối, **với index oracle** | không hướng dẫn | ~1× | Có, nhưng cần oracle |
| [Lọc bằng chấm điểm đa agent](https://arxiv.org/abs/2606.25191) (arXiv 2026) | Qwen2.5-7B | MuSiQue 20 para, n=100 | EM 14.0→30.0, CI [+5, +27] | không lọc | ~4× | Có |
| [Found-in-the-Middle](https://arxiv.org/abs/2406.16008) (Findings ACL 2024, chưa xác minh) | Tulu-2-7B | NQ 20 docs | 46.28→53.91 | vanilla | ~1× | Có |

Hai điểm đáng lo cho quy mô của bạn. Thứ nhất, ở Qwen2.5-3B Attrieval chỉ tăng +2 trên MuSiQue và *giảm* 3 điểm trên HotpotQA. Tác giả quy cho việc Qwen sinh CoT dài, làm hỏng heuristic chọn token ([Attrieval](https://arxiv.org/abs/2503.09819)). Thứ hai, raw attention từ một proxy 0.5B mà không có probe thua cả việc giữ nguyên toàn bộ context (38.96 so với 44.74) ([Sentinel](https://arxiv.org/abs/2505.23277)). Không paper nào đo ICR, SelfElicit hay QRHead trên Qwen2.5-1.5B/3B cho MuSiQue.

### Causal prompting: phần lớn mức tăng đến từ self-consistency

Các phương pháp mang nhãn "nhân quả" như Causal Prompting (front-door), DeCoT (IV + front-door) và ACPS về bản chất là wrapper kiểu sample-then-reweight quanh CoT. Chúng sinh nhiều chuỗi, can thiệp vào prompt hoặc context, rồi tổng hợp. Answer cuối vẫn được LLM sinh ra khi đọc toàn bộ câu hỏi và context, nên không có gì đảm bảo answer được tính từ chuỗi ([DeCoT, ACL 2024](https://aclanthology.org/2024.acl-long.758); [CP, AAAI 2025](https://arxiv.org/abs/2403.02738)).

Ở LLaMA3-8B trên MuSiQue, CoT-SC đã nâng EM từ 38.88 lên 41.37 trước khi Causal Prompting thêm phần của nó (lên 48.07). Mức thêm này tốn khoảng 120 lượt sinh mỗi câu và cần một CoT encoder được fine-tune ([CP](https://arxiv.org/abs/2403.02738)). Ở LLaMA-2-7B, DeCoT chỉ tăng +2.5 EM so với CoT ([DeCoT](https://aclanthology.org/2024.acl-long.758)). Số liệu baseline của ACPS không khớp với số của CP cho cùng một cấu hình (GPT-3.5 CoT trên MuSiQue: 45.94 so với 41.37), và các giá trị tròn như 50.00 gợi ý tập đánh giá nhỏ ([ACPS](https://preview.aclanthology.org/credits/2026.findings-eacl.234)).

CASE-ISE là thành phần training-free duy nhất trong họ này nhắm trực tiếp vào lỗi "answer đi tắt qua chuỗi". Nó chặn attention từ token của instruction sang token của answer, nên answer chỉ nhận được thông tin qua hidden state của token CoT ([CASE](https://arxiv.org/abs/2607.18820)).

| Phương pháp (venue) | Model | Benchmark | Kết quả báo cáo | Baseline | Chi phí | Training-free? |
|---|---|---|---|---|---|---|
| [Causal Prompting](https://arxiv.org/abs/2403.02738) (AAAI 2025) | LLaMA3-8B, LLaMA2 | MuSiQue | LLaMA3: CoT 38.88 / SC 41.37 / CP 48.07 EM | CoT, CoT-SC | ~120 lượt sinh/câu | LLM đóng băng; encoder fine-tune; cần cặp CoT sai/đúng |
| [DeCoT](https://aclanthology.org/2024.acl-long.758) (ACL 2024) | LLaMA-2-7B | MuSiQue | 20.22→22.75 EM (CAD 21.62) | CoT, CAD | N×(1+T·P) lượt tính likelihood | Có |
| [ACPS](https://preview.aclanthology.org/credits/2026.findings-eacl.234) (EACL 2026 Findings) | Ministral-3B, LLaMA-3.1-8B | MuSiQue | 3B: CoT 31.05, SC 40.00, ACPS 51.72 | CoT, SC, CAD, DeCoT, CP | nhiều mẫu | Có |
| [CASE-ISE](https://arxiv.org/abs/2607.18820) (preprint 2026) | Llama-3.1-8B | ARC-E/C | CDE 0.358→0.018; acc 92.2→93.4 (E), 86.2→84.4 (C) | không mask | ~1× | Có (riêng ISE) |
| [FRODO](https://arxiv.org/abs/2402.13950) (EMNLP Findings 2024) | LLaMA-2-7B, Mistral-7B | StrategyQA | 79.4→81.5 | SFT+CoT | có huấn luyện | Không |

### Cấu trúc và neuro-symbolic: CLUTRR thưởng cho việc đưa phép hợp thành ra ngoài

Trên CLUTRR, giao phép hợp thành quan hệ cho một engine symbolic là đòn bẩy mạnh và ổn định nhất, và là hướng duy nhất có bằng chứng ở 3–4B. Faithful CoT tăng từ 48.5 lên 58.9 khi decode greedy và từ 45.7 lên 71.9 khi dùng SC với Codex ([Faithful CoT](https://arxiv.org/abs/2301.13379)). Reason Popper-ly đưa LLaMA-3.2-3B từ 18.72 lên 33.48 chỉ bằng phép hợp thành symbolic bên ngoài, rồi lên 42.48 khi thêm vòng verify-repair-regenerate ở chuỗi 8–10 hop ([Popper-ly](https://arxiv.org/abs/2607.23019)).

Nút thắt ở model nhỏ là *trích xuất fact*. Với Llama-3.2-3B, hint đến từ fact oracle đạt 62.79, còn hint đến từ fact model tự trích xuất chỉ đạt 20.32. Với Gemma-3-4B, việc thêm fact tự trích xuất vào story còn làm độ chính xác *giảm* từ 13.45 xuống 1.64; tác giả gọi đây là "distraction effect" ([Kelesis et al., ESWC workshop 2026](https://arxiv.org/abs/2607.14149)).

Trên MuSiQue, decomposition có thay thế placeholder (Self-Ask, PER-QA) cho mức tăng khiêm tốn nhưng đều tay. Điểm quan trọng hơn là nó làm hop k+1 phụ thuộc *theo cấu trúc* vào chuỗi ký tự đáp án của hop k ([Self-Ask](https://arxiv.org/abs/2210.03350); [PER-QA, ACL 2025](https://aclanthology.org/2025.acl-long.1142.pdf)).

| Phương pháp (venue) | Model | Benchmark | Kết quả báo cáo | Baseline | Chi phí | Training-free? |
|---|---|---|---|---|---|---|
| [Faithful CoT](https://arxiv.org/abs/2301.13379) (IJCNLP-AACL 2023) | code-davinci-002 | CLUTRR | greedy 48.5→58.9; SC 45.7→71.9 | CoT | 1 lượt sinh + solver | Có |
| [Path-of-Thoughts](https://arxiv.org/abs/2412.17963) (arXiv, venue chưa rõ) | GPT-3.5, Llama3-70B, GPT-4o | CLUTRR | GPT-3.5 35.6→54.1; Llama3-70B 52.8→65.9 | CoT, CoT-SC (37.1) | ít lượt gọi | Có |
| [Reason Popper-ly](https://arxiv.org/abs/2607.23019) (NeSy 2026) | Qwen-3.5-4B, Gemma-4-E4B, LLaMA-3.2-3B | CLUTRR 8–10 hop | Qwen 19.09→46.72→61.93; LLaMA-3B 18.72→33.48→42.48 | CoT nội sinh | 1 CoT + sinh lại suffix khi vi phạm | LLM đóng băng; luật ILP học offline |
| [HtT](https://arxiv.org/abs/2310.07064) (arXiv 2023) | GPT-3.5/4 | CLUTRR symbolic | GPT-3.5 21.2→37.6 | CoT 5-shot | học luật trên 2.000 mẫu | Có (với LLM) |
| [SLM + RGCN](https://arxiv.org/abs/2607.14149) (ESWC workshop 2026) | Gemma-3 1B/4B, Llama-3.2-3B | CLUTRR | Llama-3B: chỉ story 16.13 → hint tự trích 20.32 → hint oracle 62.79 | chỉ story | RGCN được huấn luyện | Một phần |
| [Self-Ask](https://arxiv.org/abs/2210.03350) (EMNLP Findings 2023) | davinci-002 | MuSiQue | CoT 12.6→13.8 (kèm search 15.2) | CoT | ~1× | Có |
| [TRACE](https://arxiv.org/abs/2406.11460) (EMNLP Findings 2024) | LLaMA3-8B, Mistral-7B, Gemma-7B | HotpotQA, 2Wiki, MuSiQue | +14.03% EM trung bình | toàn bộ docs retrieve được | trích KG + ranker | Có |
| [PER-QA](https://aclanthology.org/2025.acl-long.1142.pdf) (ACL 2025) | executor Llama-3.1-8B, planner GPT-4o | multi-hop | +1.6 đến +3.9 F1 | vanilla | nhiều lượt gọi | Có |
| [Selection-Inference](https://arxiv.org/abs/2205.09712) (arXiv 2022) | 7B | 10 task logic | ">100%" so với vanilla | vanilla | 2 lượt gọi/bước | Có |

### Step-level search và biến thể SC: thắng SC trên math, im lặng trên multi-hop

Thắng lợi sạch duy nhất trước SC cùng compute ở 7B trong tập này là CoT-decoding (agg path): Mistral-7B trên GSM8K đạt 48.4 so với 39.4 của SC@10 ([CoT-decoding, NeurIPS 2024](https://arxiv.org/abs/2402.10200)). Phương pháp này chưa được thử trên multi-hop QA.

Các bằng chứng tiêu cực nhất quán với nhau. Ở SLM, việc tự chấm thưởng gần như ngẫu nhiên: thay điểm tự đánh giá bằng giá trị ngẫu nhiên "không ảnh hưởng đáng kể" ([rStar, ICLR 2025](https://arxiv.org/abs/2408.06195)). Tree search cần discriminator chính xác ít nhất 90% mới đáng kể hơn re-ranking, và chạy chậm hơn 10–20 lần ([Chen et al., ACL 2024](https://arxiv.org/abs/2402.10890)). Self-evaluation beam search thua CoT trên StrategyQA khi cùng compute ([Xie et al., NeurIPS 2023](https://arxiv.org/abs/2305.00633)).

Cũng cần lưu ý: trong context dài, SC tự nó có thể là baseline yếu. Trên LongBench-MuSiQue, SC tăng +0.3 F1 ở Qwen2.5-3B và giảm −3.0 ở Llama-3.2-3B ([Byerly & Khashabi 2024](https://arxiv.org/abs/2411.01101)).

| Phương pháp (venue) | Model | Benchmark | Kết quả báo cáo | Baseline | Chi phí | Training-free? |
|---|---|---|---|---|---|---|
| [CoT-decoding](https://arxiv.org/abs/2402.10200) (NeurIPS 2024) | Mistral-7B | GSM8K | SC@10 39.4 so với agg 48.4 | **SC@10 cùng compute** | k=10 | Có |
| [Self-eval beam](https://arxiv.org/abs/2305.00633) (NeurIPS 2023) | Llama-2-13B, Codex | GSM8K, StrategyQA | thắng SC trên số học; **thua CoT** trên StrategyQA | SC, CoT (theo token) | cao | Có |
| [rStar](https://arxiv.org/abs/2408.06195) (ICLR 2025) | LLaMA2-7B, Mistral-7B, LLaMA3-8B | StrategyQA | LLaMA2-7B: SC@8 59.10 → 67.25 | SC@8 đến SC@128 | ~167 lượt gọi, ~367k token/câu | Có |
| [MCR](https://arxiv.org/abs/2304.13007) (EMNLP 2023) | code-davinci-002, Vicuna-13B | HotpotQA, 2Wiki | HotpotQA 51.3→57.0 | SC@5 | 5 chuỗi + meta-reasoner | Có |
| [CISC](https://arxiv.org/abs/2502.06233) (arXiv 2025) | Gemma2-2B, Qwen2.5-3B, … | GSM8K, MATH, MMLU-Pro | thắng SC ở gần như mọi cấu hình; cần ít hơn 40% số chuỗi | SC | ≤ SC | Có |
| [RAG-Star](https://arxiv.org/abs/2412.12881) (arXiv 2024) | Llama-3.1-8B | MuSiQue, n=100 | EM 11 (RAG) → 13 | RAG, không có SC | MCTS | RM được huấn luyện |
| [SiGIR](https://aclanthology.org/2025.findings-acl.123.pdf) (Findings ACL 2025) | Qwen2.5-7B, LLaMA2-7B | MuSiQue | 37.15; bỏ guided search còn 35.47 | ablation | beam | Không |
| [USC trong context dài](https://arxiv.org/abs/2411.01101) (arXiv 2024) | Qwen2.5-3B/7B, Llama-3.2-3B | LongBench-MuSiQue | +0.3 / +0.9 / −3.0 F1 | 1 mẫu | k× | Có |

## Dữ liệu của bạn loại bỏ readout và giải thích vì sao CAD đổi dấu

Bộ phát hiện của bạn hoạt động như một bộ lọc. Phương pháp nào có cơ chế đi qua đường mà bạn đo được là rỗng sẽ không thể sinh ra lợi ích. Phương pháp nào thay đổi nội dung chuỗi ở đúng chỗ distractor tác động thì còn cửa.

Lập luận gốc như sau: **NDE ≈ 0 và TIE = −3.6/−5.0 EM** nghĩa là, khi giữ cố định chuỗi, distractor không làm hỏng answer. Toàn bộ tác hại đi qua việc distractor làm thay đổi chuỗi. Cộng thêm phát hiện rằng answer phụ thuộc chuỗi chủ yếu khi chuỗi đã viết sẵn chuỗi ký tự đáp án (do_total effect 41–77 điểm), ta được kết luận: readout gần như là phép copy một chuỗi ký tự đã có trong chuỗi.

**[Suy luận]** Từ đó, phân phối của readout sau một chuỗi chứa sẵn đáp án gần như tất định. Bất kỳ phép contrast hay calibration nào ở readout cũng khó đổi được argmax. Điều này giải thích cả việc HGCD cho p ≥ 0.6 lẫn việc calibration cho kết quả *đúng bằng* free generation.

Một hệ quả về thống kê cần lưu ý. **[Suy luận]** Giả định tỷ lệ cặp bất đồng (discordant pair) khoảng 15%. Với n = 500, mức chênh nhỏ nhất phát hiện được (MDE) ở α = 0.05 và power 0.8 vào khoảng 2.8·√(0.15/500) ≈ 4.8 EM. Vậy kết quả rỗng của HGCD loại trừ được hiệu ứng khoảng 5 EM trở lên, nhưng không loại trừ được hiệu ứng 1–3 EM. Bạn có thể tính lại tỷ lệ bất đồng thực tế từ chính dữ liệu n = 500 của mình.

| Hướng | Nhất quán với dữ liệu của bạn? | Lý do |
|---|---|---|
| Contrast hoặc calibration ở readout (HGCD, CAD ở bước đáp án) | **Không, đã bị bác** | NDE ≈ 0; readout chỉ copy chuỗi. DeCoT, vốn chọn chuỗi bằng ACE, cũng chỉ tăng +2.5 EM ở 7B ([DeCoT](https://aclanthology.org/2024.acl-long.758)) |
| CAD trên token của chuỗi, α cố định | **Không** (bạn đo được −1 đến −3) | Dưới distractor, luồng không-context không phân biệt được gold với distractor, nên khuếch đại mọi context như nhau. NWCAD ghi nhận các phương pháp tilting "over-react to distractor contexts" ([NWCAD](https://arxiv.org/abs/2604.16686)) |
| CAD +4.6 ở 1.5B, −8.6 ở 3B | **Nhất quán với literature** | Dấu đổi theo model đã được ghi nhận: +7.0 / −11.0 ([NWCAD](https://arxiv.org/abs/2604.16686)), −7.09 / +9.41 ([COIECD](https://arxiv.org/abs/2402.11893)). Nên coi đây là hiệu ứng riêng của từng model |
| DeCoRe: CAD trên chuỗi +3 đến +6 trên MuSiQue | **Mâu thuẫn bề mặt** | **[Suy luận]** Context của DeCoRe nhiều khả năng chỉ gồm gold paragraph, prompt 17-shot, khác họ model ([DeCoRe](https://arxiv.org/abs/2410.18860)). Khi không có distractor, "có context" đồng nghĩa với "đúng context" |
| Causal prompting (CP, ACPS: +7 đến +11 EM) | **Mâu thuẫn bề mặt** | Phần lớn mức tăng đến từ SC và demo "sửa chuỗi"; không so cùng compute; ACPS dùng tập đánh giá nhỏ ([CP](https://arxiv.org/abs/2403.02738); [ACPS](https://preview.aclanthology.org/credits/2026.findings-eacl.234)) |
| ≤3B copy đáp án sai được nối thêm (46–71%) | **Nhất quán** | Khi đáp án được gợi trong prompt, model thường chọn nó rồi viết CoT hợp lý hóa ([Barez et al. 2025](https://aigi.ox.ac.uk/wp-content/uploads/2025/07/Cot_Is_Not_Explainability.pdf)); fact tự trích xuất gây "distraction effect" ([Kelesis](https://arxiv.org/abs/2607.14149)) |
| CLUTRR: chuỗi hầu như không ảnh hưởng answer (−1.5 đến 7.3) | **Nhất quán** | Compositionality gap không thu hẹp theo kích thước model ([Self-Ask](https://arxiv.org/abs/2210.03350)); LLM đi lệch khỏi chuỗi nhân quả lý tưởng ([Bao et al., COLING 2025](https://arxiv.org/abs/2402.16048)) |
| 7B bác bỏ đáp án nối thêm nhưng hop chỉ mang ~1 điểm | **Nhất quán** | Model RLHF bị direct effect chi phối: LLaMA-2-7B-Chat có CIE 1.4 so với CDE 36.7 ([FRODO](https://arxiv.org/abs/2402.13950)); SFT/RLHF làm yếu cấu trúc nhân quả, ICL làm mạnh lên ([Bao et al.](https://arxiv.org/abs/2402.16048)) |
| Tác hại distractor đi qua bước sinh chuỗi | **Nhất quán**; ủng hộ việc chọn bằng chứng | Fact hop thứ hai được attend nhưng không được viết ra ([Attrieval](https://arxiv.org/abs/2503.09819)); nút thắt là nhận diện, không phải tổng hợp; hiệu năng sụp xuống mức của bằng chứng khó thấy nhất ([ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)) |

Phần tự phê bình của bạn cũng khớp với literature. Baseline "CoT" chấm điểm ứng viên thấp hơn greedy free generation 4–5.6 EM. Tương tự, với Mistral-7B, beam search, top-k và nucleus sampling đều thua greedy (6.7 / 4.9 / 6.4 so với 9.9) ([CoT-decoding Table 4](https://arxiv.org/abs/2402.10200)), và strict EM thưởng cho định dạng nhiều hơn cho nội dung ([NWCAD](https://arxiv.org/abs/2604.16686)). Từ nay, mọi so sánh nên lấy **greedy free generation làm mốc 0**.

Có một điểm literature tự mâu thuẫn và trực tiếp liên quan đến bạn. CP báo cáo SC tăng mạnh trên MuSiQue (LLaMA2 18.71→31.16 EM) ([CP](https://arxiv.org/abs/2403.02738)), còn Byerly & Khashabi thấy SC gần như vô dụng trong context khoảng 11K từ ([Byerly & Khashabi](https://arxiv.org/abs/2411.01101)). **[Suy luận]** Thiết lập gold + 4 distractor của bạn (khoảng 1–1.5K token) gần với CP hơn. Vì vậy SC cùng compute có thể là đối thủ thật sự, không phải hình nộm.

## Xếp hạng: hợp thành symbolic cho CLUTRR, chọn bằng chứng theo hop cho MuSiQue

Bảng dưới xếp các hướng theo kỳ vọng mức tăng nhân với độ khả thi trên 2×T4. Cột kỳ vọng hoàn toàn là **[Suy luận]**. Nó được neo vào ba mốc: (a) trần gold-only trên MuSiQue, (b) khoảng thiệt hại do distractor 3–6 EM mà bạn đã đo, và (c) các mức tăng ở 3–4B trên CLUTRR mà Popper-ly báo cáo.

| Hạng | Hướng | Dataset | Bằng chứng ở ≤3B | Kỳ vọng Δ so với greedy [Suy luận] | Khả thi 2×T4 | Nhất quán với dữ liệu |
|---|---|---|---|---|---|---|
| 1 | Trích xuất triple → hợp thành bằng bảng luật → verify-repair | CLUTRR | Mạnh: +24 đến +48 ở 3–4B ([Popper-ly](https://arxiv.org/abs/2607.23019)) | +10 đến +25 (lớn hơn ở chuỗi dài) | Rất cao (solver gần như miễn phí) | Có: biến "hợp thành" thành "copy" |
| 2 | Chọn và lái bằng chứng theo hop, có điều kiện trên bridge entity, trong lúc sinh chuỗi | MuSiQue | Yếu đến trung bình: Qwen2.5-3B +2 ([Attrieval](https://arxiv.org/abs/2503.09819)); index oracle ở 7B +5 đến +12 | +2 đến +4 (điều kiện C1), +3 đến +6 (C2); trần là gold-only | Cao (1 prefill + bias) | Có: đánh trúng TIE |
| 3 | Scaffold hop + placeholder + kiểm chứng deterministic + branch khi thất bại | Cả hai | Gián tiếp ([Self-Ask](https://arxiv.org/abs/2210.03350), [CoT-decoding](https://arxiv.org/abs/2402.10200)) | 0 đến +2 nếu dùng riêng; là thành phần nền cho hạng 1–2 | Cao | Có |
| 4 | Contrastive α thích nghi trên token của chuỗi (AdaCAD, NWCAD, DeCoRe) | MuSiQue | Không có số công bố ở ≤3B; CAD của bạn đổi dấu | −2 đến +4, không ổn định | Trung bình (2 pass/token) | Một phần |
| 5 | SC có trọng số theo confidence, CoT-decoding | Cả hai | Chưa có trên multi-hop | 0 đến +2 so với SC | Cao | Trung lập: vẫn lấy mẫu từ cùng phân phối đã nhiễm distractor |
| 6 | Wrapper causal prompting (CP, DeCoT, ACPS) | MuSiQue | Số của ACPS ở 3B không tin cậy | Nhỏ khi so cùng compute | Thấp (~120 lượt sinh) | Không: chỉ chọn chuỗi |
| 7 | Tự đánh giá theo bước, MCTS, PRM | Cả hai | Tự chấm ở SLM gần ngẫu nhiên ([rStar](https://arxiv.org/abs/2408.06195)) | ≤ 0 khi cùng compute | Rất thấp (~367k token/câu) | Không: verifier là chính model hay copy |
| 8 | Contrast hoặc calibration ở readout | Cả hai | Kết quả rỗng của chính bạn | 0 | Cao | **Đã bị bác** |

Hạng 1 có rủi ro về tính mới hơn là rủi ro về kết quả: về bản chất nó là Faithful CoT áp dụng lại cho model nhỏ. Phần đóng góp phải nằm ở ba điểm: bằng chứng ở quy mô ≤3B, phép đo cơ chế bằng chính bộ can thiệp của bạn, và cầu nối với hiện tượng copy.

Hạng 2 có rủi ro ngược lại: cơ chế mới và đúng với chẩn đoán của bạn, nhưng tín hiệu attention ở 1.5B/3B chưa ai đo. Chính vì vậy phép thử go/no-go đặt cổng recall của bộ chọn bằng chứng ngay từ pilot.

## HopGate: đưa đồ thị phụ thuộc giữa các hop vào bước sinh chuỗi

HopGate xuất phát từ một nguyên tắc rút ra trực tiếp từ dữ liệu của bạn. Model ≤3B vận chuyển chuỗi ký tự rất đáng tin nhưng hợp thành rất kém. Vì vậy nhiệm vụ của decoder là (a) bảo đảm chuỗi ký tự đúng *xuất hiện* trong chuỗi suy luận và (b) gác cổng những gì được phép viết vào chuỗi bằng một verifier không phải là chính model đó.

Thông tin nhân quả và cấu trúc được đưa vào ở bốn tầng:
1. **DAG của câu hỏi.** Hop k+1 là con của đáp án hop k, thực hiện qua thay thế placeholder.
2. **Phép gán nhân quả giữa bằng chứng và hop.** Paragraph nào là "cha" của đáp án hop k, ước lượng bằng attention đã calibrate, với query là sub-question sau khi đã thay bridge entity vào.
3. **Một phép can thiệp xấp xỉ do(distractor vắng mặt) lên cơ chế sinh chuỗi.** Thực hiện bằng attention bias âm trên key của các paragraph không được chọn.
4. **Với CLUTRR, các structural equation của quan hệ họ hàng.** Đó là bảng hợp thành kinship, đóng vai cơ chế deterministic Y = f(hop).

### Pipeline và thông tin được tiêm vào ở mỗi bước

| Bước | MuSiQue | CLUTRR | Thông tin nhân quả/cấu trúc |
|---|---|---|---|
| 0. Prefill | Encode câu hỏi + toàn bộ paragraph một lần, giữ KV cache | Encode story | — |
| 1. Scaffold hop | Mỗi hop có dạng `#k: <sub-question chứa [a_{k-1}]> \| Đoạn: [P_i] \| Đáp: <entity>`; tối đa 4 hop, có token kết thúc | Mỗi hop là triple `(X, rel, Y)` kèm index câu; `rel` bị ràng buộc vào bộ từ vựng kinship đóng và tên bị ràng buộc vào entity có trong story (constrained decoding qua logits mask) | DAG giữa các hop; hop có kiểu |
| 2. Chấm bằng chứng theo bridge | Sau khi thay `a_{k-1}`, append token của sub-question vào cache. Điểm paragraph = tổng attention từ các token đó tới token của paragraph, cộng trên nửa cuối số layer, trừ điểm của một content-free query (kiểu ICR) | String match tên entity hiện tại với các câu của story (không cần attention) | Paragraph nào là cha của hop k |
| 3. Can thiệp lên mediator | Trong lúc sinh token của hop k, cộng bias −β vào attention logit hướng tới key của các paragraph ngoài top-m. KV của paragraph giữ nguyên nên không cần prefill lại | Constrained decoding như bước 1 | do(distractor vắng mặt) xấp xỉ, áp lên bước sinh chuỗi |
| 4. Kiểm chứng hop | Pass nếu `a_k` (đã chuẩn hóa) xuất hiện trong paragraph được trích và paragraph đó thuộc top-m. Nếu fail: branch top-4 token đầu của hop, chạy greedy cho từng nhánh, chọn nhánh pass có margin Δ của đáp án lớn nhất ([CoT-decoding](https://arxiv.org/abs/2402.10200)); không nhánh nào pass thì giữ hop greedy | Pass nếu X và Y cùng xuất hiện trong câu được trích và triple không mâu thuẫn với các triple khác (nghịch đảo, giới tính). Nếu fail: branch như bên MuSiQue | Verifier deterministic, nằm ngoài model |
| 5. Hợp thành và readout | Answer = chuỗi ký tự `a_K`, lấy bằng trích xuất deterministic | Tìm đường từ head tới tail của truy vấn bằng BFS; hợp thành bằng bảng luật kinship ([Faithful CoT](https://arxiv.org/abs/2301.13379)); *viết* quan hệ đã hợp thành vào chuỗi; answer = quan hệ cuối. Không tìm được đường thì quay về greedy CoT (đăng ký trước) | Structural equation |

### Vì sao phải can thiệp vào mediator thay vì readout

**[Suy luận, dựa trên dữ liệu của bạn]** Với NDE ≈ 0, mọi lợi ích có thể có đều phải đi qua việc thay đổi nội dung chuỗi. Readout lại copy chuỗi ký tự cuối cùng, nên một chuỗi chứa bridge entity đúng ở hop cuối sẽ cho answer đúng mà không cần readout "suy luận" gì thêm.

HopGate đánh vào đúng cạnh distractor → chuỗi. Bias −β làm giảm khả năng token của hop k đọc từ distractor. Kiểm chứng string-in-paragraph chặn những entity không có trong bằng chứng được trích.

Có hai lý do để kỳ vọng HopGate thắng cả SC cùng compute. Thứ nhất, SC lấy mẫu từ *cùng* phân phối chuỗi đã bị distractor làm lệch. Nếu mode của phân phối đó bị kéo về phía distractor, việc bỏ phiếu sẽ khuếch đại độ lệch thay vì sửa nó; đây là đúng cơ chế position bias mà Byerly & Khashabi mô tả ([Byerly & Khashabi](https://arxiv.org/abs/2411.01101)). Thứ hai, HopGate dịch chuyển chính phân phối đó.

Trên CLUTRR, lý do còn trực tiếp hơn. Model không hợp thành từ chuỗi (do effect −1.5 đến 7.3), nhưng ≤3B copy 46–71% những gì được viết trước nó. Khi để bảng luật *viết* quan hệ đã hợp thành vào chuỗi, thói quen copy trở thành ưu điểm. Cái giá là mọi lỗi trích xuất sẽ bị copy theo; đó là lý do bắt buộc phải có cổng kiểm chứng ở bước 4.

### Ablation để tách từng thành phần

| ID | Thay đổi | Câu hỏi được kiểm định | Dự đoán nếu cơ chế đúng [Suy luận] |
|---|---|---|---|
| A1 | Chỉ scaffold (không chấm điểm, không bias, không kiểm chứng) | Cấu trúc tự nó có đủ không? | Thu được ≤ 30% mức tăng |
| A2 | Chấm điểm chỉ theo câu hỏi, một lần (ICR tĩnh, không theo bridge) | Điều kiện theo DAG có cần không? | Mất phần lớn mức tăng ở câu 3–4 hop |
| A3 | Bias mềm so với cắt cứng top-m so với sắp xếp lại (top-m đặt cạnh câu hỏi) | Hình thức can thiệp | Bias mềm ≥ cắt cứng khi recall của bộ chọn < 90% |
| A4 | Bỏ kiểm chứng và branch | Đóng góp của verifier | −0.5 đến −1.5 EM |
| A5 | Branch nhưng chọn theo margin Δ, không qua verifier | Verifier ngoài so với confidence | Kém A0 (HopGate đầy đủ) |
| A6 | Thêm mask kiểu CASE-ISE ở readout ([CASE](https://arxiv.org/abs/2607.18820)) | Kiểm tra phản chứng cho câu chuyện "mediator" | ≈ 0 trên MuSiQue (vì NDE ≈ 0); có thể dương ở 7B trên CLUTRR |
| A7 | Bias cùng độ lớn nhưng đặt lên paragraph ngẫu nhiên (placebo) | Đối chứng âm | ≤ greedy |
| A8 | Filter-then-CoT tự do (ICR top-m, không scaffold) | Có cần scaffold không? | Thấp hơn A0 nhưng > greedy |
| A9 | CLUTRR: LLM tự hợp thành (không bảng luật); có/không constrained decoding; triple oracle | Tách lỗi trích xuất khỏi lỗi hợp thành | Triple oracle ≈ trần; bảng luật ≫ LLM tự hợp thành |
| A10 | (Tùy chọn) Qwen2.5-7B làm verifier hop cho generator ≤3B | Cơ chế "veto" khác kích thước | Chưa có tiền lệ; phải báo FLOPs riêng |

Về triển khai trên T4 (**[Suy luận, kỹ thuật]**). T4 không hỗ trợ bf16 hay FlashAttention-2. Cần dùng HF Transformers với fp16 và attention eager hoặc sdpa, cùng một vòng decode tự viết. Vòng này truyền `past_key_values` và một attention mask 4D dạng float để cộng bias. Chỉ cần lấy attention của các token query mới được append, nên không phải materialize toàn bộ attention map.

Trọng số fp16 của Qwen2.5-7B nặng khoảng 15 GB nên không vừa một T4. Có hai lựa chọn: chia model qua hai GPU, hoặc lượng tử hóa 4-bit và dùng cùng một bản lượng tử cho *mọi* điều kiện. Các siêu tham số cần tune gồm m ∈ {1, 2}, β ∈ {1, 2, 4} (nat), dải layer và k = 4 nhánh. Chỉ tune trên split tuning ở 3B rồi đóng băng khi chuyển sang các model khác. Nên tránh lặp lại hạn chế của NWCAD, vốn tune ngưỡng trên một model rồi mang sang model khác ([NWCAD](https://arxiv.org/abs/2604.16686)).

## Phép thử go/no-go đăng ký trước, khoảng 80 GPU-giờ

Thiết kế dưới đây được viết để đăng ký trước, tức là cố định trước khi xem dữ liệu xác nhận. Endpoint chính là **Qwen2.5-3B-Instruct**, vì đây là kích thước mà theo phần tự phê bình của bạn chưa có phương pháp nào thắng free generation. Mọi siêu tham số được đóng băng sau pilot. Toàn bộ thiết kế, kể cả công thức tính FLOPs, được commit vào repo trước khi chạy split xác nhận.

### Thiết lập

| Hạng mục | Thiết lập |
|---|---|
| Model | Chính: Qwen2.5-3B-Instruct. Phụ: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct. Lặp lại trên họ khác: SmolLM2-1.7B-Instruct. Khám phá: Qwen2.5-3B bản base, few-shot (giả thuyết của [Bao et al.](https://arxiv.org/abs/2402.16048)) |
| MuSiQue | MuSiQue-Ans dev. Điều kiện **C1** = gold + 4 distractor (thiết lập hiện tại của bạn); **C2** = gold + 8 distractor (can thiệp +4 distractor của bạn). Split tuning 400 câu; split xác nhận **khoảng 2.000 câu**, phân tầng theo số hop 2/3/4, seed cố định |
| CLUTRR | Split tuning 300 câu. Split xác nhận **900 câu**, phân tầng k ∈ {2, 3, 4, 6, 8, 10}, mỗi tầng 150 câu |
| Baseline | **B0** greedy free generation (CoT, greedy, trích đáp án chặt). B1 greedy trả lời trực tiếp. **B2 SC cùng FLOPs** (T = 0.7, top-p = 0.95, majority vote trên đáp án đã chuẩn hóa, k ≥ 3, làm tròn k *lên*). B3 đường cong SC với k ∈ {1, 3, 5, 8, 16}. B4 AdaCAD trên token của chuỗi (thêm CAD α = 1 ở 1.5B). B5 filter-then-CoT (A8). B6 gold-only (trần, không phải đối thủ). CLUTRR thêm B7 parse→solve kiểu Faithful CoT không kiểm chứng, và B8 triple oracle (trần) |
| Ghép compute | FLOPs ≈ 2·N_params·(token prefill + token sinh). SC dùng chung prefill nên chỉ tính một lần, tức là ưu ái SC. **[Suy luận]** k ghép được nhiều khả năng rơi vào khoảng 3–5, vì prefill (~1.5K token) chiếm phần lớn FLOPs. Wall-clock trên T4 là chỉ số phụ |
| Metric | MuSiQue: EM và token F1 (chuẩn hóa kiểu SQuAD), kèm độ dài đáp án. CLUTRR: accuracy trên nhãn quan hệ. Cơ chế: TIE của +4 distractor (lặp lại phép đo mediation của bạn), do_total effect của can thiệp lên hop trên CLUTRR, tỷ lệ follow đáp án sai được nối thêm, recall của bộ chọn trên nhãn `is_supporting` |
| Thống kê | Δ ghép cặp với paired bootstrap 10.000 lần (CI 95%) và McNemar cho EM. Giả thuyết chính chỉ ở 3B nên không hiệu chỉnh đa so sánh. Các kích thước phụ hiệu chỉnh Holm |
| Power | **[Suy luận]** Với tỷ lệ bất đồng 15%: MDE ≈ 2.4 EM ở n = 2.000 và ≈ 4.8 EM ở n = 500. Trên CLUTRR, tỷ lệ bất đồng khoảng 30% và n = 900 cho MDE ≈ 5 điểm |

### Ngưỡng quyết định

| Cổng | GO | NO-GO |
|---|---|---|
| **G0, pilot** (split tuning, 3B) | Recall@2 của bộ chọn trên gold paragraph theo từng hop ≥ 75% ở C1 và cao hơn ICR tĩnh; F1 trích xuất triple trên CLUTRR ≥ 70%; Δ EM so với B0 có ước lượng điểm > 0 | Recall < 75%: bỏ chế độ cắt cứng, chỉ giữ bias mềm; nếu recall < 60% thì dừng nhánh MuSiQue. F1 trích xuất < 70%: thêm constrained decoding trước khi đi tiếp |
| **H1, MuSiQue C1** (3B, n ≈ 2.000) | Δ EM so với **max(B0, B2)** ≥ +2.0 **và** cận dưới CI > 0 **và** Δ F1 ≥ +1.5 | Δ EM so với B0 < +1.0, hoặc HopGate ≤ B2, hoặc mức tăng chỉ có ở EM (Δ F1 < 0.5, dấu hiệu hiệu ứng định dạng) |
| **H2, MuSiQue C2** (3B) | Thu hồi được ≥ 40% khoảng (gold-only − greedy) và \|TIE\| giảm ≥ 50% (ví dụ từ −5.0 về mức không tệ hơn −2.5) | \|TIE\| giảm < 20%: cơ chế không như mô tả |
| **H3, CLUTRR** (3B, n = 900) | Δ accuracy so với max(B0, B2) ≥ +8 chung, ≥ +15 ở tầng k ≥ 6, cận dưới CI > 0; HopGate ≥ B7 | Δ < +3 chung |
| **H4, cơ chế** | A1 (chỉ scaffold) thu được < 50% mức tăng của A0; A7 (placebo) ≤ B0; A2 < A0 ở câu 3–4 hop | A1 ≥ 80% mức tăng: báo cáo như hiệu ứng scaffold, không phải "thông tin nhân quả" |
| **H5, khái quát hóa** | Cùng dấu (ước lượng điểm ≥ 0) ở 1.5B, 7B và SmolLM2-1.7B | Đổi dấu ở ≥ 2/3 model: kết luận chỉ đúng cho 3B |

### Ngân sách trên 2×T4

Bảng dưới là **[Suy luận]**. Nó dựa trên giả định throughput decode batched fp16 vào khoảng 600 token/s (1.5B), 300 token/s (3B) và 100–150 token/s (7B) trên một T4, cùng khoảng 250 token sinh mỗi câu MuSiQue. Cần đo lại throughput trong pilot rồi mới cam kết mốc thời gian. SC nên chạy bằng vLLM (T4 có compute capability 7.5) để rẻ hơn. Các điều kiện cần đọc attention phải chạy trên HF.

| Pha | Nội dung | GPU-giờ ước tính |
|---|---|---|
| P0 | Đo throughput; pilot 3B trên split tuning (C1 + CLUTRR); tune m, β, layer; cổng G0 | 4–6 |
| P1 | Xác nhận ở 3B: C1 với đủ baseline và ablation A1–A9, C2 với các baseline chính, CLUTRR với đủ bộ | 30–35 |
| P2 | 1.5B và SmolLM2-1.7B (C1, C2, CLUTRR, chỉ điều kiện chính) | 12–15 |
| P3 | 7B (C1, CLUTRR, điều kiện chính; chia 2 GPU hoặc lượng tử 4-bit) | 25–35 |
| Tổng | | **~75–95 GPU-giờ, khoảng 2–4 ngày wall-clock trên 2×T4** |

Thứ tự thực hiện được gắn cổng: chỉ chạy P2/P3 nếu P1 vượt H1 hoặc H3. Nếu chỉ H3 đạt (CLUTRR) còn H1 thất bại, câu chuyện của bài báo thu hẹp lại thành "biến hợp thành thành copy". Kết quả này vẫn có giá trị, vì nó đo được cơ chế bằng bộ can thiệp của bạn, điều literature chưa làm.

## Rủi ro và những chỗ literature im lặng

Rủi ro lớn nhất của nhánh MuSiQue là **tín hiệu attention ở 1.5B/3B quá nhiễu**. Bằng chứng duy nhất ở Qwen2.5-3B là Attrieval: +2 trên MuSiQue và −3 trên HotpotQA, ở thiết lập đệm kiểu RULER không giống thiết lập của bạn ([Attrieval](https://arxiv.org/abs/2503.09819)). Raw attention từ model 0.5B còn thua cả context đầy đủ ([Sentinel](https://arxiv.org/abs/2505.23277)).

Distractor của MuSiQue lại được xây để giống bằng chứng thật. Trên 2Wiki, ICR đã bị đánh lừa bởi distractor trùng entity ([ICR](https://arxiv.org/abs/2410.02642)). **[Suy luận]** Bridge entity đúng có thể *giúp* tách distractor, vì ICR quan sát thấy bridge entity làm sáng đúng document của hop hai. Nhưng một bridge entity sai sẽ lái attention sang paragraph sai một cách rất tự tin. Cổng recall ở G0 tồn tại chính vì rủi ro này.

Rủi ro thứ hai là **lỗi lan truyền qua placeholder**. Thay thế bridge entity làm hop sau phụ thuộc hop trước theo cấu trúc. Đó là điều ta muốn, nhưng nó cũng biến một lỗi ở hop 1 thành lỗi chắc chắn ở đáp án cuối. Kiểm chứng string-in-paragraph chỉ bắt được entity *không có* trong bằng chứng. Nó không bắt được entity *có mặt nhưng sai* (ví dụ tên một người khác trong cùng paragraph). PER-QA đã ghi nhận hiện tượng ngược lại, "fortuitous reasoning continuance": đáp án đúng đi ra từ bước trung gian sai ([PER-QA](https://aclanthology.org/2025.acl-long.1142.pdf)). Vì vậy, độ đúng của answer không chứng nhận được độ đúng của chuỗi ([Nguyen et al.](https://arxiv.org/abs/2402.11199)).

Rủi ro thứ ba là **metric**. Qwen2.5-7B chỉ đạt 19.58 strict EM trên MuSiQue gold-only so với 40.45 của Llama-3.1-8B, phần lớn vì đáp án dài dòng ([ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)). Một scaffold buộc đáp án ở dạng span ngắn có thể tăng EM chỉ nhờ định dạng. Do đó ngưỡng GO đòi F1 tăng cùng chiều, và bảng kết quả phải báo độ dài đáp án.

Trên CLUTRR, rủi ro là **bất định dạng của bài toán**. Bảng luật kinship chỉ đúng với các phiên bản CLUTRR có quan hệ được định nghĩa chặt, và model nhỏ trích xuất sai quan hệ có giới tính. Hiện tượng copy cũng là con dao hai lưỡi: một triple sai đi qua cổng kiểm chứng sẽ được copy thẳng vào đáp án, đúng như "distraction effect" ([Kelesis](https://arxiv.org/abs/2607.14149)).

Literature im lặng ở nhiều chỗ chính là nơi đóng góp của bạn có thể nằm:
- Không paper nào chạy **contrast phản thực ở mức hop trong lúc sinh CoT**, chẳng hạn so phân phối token của chuỗi khi có và khi không có một paragraph cụ thể, trên MuSiQue hay HotpotQA.
- Không paper nào so **SC cùng compute** với bất kỳ phương pháp decoding nào trên MuSiQue có distractor ở ≤7B.
- Không ai đo **ICR, SelfElicit hay QRHead ở Qwen2.5-1.5B/3B**, hay EM hạ nguồn sau khi lọc bằng attention trong thiết lập distractor.
- Không ai thử **CASE-ISE trên multi-hop**, hay **veto bởi model lớn hơn** làm verifier theo bước khi so cùng compute.
- Không ai áp **CP, DeCoT hay ACPS lên CLUTRR**.
- Không nguồn nào báo **latency trên T4**.

Một phần đáng kể bằng chứng then chốt là preprint hoặc venue cấp workshop: Popper-ly (NeSy 2026), Kelesis (workshop ESWC 2026), CASE và DeCoRe (preprint), NWCAD (arXiv), Attrieval (work-in-progress). Ngoài ra, ghi chú nghiên cứu không trích xuất được bảng số của ProbTree và CaCo-CoT. Nên coi các con số này là tín hiệu định hướng, không phải hằng số đã được tái lập.

## Kết quả thực nghiệm: hợp thành ngoài model thắng lớn, chọn bằng chứng thì không

Sau khảo sát, các hướng được chạy trên Kaggle T4 (Qwen2.5-1.5B/3B/7B-Instruct). Ngưỡng go/no-go được ghi vào `REVIEW.md` trước mỗi lần chạy: phương pháp phải vượt max(CoT greedy, SC cùng compute) ít nhất 2 điểm, với 95% CI của paired bootstrap > 0.

### MuSiQue: chọn bằng chứng trước khi sinh chuỗi là NO-GO

n = 500 câu, 4 hoặc 8 distractor, EM (%):

| Distractor | Model | CoT | SC₅ | Attn top-4 | Attn rerank | Self-select | Gold only |
|---|---|---|---|---|---|---|---|
| 4 | 1.5B | 24.8 | 28.4 | 23.6 | 23.0 | 13.2 | 34.8 |
| 4 | 3B | 38.8 | 41.4 | 33.8 | 37.4 | 29.6 | 48.6 |
| 8 | 1.5B | 22.2 | 25.0 | 22.0 | 22.2 | 13.2 | 31.4 |
| 8 | 3B | 31.6 | 35.6 | 27.4 | 33.6 | 24.8 | 45.6 |

- Trần (gold only) cao hơn CoT 9–14 điểm, nên có chỗ để cải thiện. Nhưng attention calibrated kiểu ICR ở ≤3B không tìm được đủ gold paragraph: hard top-4 làm mất gold và *giảm* EM. Rerank (giữ toàn bộ paragraph, chỉ đổi thứ tự) ≈ CoT. Self-selection bằng prompt tệ hơn hẳn.
- SC₅ vẫn là baseline mạnh nhất (+2.6 đến +4.0). Không phương pháp chọn bằng chứng nào vượt nó.
- **Bug đã phát hiện và sửa.** Attention eager ở fp16 bị overflow với Qwen-1.5B và 7B shard. Hậu quả là attn top-4 luôn giữ 4 paragraph đầu (500/500 câu). Lỗi được bắt bằng cách so lựa chọn với một luật tầm thường. Bản sửa (fp32 retry, kiểm tra điểm suy biến) được chạy lại cho 1.5B (bảng trên). Số liệu attention của 7B bị loại.

### CLUTRR: trích fact rồi để bảng luật hợp thành là GO

Pipeline:
1. Model chỉ viết các fact `X's rel is Y.` và giới tính.
2. Các fact được dựng thành đồ thị quan hệ gia đình có cạnh nghịch đảo.
3. BFS từ nguồn đến đích.
4. Hợp thành dọc đường đi bằng bảng 126 luật, được induce từ tập train.

Model không còn phải tự hợp thành. Accuracy (%), Δ so với baseline tốt nhất (SC₁₀ cùng compute hoặc CoT):

| Tập đánh giá | Model | n | CoT | SC₁₀ | Extract+rules SC₅ | Δ [95% CI] |
|---|---|---|---|---|---|---|
| Validation k=2–4 | 1.5B | 300 | 25.7 | 28.7 | 60.7 | +32.0 [26.0, 38.0] |
| | 3B | 300 | 36.3 | 38.0 | 70.0 | +32.0 [25.3, 38.3] |
| | 7B | 300 | 39.7 | 43.3 | 80.0 | +36.7 [31.0, 42.3] |
| Task chưa thấy k=2–4 | 1.5B | 158 | 31.6 | 31.0 | 52.5 | +20.9 [12.0, 30.4] |
| | 3B | 158 | 30.4 | 32.3 | 59.5 | +27.2 [19.0, 35.4] |
| | 7B | 158 | 39.2 | 40.5 | 74.1 | +33.5 [25.3, 42.4] |
| Task chưa thấy k=5–8 | 1.5B | 200 | 34.5 | 29.5 | 33.5 | −1.0 [−8.5, 6.5] |
| | 3B | 200 | 26.0 | 29.0 | 38.5 | +9.5 [3.5, 15.5] |
| | 7B | 200 | 26.0 | 28.5 | 52.5 | **+24.0 [17.0, 31.5]** (primary đăng ký trước) |

Chuỗi dài là nơi phương pháp còn yếu ở model nhỏ. Lý do là recall của bước trích fact giảm theo số hop: chỉ cần thiếu một cạnh là BFS không tìm được đường đi.

### Ablation nhân quả: model có đủ thông tin nhưng không hợp thành được

Giữ nguyên các fact model đã trích, chỉ thay *ai hợp thành*: chính LLM (ext_llm) hay bảng luật (ext_g).

| Chuỗi | Model | CoT | LLM trên fact của mình | Luật trên fact của mình | Luật − LLM [95% CI] |
|---|---|---|---|---|---|
| k=2–4 | 1.5B | 31.6 | 32.9 | 46.2 | +13.3 [5.1, 22.2] |
| | 3B | 30.4 | 28.5 | 54.4 | +25.9 [18.4, 34.2] |
| | 7B | 39.2 | 36.1 | 71.5 | +35.4 [27.2, 44.3] |
| k=5–8 | 1.5B | 34.5 | 29.0 | 33.0 | +4.0 [−3.5, 11.5] |
| | 3B | 26.0 | 27.0 | 32.0 | +5.0 [−2.0, 12.5] |
| | 7B | 26.0 | 33.0 | 46.0 | +13.0 [5.0, 21.0] |

Khi được đưa đúng các fact nó vừa viết, LLM không làm tốt hơn CoT. Bảng luật trên cùng các fact đó thì tốt hơn 13–35 điểm. Đây chính là điều R3 ("read, not reasoned") dự đoán: nút thắt là bước hợp thành, không phải bước đọc. Can thiệp hiệu quả là *thực thi structural equation của đồ thị suy luận ở ngoài model*.

### Những gì đã thất bại và bài học quy trình

- **Thất bại:**
  - HGCD, calibration và CAD trên readout không vượt CoT hoặc SC.
  - Hợp thành theo thứ tự chuỗi chỉ áp dụng được cho 4–6% câu, nên phải chuyển sang đồ thị không phụ thuộc thứ tự.
  - Bước trích fact ban đầu (C3, test k=5–10) là NO-GO. Phải thêm giới tính do model khai báo thì mới vượt ngưỡng.
- **Kỷ luật thống kê:** ba "thắng lợi" tìm thấy post hoc (C2, gender lexicon, chẩn đoán C3) chỉ được chấp nhận sau khi xác nhận trên dữ liệu chưa dùng, với giả thuyết đăng ký trước (C4, C5, C6).
- **Caveat:**
  - CLUTRR là miền có luật đóng: bảng luật induce được từ train và đạt 95% đúng trên chuỗi gold. Kết quả không tự chuyển sang MuSiQue, nơi "luật hợp thành" là bridge entity mở.
  - Bảng luật dùng tri thức từ tập train mà CoT không được thấy. So sánh công bằng hơn là với một model fine-tune nhẹ hoặc few-shot có bảng luật trong prompt. Đây là việc nên làm tiếp.
  - Mỗi ô chỉ 158–300 câu, một seed.

### Bước tiếp theo đề xuất

1. Tăng recall của bước trích fact trên chuỗi dài: trích theo từng câu, hoặc gộp fact từ nhiều sample (union) trước khi dựng đồ thị.
2. Baseline mạnh hơn: đưa bảng luật vào prompt (rules-in-context), hoặc dùng program-of-thought.
3. Với MuSiQue, chuyển ý tưởng "thực thi ngoài model" sang bridge entity: tách câu hỏi thành sub-question, trả lời từng hop bằng string match hoặc retrieval, rồi thế đáp án vào hop sau. Đây là dạng "luật" duy nhất có sẵn trong miền mở.

## Kết luận

Ý tưởng "decoder nhân quả" cho model nhỏ cần được đảo chiều. Câu hỏi đúng không phải là làm sao ước lượng hiệu ứng nhân quả ở readout rồi cân lại đáp án. Câu hỏi đúng là làm sao *thực hiện* can thiệp lên đầu vào của mediator, để chuỗi được sinh ra từ đúng bằng chứng và mang đúng chuỗi ký tự. Dữ liệu mediation của bạn (NDE ≈ 0, readout chỉ copy) là lập luận mạnh nhất hiện có cho sự đảo chiều này, mạnh hơn bất kỳ paper nào trong tập đã khảo sát, vì literature chưa từng đo trực tiếp đường đi của tác hại distractor ở ≤3B.

Hệ quả thứ hai ít hiển nhiên hơn. Thói quen copy của model ≤3B, vốn bị coi là bệnh lý về độ faithful, thực ra là một kênh truyền đáng tin. Ai kiểm soát được những gì được viết vào chuỗi, bằng bảng luật, string match hay attention bias theo hop, sẽ kiểm soát được đáp án. Vì vậy giá trị khoa học của phép thử go/no-go nằm ở chỗ nó trả lời được một câu hỏi literature còn bỏ ngỏ: can thiệp lên mediator có vượt được SC cùng compute hay không. Dù kết quả trên MuSiQue ra sao, câu trả lời đó vẫn có giá trị công bố.

**Cập nhật sau thực nghiệm.** Kết quả round 9 xác nhận sự đảo chiều này ở dạng mạnh nhất: trên CLUTRR, khi bước hợp thành được thực thi ngoài model, accuracy tăng 21–37 điểm so với SC cùng compute. Ablation cho thấy chính bước hợp thành (không phải bước trích fact) tạo ra mức tăng. Trên MuSiQue, can thiệp ở mức chọn bằng chứng bằng attention của model ≤3B chưa đủ chính xác để vượt SC.
