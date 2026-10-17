# Contrastive, context-aware and attention-steering decoding (training-free) for multi-hop QA with distractors

Scope note: every number below comes from the PDF text of the cited paper (downloaded from arXiv / ACL Anthology on 2026-10-17 and read directly). Table and section numbers refer to the arXiv version linked. "Venue (unverified)" means I did not confirm acceptance from a primary page. Abbreviations: EM = exact match; SC = self-consistency; CoT = chain-of-thought; "with-context greedy" = standard greedy decoding with the passages in the prompt (the researcher's "free generation" baseline).

## Q1. Which variants exist beyond CAD? Mechanism, models, benchmarks, reported gain, baselines, compute

### Takeaway
Since 2024 the family has split three ways: (i) output-level two-stream contrast (CAD, then adaptive-α versions: COIECD, ACD, AdaCAD, CoCoA, NWCAD; also multi-input/irrelevant-passage contrast); (ii) internal-contrast decoding (DoLa layers, DeCoRe retrieval-head masking, entropy-layer contrast); and (iii) attention or representation steering (PASTA/AutoPASTA, Found-in-the-Middle calibration, attention sorting, DAGCD, SpARE). The consistent 2024–2026 finding is that **fixed-α CAD often loses to plain with-context greedy on non-conflict QA**. Adaptive-α successors fix much of that loss but beat greedy only by small margins, and some do not beat it at all, on gold-context QA. Almost none report SC.

### Cited Findings

**Anchor: CAD (Shi et al., NAACL 2024 short).**
- Mechanism: y_t ~ softmax[(1+α)·logit(y|c,x,y<t) − α·logit(y|x,y<t)]. α=0.5 for summarization and α=1 for knowledge-conflict sets. Cost is 2 forward passes per token. Models: OPT-13B/30B, GPT-Neo-2.7B/20B, LLaMA-13B/30B, FLAN-T5-3B/11B. — [Shi et al. 2023/2024, arXiv 2305.14739 §2–3](https://arxiv.org/abs/2305.14739); [ACL Anthology 2024.naacl-short.69](https://aclanthology.org/2024.naacl-short.69.pdf)
- Gains are on knowledge-conflict sets (Table 3). Examples: OPT-13B NQ-SWAP 18.8→36.9 EM and MemoTrap 32.5→44.5; GPT-Neo-20B MemoTrap 37.1→57.3 and NQ-SWAP 16.1→36.8. On summarization, LLaMA-30B CNN-DM ROUGE-L 25.8→31.8 (Table 2). No multi-hop QA and no SC baseline. — [arXiv 2305.14739](https://arxiv.org/abs/2305.14739)

**COIECD (Yuan et al., 2024; venue Findings of ACL 2024, unverified).**
- Mechanism: an entropy-based constraint classifies each token step as conflicting or not and switches between CAD-style contrast and regular decoding (λ=0.25, α=1 for QA). — [arXiv 2402.11893](https://arxiv.org/abs/2402.11893)
- Table 1 is notable because it **includes a self-consistency (SC) baseline**. LLaMA2-13B NQ EM: Regular 46.48, SC 46.66, CAD 46.79, COIECD 47.42. On FLAN-T5-3B NQ, CAD −7.09 EM vs regular while COIECD +2.84. OPT-6.7B NQ: CAD +9.41 and COIECD +10.33. Datasets: NQ, SQuAD, StrategyQA, Counterfacts. Not multi-hop with distractors. — [arXiv 2402.11893 Table 1](https://arxiv.org/abs/2402.11893)

**ACD: Adaptive Contrastive Decoding for noisy contexts (Kim et al., Findings of EMNLP 2024).**
- Mechanism: α_ACD comes from the entropy of the no-context vs with-context next-token distributions. If the context lowers entropy, trust the context; if it raises entropy, lean toward the parametric distribution. Targets noisy retrieved passages. — [arXiv 2408.01084 §3](https://arxiv.org/abs/2408.01084); [ACL Anthology 2024.findings-emnlp.136](https://preview.aclanthology.org/new-sigs/2024.findings-emnlp.136/)
- Table 1 (EM, TriviaQA "All"): Llama3-8B RegOpn 61.27, **CAD 49.70**, MICD_D 64.01, ACD 66.32. Llama2-7B RegOpn 60.23, CAD 49.02, ACD 64.85. On the TriviaQA noisy-context subset, Llama2-7B goes RegOpn 33.50 / CAD 24.75 / ACD 42.06. Datasets: TriviaQA, NQ, PopQA (Contriever retrieval). No HotpotQA/MuSiQue and no SC. — [arXiv 2408.01084 Table 1](https://arxiv.org/abs/2408.01084)

**Irrelevant-passage contrastive decoding (Zhao et al., NAACL 2024 long).**
- Mechanism: contrast the relevant-context distribution against an adversarial **irrelevant passage** as the negative, instead of against no context. F (fixed) and D (dynamic-α) variants; this is the "MICD" baseline in the ACD paper. — [arXiv 2405.02750](https://arxiv.org/abs/2405.02750); [ACL 2024.naacl-long.237](https://aclanthology.org/2024.naacl-long.237.pdf)
- Table 1 shows **CAD collapsing on TriviaQA**: Falcon-40B Reg-Open 72.05 → CAD 20.72; LLaMA-65B 74.76 → 24.51; Llama-2-70B 76.07 → 31.36. The authors' method recovers to 76–82 (Llama-2-70B Ours-D 81.7). Models: FLAN-T5-11B up to Llama-2-70B; no ≤3B decoder models in the main table. — [arXiv 2405.02750 Table 1](https://arxiv.org/abs/2405.02750)

**AdaCAD (Wang, Prasad, Stengel-Eskin, Bansal; NAACL 2025 long).**
- Mechanism: α_t = JSD(p(y|x,y<t) ‖ p(y|c,x,y<t)) at each token, then CAD with that α. Cost is the same 2 passes as CAD. — [arXiv 2409.07394 §3](https://arxiv.org/abs/2409.07394); [ACL 2025.naacl-long.581](https://aclanthology.org/2025.naacl-long.581)
- Table 1 (zero-shot, greedy, **gold contexts**; HotpotQA column), all EM:

  | Model | Greedy | CAD | COIECD | ConfCD | AdaCAD |
  |---|---|---|---|---|---|
  | Llama2-13B | 38.27 | 31.92 | 42.81 | 35.11 | 37.83 |
  | Llama3-8B | 51.42 | 39.53 | 45.63 | 47.13 | 50.53 |
  | Llama3-70B | 56.11 | 40.86 | 52.03 | 50.14 | 55.00 |
  | Mistral-7B | 40.89 | 18.49 | 35.83 | 29.83 | 41.35 |

  AdaCAD **does not beat greedy on HotpotQA** for 3 of 4 models. Across all tasks, CAD gives an 18.58% average drop vs greedy (§4.2). — [arXiv 2409.07394 Table 1, §4.2](https://arxiv.org/abs/2409.07394)
- Even with tuned α, CAD on HotpotQA (Llama3-8B) reaches 46.81 at α=0.5, vs AdaCAD 50.53 and greedy 51.42 (Table 5). — [arXiv 2409.07394 Table 5](https://arxiv.org/abs/2409.07394)
- No-conflict NQ-SYNTH with Llama3-70B (Table 3): Greedy 88.20, **CAD 64.00**, COIECD 83.60, AdaCAD 86.40. On conflicting NQ-SWAP: Greedy 51.60, CAD 79.60, AdaCAD 62.80. — [arXiv 2409.07394 Table 3](https://arxiv.org/abs/2409.07394)

**CoCoA (Khandelwal et al., EMNLP 2025 main).**
- Mechanism: a token-level gate built from the entropy gap, contextual peakedness (top-1 minus top-2 margin under context) and Rényi divergence between the prior and contextual distributions. — [arXiv 2508.17670 §3.3](https://arxiv.org/abs/2508.17670); [ACL 2025.emnlp-main.348](https://aclanthology.org/2025.emnlp-main.348.pdf)
- Table 1 HotpotQA EM (baseline rows copied from AdaCAD): Llama3-8B greedy 51.42 vs **CoCoA 51.01**, so it still does not beat greedy. Llama3-70B 56.11→59.29, Mistral-7B 40.89→45.93, Llama2-13B 38.27→43.04. — [arXiv 2508.17670 Table 1](https://arxiv.org/abs/2508.17670)

**NWCAD: No-Worse CAD (Tao & Agrawal, arXiv Apr 2026; venue unknown).**
- Mechanism: a two-stage gate on top of the two streams. It backs off to no-context decoding when the context is judged non-informative, and otherwise uses with-context decoding with a CAD-style fallback (CoCoA in the main results) under uncertainty. JS divergence uses a top-K=50 union approximation. Models: Llama-3.1-8B/70B-Instruct and Ministral-3-8B; greedy, max_new_tokens=32. — [arXiv 2604.16686 §4.2](https://arxiv.org/abs/2604.16686)
- Includes **HotpotQA-distractor** (1,000-example subset). EM, Table D.1:
  - Llama-3.1-8B: with-context 39.20, CAD 39.60, AdaCAD 48.40, CoCoA 46.60, NWCAD 53.80.
  - Llama-3.1-70B: with-context 40.38, **CAD 29.38**, AdaCAD 44.50, CoCoA 46.38, NWCAD 53.28.
  - Ministral-3-8B: with-context 36.40, CAD 43.40, AdaCAD 50.40, NWCAD 52.60.
  - On "Distractor-hard" NQ slices with Llama-3.1-8B: with-context 36.20, CAD 11.40, NWCAD 66.00.

  — [arXiv 2604.16686 Table D.1](https://arxiv.org/abs/2604.16686)
- Under GPT-4o-mini semantic judging (Table E.1) the HotpotQA-distractor gaps shrink a lot. Llama-3.1-8B: with-context 72.10, CAD 69.10, AdaCAD 76.70, NWCAD 75.80. Part of the EM gain therefore looks like answer-format effects under strict EM with 32 tokens. — [arXiv 2604.16686 Table E.1](https://arxiv.org/abs/2604.16686)

**Entropy-based document ensemble + layer contrast, LeEns/CLeHe (Qiu et al., NAACL 2025 long).**
- Mechanism: decode each retrieved document in parallel and ensemble them with entropy weights (LeEns). Then contrast against the highest-entropy intermediate layer as a stand-in for parametric knowledge (CLeHe). Targets distraction from retrieved documents. — [ACL 2025.naacl-long.236](https://aclanthology.org/2025.naacl-long.236.pdf)
- Table 1 average gain over naive RAG: Llama-2-7B +11.74, Llama-2-13B +3.18, Mistral-7B +4.25, Llama-3-8B **+1.17**. The gain shrinks as the base model gets stronger. Datasets: NQ, TQA, WebQ, PopQA with top-5/10/20 DPR documents. No multi-hop. — [ACL 2025.naacl-long.236 Table 1](https://aclanthology.org/2025.naacl-long.236.pdf)
- Latency on Llama-2-7B with 20 documents: 32.56 ms/token naive vs 38.49 for CLeHe (×1.18) (Table 3). — [ACL 2025.naacl-long.236 Table 3](https://aclanthology.org/2025.naacl-long.236.pdf)

**Contrastive decoding (Li et al. 2023) for reasoning (O'Brien & Lewis, arXiv 2309.09117).**
- Mechanism: expert vs amateur LM, score (1+β)s_e − βs_a with α-mask plausibility; β=0.5, α=0.1. Applied to **every CoT token**. — [arXiv 2309.09117 §2–3](https://arxiv.org/abs/2309.09117)
- Table 1, GSM8K: 7B 10.7→13.6, 65B 51.0→56.8. Table 2: 65B maj@20 (SC) 68.0→74.0 with CD-sampled chains. Table 3, commonsense: CD **hurts** smaller experts (CSQA 7B −2.7, 30B −3.6). A 7B amateur hurts a 65B expert; a 1.5B amateur helps. — [arXiv 2309.09117 Tables 1–3, Fig. 4](https://arxiv.org/abs/2309.09117)

**DoLa (Chuang et al., ICLR 2024).**
- Mechanism: contrast the final-layer logits with a dynamically chosen premature layer; needs one pass. — [arXiv 2309.03883](https://arxiv.org/abs/2309.03883)
- Table 1, StrategyQA with CoT: LLaMA-7B 60.1→64.1, 65B 70.5→72.9. GSM8K: 7B 10.8→10.5, 13B 16.7→18.0. CD with a 7B amateur worsened StrategyQA (13B 66.6→60.3). — [arXiv 2309.03883 Table 1](https://arxiv.org/abs/2309.03883)

**DeCoRe: contrast with a retrieval-heads-masked copy of the same model (Gema et al., arXiv 2410.18860; "Preprint" in the PDF).**
- Mechanism: mask the top retrieval heads (Wu et al.'s retrieval score) to build a "hallucinating" model. Contrast base − masked with α scaled by the base model's conditional entropy (DeCoRe_entropy). Cost is 2 passes per token; the "lite" variant uses a smaller masked model. — [arXiv 2410.18860 §2–3](https://arxiv.org/abs/2410.18860)
- Baselines: greedy, CD (8B amateur for 70B), CAD, DoLa, Activation Decoding, ITI. No SC. — [arXiv 2410.18860 §3.3](https://arxiv.org/abs/2410.18860)
- MuSiQue numbers are under Q2.

**Retrieval heads (Wu et al., arXiv 2404.15574; ICLR 2025, unverified).**
- Shows that a sparse set of "retrieval heads" does copy-paste retrieval from long context. Masking them breaks needle-in-a-haystack retrieval and CoT. This is the mechanistic basis for DeCoRe. — [arXiv 2404.15574](https://arxiv.org/abs/2404.15574)
- DeCoRe Table 9/18 (Llama3-8B, MuSiQue open-book): masking 100 retrieval heads drops EM from about 58.8 to 38.9 without CoT and from about 69.8 to 23.3 with CoT. Masking the same number of random heads barely changes EM. — [arXiv 2410.18860 Tables 9, 18–19](https://arxiv.org/abs/2410.18860)

**PASTA (Zhang et al., ICLR 2024) and AutoPASTA (Zhang et al., arXiv 2409.10790; Findings of EMNLP 2024, unverified).**
- PASTA: scale down attention to non-highlighted tokens in a profiled subset of heads. Profiling is done once per model. Reported average accuracy +22% over few-shot prompting for LLaMA-7B on JSON formatting, pronoun change, Bias in Bios and CounterFact. Models: GPT-J-6B and LLaMA-7B; no multi-hop. — [arXiv 2311.02262](https://arxiv.org/abs/2311.02262)
- AutoPASTA: the model first names the key sentence(s), per hop for multi-hop, and PASTA then steers attention to them. On HotpotQA, MRQA version, with greedy decoding (Table 2), EM:
  - Llama3-8B-Instruct: direct 42.58 → AutoPASTA in-domain 58.08 (out-of-domain head set 51.96).
  - Vicuna-7B: 18.11 → 29.54.
  - Llama3-70B: 42.63 → 58.32 (out-of-domain) or 50.90 (in-domain).
  - "Iterative prompting" (put the key sentence in the prompt without steering) does not help: 42.10 for Llama3-8B.

  — [arXiv 2409.10790 Tables 2–3](https://arxiv.org/abs/2409.10790)

**Found-in-the-Middle attention calibration (Hsieh et al., arXiv 2406.16008; Findings ACL 2024, unverified) and attention sorting (Peysakhovich & Lerer 2023).**
- Mechanism: remove the U-shaped positional component from attention to documents. Table 5, NQ with 20 documents, Tulu-2-7B average: vanilla 46.28 → calibrated 53.91; attention sorting 44.57. Vicuna-7B-16k average: 56.64 → 58.11, with attention sorting at 62.89, so attention sorting is better for Vicuna. — [arXiv 2406.16008 Table 5](https://arxiv.org/abs/2406.16008)

**DAGCD: Dynamic Attention-Guided Context Decoding (Huang et al., Findings of ACL 2025).**
- Mechanism: a small context-utilization detector over attention maps (trained on 100 HotpotQA samples, so not strictly training-free), gated by normalized output entropy, boosts context-supported tokens. α=2 for base models and 4 for instruction models. — [arXiv 2501.01059 §5.1](https://arxiv.org/abs/2501.01059); [ACL 2025.findings-acl.269](https://aclanthology.org/2025.findings-acl.269/)
- Table 1 HotpotQA EM (MRQA version): LLaMA2-7B greedy 44.74 / CAD 44.13 / COIECD 42.03 / DAGCD 47.35. Mistral-7B 53.41 / 41.57 / 46.43 / 63.45. Mistral-7B-Instruct 58.70 / 49.30 / 59.74 / 60.55. — [arXiv 2501.01059 Table 1](https://arxiv.org/abs/2501.01059)

**SpARE: activation steering for knowledge selection (Zhao et al., arXiv 2410.15999; NAACL 2025, unverified).**
- Mechanism: features from pre-trained SAEs, found at mid-layers, steer the model toward context or toward memory. The abstract claims +10% over other representation-engineering methods and +15% over contrastive decoding on open-domain QA with knowledge conflicts (NQSwap/Macnoise). I did not extract the per-table numbers. — [arXiv 2410.15999 abstract](https://arxiv.org/abs/2410.15999)

**Lookback Lens (Chuang et al., arXiv 2407.07071; EMNLP 2024, unverified).**
- Mechanism: a linear classifier on the ratio of attention to context vs to generated tokens picks among sampled chunk candidates during decoding (guided decoding, so it is multi-sample). Abstract: −9.6% contextual hallucination on XSum with LLaMA-2-7B-chat, and −3.2% when transferred to 13B. — [arXiv 2407.07071](https://arxiv.org/abs/2407.07071)

### Inferences
- Compute: every two-stream method (CAD, COIECD, ACD, AdaCAD, CoCoA, NWCAD, DeCoRe) roughly doubles per-token forward cost (a no-context or masked pass). DoLa and Found-in-the-Middle add almost nothing. AutoPASTA adds one extra generation round for key-sentence identification. LeEns/CLeHe costs ×1.1–1.2 latency but O(k) memory for k documents. All are cheaper than SC with k≥5 samples.
- In terms of what could beat strong baselines, the 2025–2026 evidence on HotpotQA with distractors ranks NWCAD ≳ AdaCAD/CoCoA > with-context greedy ≳ CAD under strict EM. But the semantic-judge rerun in NWCAD shrinks this to about ±4 points, so EM gains may partly reflect shorter or cleaner answer strings rather than better reasoning.

### Gaps
- I could not verify the numbers for ICD (induced-hallucination contrast), SH2, Activation Decoding, PH3 head-pruning ("Cutting off the head ends the conflict"), JUICE, or "focus directions". Searches did not surface primary pages within budget.
- Several venues (COIECD, AutoPASTA, Found-in-the-Middle, SpARE, Lookback Lens, retrieval heads, DeCoRe, Attrieval, NWCAD) are listed from memory or not stated in the PDF.

## Q2. Which methods were evaluated on multi-hop QA with distractors (MuSiQue, HotpotQA, 2Wiki) with ≤7B models, and do gains hold against greedy, SC and across scales?

### Takeaway
Very few were. DeCoRe is the only output-level contrastive method I found with MuSiQue numbers: ≤8B models, CoT, 17-shot, open-book. Its context setup is not stated and the high EM (about 60–70) suggests gold-only paragraphs. NWCAD (2026) is the only one with an explicit HotpotQA-distractor condition. AdaCAD, CoCoA, DAGCD and AutoPASTA use gold or MRQA HotpotQA contexts. **No paper in this set compares against SC on multi-hop QA. Gains shrink or flip at larger scale, and CAD's effect varies in sign across model families**, which matches the researcher's +4.6 at 1.5B / −8.6 at 3B pattern.

### Cited Findings
- **DeCoRe on MuSiQue** (Table 3 and Table 22), EM, open-book. Closed-book scores are all ≤21.

  | Model | Setting | Base | CAD | DoLa | DeCoRe_entropy |
  |---|---|---|---|---|---|
  | Llama3-8B-Instruct | no CoT | 58.83 | **57.88** | 59.08 | 61.98 |
  | Llama3-8B-Instruct | CoT | 69.84 | 73.02 | 69.92 | 74.47 |
  | Llama3-70B-Instruct | no CoT | 68.56 | 68.64 | 68.68 | 69.84 |
  | Llama3-70B-Instruct | CoT | 74.43 | **74.02** | 74.64 | 74.93 (static 75.05) |
  | Mistral-7B-v0.3 | no CoT | 58.01 | **50.10** | — | 62.72 |
  | Mistral-7B-v0.3 | CoT | 59.70 | 63.55 | — | 65.12 |
  | Qwen2-7B-Instruct | no CoT | 63.01 | 64.58 | — | 66.49 |
  | Qwen2-7B-Instruct | CoT | 60.57 | 66.41 | — | 67.98 |

  Two patterns: the gain shrinks from about +4.6 at 8B to about +0.5 at 70B with CoT, and CAD's sign varies by model family in the no-CoT setting. — [arXiv 2410.18860 Tables 3, 22](https://arxiv.org/abs/2410.18860)
- DeCoRe MuSiQue prompts are 17-shot ("Answer the following question by reasoning step-by-step"). The table of settings does not describe paragraph selection or distractors for open-book MuSiQue. — [arXiv 2410.18860 Table 26, App. J](https://arxiv.org/abs/2410.18860)
- **HotpotQA with explicit distractors, NWCAD Table D.1 (EM):** CAD −11.0 vs with-context at 70B (40.38→29.38), but +0.4 at Llama-3.1-8B and +7.0 at Ministral-3-8B. AdaCAD gives +9.2 / +4.1 / +14.0 and NWCAD +14.6 / +12.9 / +16.2 for the three models. — [arXiv 2604.16686 Table D.1](https://arxiv.org/abs/2604.16686)
- **HotpotQA with gold contexts (AdaCAD/CoCoA Table 1):** CAD loses 6–22 EM to greedy on all four models. AdaCAD is within ±1.1 of greedy. CoCoA beats greedy except on Llama3-8B (−0.41). — [arXiv 2409.07394 Table 1](https://arxiv.org/abs/2409.07394); [arXiv 2508.17670 Table 1](https://arxiv.org/abs/2508.17670)
- **HotpotQA, MRQA version (DAGCD Table 1):** CAD loses to greedy on every model (e.g., Mistral-7B 53.41→41.57). DAGCD gains between +1.85 (Mistral-7B-Instruct) and +10.04 (Mistral-7B base). — [arXiv 2501.01059 Table 1](https://arxiv.org/abs/2501.01059)
- **AutoPASTA on HotpotQA:** +15.5 EM for Llama3-8B-Instruct and +11.4 for Vicuna-7B over direct prompting, which is the only baseline (no greedy CoT, no SC). — [arXiv 2409.10790 Table 2](https://arxiv.org/abs/2409.10790)
- **Attrieval: attention-guided fact retrieval from CoT attention (Zhang et al., arXiv 2503.09819; venue unverified).**
  - Training-free. It reads the attention weights of the generated CoT tokens over the context, retrieves the top-n=10 facts, and re-inserts them into the reasoning.
  - MuSiQue overall (RULER-style padding, 100 examples per length; Table 1): Llama-3.2-3B CoT 36 → 41 (kl variant 45); Llama-3.1-8B 53 → 63; Qwen2.5-3B 42 → 44.
  - HotpotQA: no gain, e.g., Qwen2.5-3B 59 → 56 and Llama-3.2-3B 63 → 63.
  - Baseline is CoT only, not SC.

  — [arXiv 2503.09819 Table 1](https://arxiv.org/abs/2503.09819)
- **Weakest Link Effect (arXiv 2601.12499):** on MuSiQue and NeoQA, "multi-hop reasoning performance collapses to the level of the least visible evidence, governed by absolute position". Matched attention instruction (a prompt, not decoding) improves low-visibility positions by up to 11.49%. — [arXiv 2601.12499 abstract](https://arxiv.org/abs/2601.12499)
- **SC baseline:** among all papers read, only COIECD (single-hop NQ/SQuAD/StrategyQA/Counterfacts) and O'Brien & Lewis (math and commonsense, maj@20) report SC. In COIECD, SC barely moves (LLaMA2-13B NQ 46.48→46.66). — [arXiv 2402.11893 Table 1](https://arxiv.org/abs/2402.11893); [arXiv 2309.09117 Table 2](https://arxiv.org/abs/2309.09117)

### Inferences
- For MuSiQue with distractors at ≤7B, there is **no published decoding-time result I could find that beats greedy CoT plus SC**. The closest evidence (DeCoRe) uses likely-gold contexts, 17-shot prompts, and no SC. Its gains of +4.6 to +7.4 EM at 7–8B are about the size SC usually gives, so the question is open.
- The researcher's CAD result (+4.6 at 1.5B, −8.6 at 3B) fits the literature: CAD's effect on with-context greedy varies by model and dataset. Examples are −7.9 (Mistral-7B, DeCoRe no-CoT), +7.0 (Ministral-3-8B, HotpotQA-distractor) and −11.0 (Llama-3.1-70B, HotpotQA-distractor). The pattern is not a clean scale trend.
- The best-supported candidates to test against greedy+SC on MuSiQue at ≤7B are: (a) DeCoRe_entropy, with retrieval heads computed for Qwen2.5; (b) AdaCAD or NWCAD-style gating, because these mostly remove CAD's downside; and (c) AutoPASTA-style per-hop key-sentence attention steering, which targets distractor attention directly and gave the largest HotpotQA gains at 7–8B. Option (c) also matches the researcher's finding (b) that harm passes through chain generation.

### Gaps
- No paper reports CAD/AdaCAD/DeCoRe on 2WikiMultiHopQA, or on MuSiQue in the standard 20-paragraph distractor setting with Qwen2.5 or SmolLM. No ≤3B results exist except Attrieval (3B) and the researcher's own.
- DeCoRe's open-book MuSiQue context (gold-only vs all 20 paragraphs) could not be confirmed from the PDF.

## Q3. Is there work applying contrast at the chain-generation step (token- or step-level during CoT), rather than only at the final answer?

### Takeaway
Yes, but mostly by accident of design. Token-level methods (CD, DoLa, CAD, DeCoRe) run on every generated token, CoT included. DeCoRe is the only one that reports CoT-on vs CoT-off on MuSiQue, and its CoT setting is where contrast helps most. No paper I found contrasts the chain with and without a specific hop or distractor at the step level. Recent "step-level" contrastive decoding work targets math reasoning or CoT length, not context faithfulness.

### Cited Findings
- DeCoRe states that "retrieval heads may be crucial in this process as they allow models to reference earlier generated tokens". The largest gains come with CoT in open-book MuSiQue: Llama3-8B +4.63, Mistral-7B +5.42, Qwen2-7B +7.41 EM. CAD applied over CoT tokens also gains in that setting (+3.18 / +3.85 / +5.84), while CAD without CoT is mixed (−0.95 / −7.91 / +1.57). — [arXiv 2410.18860 §4, Tables 3, 22](https://arxiv.org/abs/2410.18860)
- DeCoRe_entropy gives lower length-normalized entropy in open-book MuSiQue CoT generations. Entropy correlates with correctness (Fig. 7, Tables 12–13, 28). — [arXiv 2410.18860 App. K](https://arxiv.org/abs/2410.18860)
- O'Brien & Lewis apply expert–amateur CD on CoT tokens and sample CoT chains for SC. GSM8K 65B maj@20: 68.0→74.0. With smaller experts it hurts commonsense CoT (CSQA 7B −2.7). — [arXiv 2309.09117 Tables 2–3](https://arxiv.org/abs/2309.09117)
- DoLa on StrategyQA CoT (multi-hop implicit reasoning, no context): +1 to +4 points; GSM8K −0.3 at 7B. — [arXiv 2309.03883 Table 1](https://arxiv.org/abs/2309.03883)
- AutoPASTA identifies key sentences "for each individual hop separately" before answering. The steering is applied during answer generation, not chain generation. — [arXiv 2409.10790 §3](https://arxiv.org/abs/2409.10790)
- Attrieval uses the **CoT tokens' attention** to find implicit second-hop facts the CoT failed to write out ("the model allocates substantial attention to second-hop factual relationships, even when these fail to surface in CoT generations"), then re-injects them. This is chain-level but retrieval-based, not contrastive. — [arXiv 2503.09819 abstract, Table 1](https://arxiv.org/abs/2503.09819)
- Thinking by Subtraction / CCD (arXiv 2602.18232): finds low-confidence tokens in a reasoning chain and subtracts a "confused" reference built by replacing high-confidence tokens with placeholders. Evaluated on math (AIME-style); the KV-cache overhead is minimal. — [arXiv 2602.18232 abstract](https://arxiv.org/abs/2602.18232)
- DoubtLess (arXiv 2610.18862): a step-wise logit difference from a small model prompted brief vs verbose shortens Qwen3-32B CoT by 31% with unchanged accuracy (per the search snippet; not read in full). — [arXiv 2610.18862](https://arxiv.org/abs/2610.18862)

### Inferences
- The researcher's result (CAD on chain tokens loses 1–3 points to free generation) differs from DeCoRe's (CAD on CoT +3 to +6 on MuSiQue). The most likely explanations are distractor presence (DeCoRe's open-book probably gold-only), 17-shot prompting, and model family; this is unverified. Under distractors the no-context stream gives no signal about *which* passage to trust, so CAD-on-chain amplifies all context equally, distractors included. A contrast whose negative stream is "context with gold hops removed" or "irrelevant-passage context" (Zhao et al. 2024 style) is more aligned with the indirect-effect finding.
- Because the answer is mostly copied from the chain, interventions on retrieval or copy heads during chain generation (DeCoRe, PASTA on hop sentences) target the mediator the researcher identified. Readout-level contrast does not, which matches the researcher's null result in (a).

### Gaps
- I found no paper that runs a hop-level counterfactual contrast during CoT (e.g., contrasting the chain-token distribution with vs without a particular supporting or distractor paragraph) on MuSiQue or HotpotQA.

## Q4. Known failure modes and negative results (scale dependence, hurting when context is right, α sensitivity)

### Takeaway
The failure modes are well documented. Fixed-α CAD badly hurts no-conflict and gold-context QA, by up to −50 EM on TriviaQA with large models. It "over-reacts" to type-matched distractors and flips correct answers (neutral regression). Its benefits depend on model, dataset and α. Adaptive-α methods mostly remove the downside but rarely add much over greedy on realistic QA. Gains of internal-contrast methods shrink with scale.

### Cited Findings
- AdaCAD: "CAD tends to overemphasize irrelevant tokens in the vocabulary, leading to incorrect answers in low-conflict examples" (Fig. 2). CAD degrades about 24% vs greedy on no-conflict NQ-SYNTH (Llama3-70B: 88.20→64.00), and averages −18.58% across models and tasks. — [arXiv 2409.07394 §4.2, §5.1, Table 3](https://arxiv.org/abs/2409.07394)
- α sensitivity: CAD with α tuned per dataset (0.25–0.5) still trails greedy on HotpotQA with Llama3-8B (46.81 vs 51.42). The CAD paper itself uses α=0.5 for summarization and α=1 for conflict tasks. — [arXiv 2409.07394 Table 5](https://arxiv.org/abs/2409.07394); [arXiv 2305.14739 §3](https://arxiv.org/abs/2305.14739)
- Zhao et al. 2024: CAD on TriviaQA with gold context drops Falcon-40B 72.05→20.72 and LLaMA-65B 74.76→24.51 EM. — [arXiv 2405.02750 Table 1](https://arxiv.org/abs/2405.02750)
- ACD: CAD lowers EM on noisy retrieved contexts too (Llama3-8B TriviaQA noisy subset 36.02→27.31). — [arXiv 2408.01084 Table 1](https://arxiv.org/abs/2408.01084)
- NWCAD: "continuous tilting methods appear to over-react to distractor contexts (large drops on distractor-hard BC)". With-context decoding itself regresses on baseline-correct items under type-matched distractors. Example: CAD, AdaCAD and CoCoA change the correct "January 1, 1979" to "1978". On Distractor-hard with Llama-3.1-70B, EM: with-context 36.40, CAD 26.60, AdaCAD 30.40, CoCoA 16.80. — [arXiv 2604.16686 §4.5, Tables 2, D.1](https://arxiv.org/abs/2604.16686)
- NWCAD limitation: its thresholds are tuned on Llama-3.1-8B and transferred unchanged to the other models. — [arXiv 2604.16686 Limitations](https://arxiv.org/abs/2604.16686)
- Scale dependence:
  - DeCoRe MuSiQue CoT gain: +4.6 EM at 8B vs +0.5–0.6 at 70B. — [arXiv 2410.18860 Table 3](https://arxiv.org/abs/2410.18860)
  - CLeHe gain falls from +11.74 (Llama-2-7B) to +1.17 (Llama-3-8B). — [ACL 2025.naacl-long.236 Table 1](https://aclanthology.org/2025.naacl-long.236.pdf)
  - Expert–amateur CD hurts smaller experts on commonsense tasks, and a 7B amateur harms a 65B expert. — [arXiv 2309.09117 Table 3, Fig. 4](https://arxiv.org/abs/2309.09117)
- DeCoRe's DeCoRe_entropy-lite (smaller masked model) underperforms the base Llama3-70B with CoT on MuSiQue (73.36 vs 74.43). — [arXiv 2410.18860 Table 3](https://arxiv.org/abs/2410.18860)
- Model-family instability: CAD on FLAN-T5-3B NQ is −7.09 EM vs regular decoding, but +9.41 on OPT-6.7B (COIECD Table 1). — [arXiv 2402.11893 Table 1](https://arxiv.org/abs/2402.11893)
- Metric fragility: in NWCAD, Llama-3.1-8B HotpotQA-distractor with-context scores 39.20 under EM but 72.10 under a GPT-4o-mini judge, and the method ranking compresses. — [arXiv 2604.16686 Tables D.1, E.1](https://arxiv.org/abs/2604.16686)
- Attention-sorting vs calibration: neither wins uniformly. Attention sorting is better for Vicuna-7B with 20 documents (62.89 vs 58.11) and worse for Tulu-2-7B (44.57 vs 53.91). — [arXiv 2406.16008 Table 5](https://arxiv.org/abs/2406.16008)
- Attrieval hurts on HotpotQA for Qwen2.5-3B (59→56) while helping on MuSiQue. — [arXiv 2503.09819 Table 1](https://arxiv.org/abs/2503.09819)

### Inferences
- Known failure modes point to three experimental controls for the MuSiQue ≤7B study:
  1. Report CAD-family results at several α values and with an adaptive α (JSD or entropy), since fixed α=1 is known to be pathological on non-conflict QA.
  2. Report EM together with F1 or a semantic judge, because decoding changes answer form.
  3. Always include greedy-CoT+SC at matched compute, since no prior multi-hop paper does. A two-stream method costs about 2× per token, so the fair comparison is against SC with about 2 samples, or the methods should be compared on a cost-normalized curve.
- A meaningful gain probably needs to come from steering *which context* the chain attends to (head- or attention-level, per hop), not from "context vs no context". In MuSiQue's distractor setting, both the gold and the distractor paragraphs are "context".

### Gaps
- No large-scale reproducibility study isolates CAD's harm under irrelevant or distractor context on multi-hop QA. The CAD reproducibility paper on query-focused summarization (arXiv 2312.14335) found CAD reduces hallucination on summarization at extra FLOPs, but I did not extract its numbers.
- I found no survey numbers (e.g., knowledge-conflict surveys 2024–2026) within budget.
