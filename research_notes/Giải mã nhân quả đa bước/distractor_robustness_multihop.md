# Inference-time robustness to distractors in multi-hop QA for small open LLMs (≤8B), 2024–2026

Scope note: the researcher's own setting is MuSiQue-Ans with the supporting paragraphs plus 4 distractors, and a +4-distractor intervention, using Qwen2.5-1.5B/3B. Almost no published paper uses that exact setting. The published settings differ in four ways, so read every number below with these in mind. (a) The number of paragraphs: 2 gold only; 18 or 20 paragraphs; LongBench's roughly 11K-word contexts; or RULER-style padding up to 32K tokens. (b) Gold versus open retrieval. (c) The metric: strict EM, token F1, substring "contains"/accuracy, or Recall@k. (d) The answer extraction and prompt. One paper reports that strict EM versus F1/contains alone shifts MuSiQue scores by about 8–15 points ([Zhang, Meng & Collier, ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)).

## Q1. Which training-free distractor-filtering or evidence-selection methods report gains on MuSiQue/2Wiki/HotpotQA with ≤8B models, and how large are the gains vs greedy CoT and self-consistency?

### Takeaway
Six families of training-free methods report gains with ≤8B models. Ranked by evidence:
- **Attention-guided fact re-injection after a first CoT pass (Attrieval).** CoT→Attrieval: +10 points on MuSiQue for Llama-3.1-8B, but only +2 for Qwen2.5-3B.
- **Attention-based evidence highlighting (SelfElicit).** +5 to +12 EM on HotpotQA for 7–8B models, gold-context setting.
- **Multi-agent or LLM relevance scoring followed by filtering.** +16 EM on MuSiQue for Qwen2.5-7B, but on only 100 queries.
- **Attention-based re-ranking (ICR).** +5–6 points Recall@2 over ColBERTv2 on MuSiQue with Llama-3.1-8B. No downstream QA numbers are reported.
- **Prompt-level "attend to documents i, j" instructions.** +5 to +12 points in low-visibility positions, but only with *oracle* gold indices.
- **CoT-guided filtering plus information extraction (LongRAG).** Gains with GPT-3.5/GLM-4 without fine-tuning; with ≤8B models the authors fine-tuned the components.

Self-consistency is not a reliable baseline to beat in long, distractor-heavy contexts. On LongBench-MuSiQue it gives roughly +1 F1 for Qwen2.5-7B, +0.3 for Qwen2.5-3B and −3.0 for Llama-3.2-3B ([Byerly & Khashabi 2024](https://arxiv.org/abs/2411.01101)).

### Cited Findings

**Attrieval (attention-guided retrieval inside CoT; training-free)**
- **Method.** The model first generates a CoT. The attention weights from the generated CoT tokens (last layers) are then used to retrieve top facts from the long context. Those facts are re-inserted and the answer is regenerated. The authors' core diagnosis is that failures come mostly from missed *implicit or second-hop* facts, not faulty reasoning. They find that "internal attention weights from the generated CoT tokens can effectively ground implicit facts, even when these facts are not explicitly recalled" — [Zhang et al. 2025, arXiv:2503.09819](https://arxiv.org/abs/2503.09819)
- **Setup.** MuSiQue and HotpotQA are built RULER-style (Hsieh et al. 2024): gold paragraphs padded with distractors to 0K–32K tokens. For MuSiQue the context "already exceeds 4K tokens". Few examples are evaluated per length because of compute — [arXiv:2503.09819](https://arxiv.org/abs/2503.09819)
- **MuSiQue results, overall score (CoT → Attrieval / Attrieval-kl), with the shortest-context column in brackets:**

  | Model | CoT | Attrieval | Attrieval-kl |
  |---|---|---|---|
  | Llama-3.1-8B-Instruct | 53 [78] | 63 [87] | 62 |
  | Llama-3.2-3B-Instruct | 36 | 41 | 45 |
  | Qwen2.5-3B-Instruct | 42 [69] | 44 [72] | 42 |

  At 32K tokens, Llama-3.1-8B goes from 29 → 45 and Qwen2.5-3B from 23 → 18 (worse) — [arXiv:2503.09819](https://arxiv.org/abs/2503.09819)
- **HotpotQA results (CoT → Attrieval):** Llama-3.1-8B 70 → 71; Llama-3.2-3B 63 → 63; Qwen2.5-3B 59 → 56 (worse). The authors note that the gains scale with model capacity, and that Qwen models generate longer CoTs, which weakens their token-selection heuristic — [arXiv:2503.09819](https://arxiv.org/abs/2503.09819)
- **Cost and status.** The method "requires two steps of response generation … which approximately doubles the inference costs". The paper is marked as a work-in-progress preprint — [arXiv:2503.09819](https://arxiv.org/abs/2503.09819)

**SelfElicit (attention-based sentence highlighting; training-free, one extra forward pass)**
- **Method.** Sentence evidence scores are computed from mean attention in the last 50% of layers while the model generates the first token. Sentences above a threshold are wrapped in highlight markers, and the model then answers. The paper shows that deeper layers attend much more to gold-evidence sentences *even when the model answers incorrectly* — [Liu et al. 2025, arXiv:2502.08767](https://arxiv.org/abs/2502.08767)
- **HotpotQA EM/F1 (base → CoT → SelfElicit):**

  | Model | Base | CoT | SelfElicit |
  |---|---|---|---|
  | Llama-3.1-8B | 58.9/57.7 | 60.4/58.6 | 68.5/69.5 |
  | Mistral-7B | 70.4/45.6 | 71.4/44.0 | 74.3/61.6 |
  | Qwen2.5-7B | 65.2/65.8 | 70.7/37.9 | 69.1/71.4 |

  The CoT row for Qwen2.5-7B shows F1 collapsing (65.8 → 37.9) because of verbose output, with "171.9% more inference time" — [arXiv:2502.08767](https://arxiv.org/abs/2502.08767)
- **Caveat (apples-to-oranges).** The main HotpotQA table uses the *gold-context* setup. In the separate HotpotQA *distractor* variant, context length grows by 1443%. There, SelfElicit "maintains the advantage over the baseline", but absolute performance "worsens meaningfully". The selected-evidence ratio falls to under 10% of the context. No MuSiQue or 2Wiki results are reported — [arXiv:2502.08767](https://arxiv.org/abs/2502.08767)

**LLM relevance scoring and filtering of documents (multi-agent assessment; training-free)**
- **MuSiQue results.** Qwen2.5-7B-Instruct was tested on 100 MuSiQue queries with 20 supporting/distractor paragraphs each, at temperature 0.6:

  | Method | EM | F1 |
  |---|---|---|
  | No-Filter | 14.0 | 21.3 |
  | 3-Agent scoring baseline | 30.0 | 39.6 |
  | Percentile-rank normalized scoring (SDA) | 30.0 | 38.7 |
  | CoT De-Polarization | 21.0 | 30.9 |
  | Per-Document Extraction | 24.0 | 34.4 |
  | Random extraction | 12.0 | 19.6 |

  The 95% CI for the +16 pp gain is [+5, +27] pp. The authors note that isolated per-document extraction "cannot perform cross-document synthesis". Cost multipliers are about 4× — [arXiv:2606.25191 "To Isolate or to Score?"](https://arxiv.org/abs/2606.25191)
- **Gains depend on the model–task pair.** "Weak-baseline models (EM_NF < 30%)" benefit, while models with EM_NF ≥ 60% "show no benefit" (e.g., Qwen3 on CONFLICTS drops from 60.8% to 58.6%) — [arXiv:2606.25191](https://arxiv.org/abs/2606.25191)

**ICR — In-context re-ranking from attention (training-free; 2 forward passes)**
- **Re-ranking results.** ICR re-ranks the top-20 ColBERTv2 results on 1000 sampled queries per dataset, using Mistral-7B-Instruct-v0.2 and Llama-3.1-8B-Instruct. Recall@2 / Recall@5:

  | Re-ranker | MuSiQue | 2Wiki | HotpotQA |
  |---|---|---|---|
  | ColBERTv2 (no re-ranking) | 37.9/49.2 | 59.2/68.2 | 64.7/79.3 |
  | RankGPT (Llama-3.1-8B) | 39.3/51.4 | 60.8/72.0 | 69.9/84.8 |
  | ICR (Llama-3.1-8B) | 44.3/55.9 | 67.0/72.6 | 77.3/86.2 |
  | ICR (Mistral-7B) | 39.5/52.6 | 62.7/71.3 | 71.2/84.3 |

  All-Recall@5 (all gold paragraphs in the top 5) on MuSiQue: 16.1 (ColBERTv2) → 23.5 (ICR, Llama-3.1-8B) — [Chen, Gutiérrez & Su, ICLR 2025, arXiv:2410.02642](https://arxiv.org/abs/2410.02642)
- **Caveats.** ICR uses open retrieval over a candidate pool, and reports only re-ranking, not downstream EM. Plain last-token attention sorting (Peysakhovich & Lerer 2023) is much weaker: MuSiQue R@5 is 41.1 versus 55.9 for full ICR. The two components that make the difference are aggregation over all query tokens and calibration with a content-free query — [arXiv:2410.02642](https://arxiv.org/abs/2410.02642)

**Attention-steering instructions (MFAI) — probe of the "recognition bottleneck"**
- **Setting.** 18 documents in 3 positional buckets, MuSiQue and NeoQA, with Qwen2.5-7B, Qwen2.5-14B, Llama-3.1-8B, Ministral-8B and Qwen3-8B.
- **Result.** "Matched MFAI" — instructions naming the gold document indices — raises Tail-bucket accuracy by 4.83–11.49% on MuSiQue. McNemar gains are +10.7 to +11.9 for Llama-3.1-8B and +10.2 to +12.2 for Qwen2.5-7B. Misleading indices *hurt* on MuSiQue.
- **Weakest Link Effect.** "Performance collapses to the level of the least visible evidence, governed by absolute position rather than the linear distance between facts."
- **Caveat.** Matched MFAI uses oracle indices, so it is an upper-bound probe, not a deployable method. Its relevance: once the evidence is *recognized*, small models largely can synthesize — [Zhang, Meng & Collier, ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)
- **Partially correct cues still help.** One correct gold index beats unguided; random indices hurt — [ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)

**Context reduction / "lost in the middle" in multi-hop**
- **Position and spread both matter.** Multi-hop accuracy degrades with evidence distance from the edges *and* with separation between evidence documents (20-document contexts). Models tested: MPT-7B-8k-instruct, Llama-2-7B-LongLoRA and GPT-3.5.
- **CoT.** CoT "aids in identifying relevant documents but fails to resolve the performance disparity caused by evidence document positions".
- **Summarization / KG-triple reduction.** These flatten the positional curve but lower overall accuracy for instruction-tuned models, because information is lost — [Baker et al. 2024, arXiv:2412.10079](https://arxiv.org/abs/2412.10079)

**LongRAG (CoT-guided chunk filter + global information extractor)**
- **Setting.** LongBench HotpotQA/2Wiki/MuSiQue, standardized for RAG, F1.
- **Without SFT.** GPT-3.5-Turbo vanilla RAG → LongRAG: MuSiQue 25.22 → 32.83, HotpotQA 52.31 → 56.17, 2Wiki 43.44 → 51.37.
- **≤8B models.** Results were reported *with SFT* of the components (e.g., Llama3-8B-8k MuSiQue 19.66 → 31.70), so they are not training-free — [Zhao et al., EMNLP 2024, arXiv:2410.18050](https://arxiv.org/abs/2410.18050)

**Self-consistency in long, distractor-heavy contexts**
- **Headline.** Self-consistency "not only fails to improve but actively degrades performance on long-context tasks". The degradation is driven by position bias and "worsens with longer context lengths and smaller model sizes".
- **LongBench-MuSiQue (200 items, ~11K words, F1), Universal SC vs single sample:**

  | Model | Single sample → USC | Change |
  |---|---|---|
  | Llama-3.1-8B | 20.9 → 25.5 | +4.6 (significant) |
  | Llama-3.2-3B | 20.9 → 17.9 | −3.0 |
  | Qwen2.5-7B | 28.7 → 29.6 | +0.9 (n.s.) |
  | Qwen2.5-3B | 12.5 → 12.8 | +0.3 (n.s.) |

  Source: [Byerly & Khashabi 2024, arXiv:2411.01101](https://arxiv.org/abs/2411.01101)

**Iterative retrieve-and-reason (IRCoT / Search-o1) with Qwen2.5 — open retrieval, not the distractor setting**
- **Setting.** E5 retriever over 2018 Wikipedia, 3 passages, EM.
- **Qwen2.5-7B-Instruct:**

  | Method | HotpotQA | 2Wiki | MuSiQue |
  |---|---|---|---|
  | Direct | 0.183 | 0.250 | 0.031 |
  | CoT | 0.092 | 0.111 | 0.022 |
  | IRCoT | 0.133 | 0.149 | 0.072 |
  | Search-o1 | 0.187 | 0.176 | 0.058 |
  | RAG | 0.299 | 0.235 | 0.058 |

- **Qwen2.5-3B-Instruct:**

  | Method | HotpotQA | 2Wiki | MuSiQue |
  |---|---|---|---|
  | CoT | 0.021 | 0.021 | 0.002 |
  | IRCoT | 0.164 | 0.171 | 0.067 |
  | RAG | 0.255 | 0.226 | 0.047 |

- In this setup, prompting-only IRCoT for small Qwen2.5 does *not* beat one-shot RAG on HotpotQA or 2Wiki, and gives tiny absolute gains on MuSiQue. RL-trained Search-R1-instruct reaches 0.370/0.414/0.146 (7B) — [Jin et al. 2025, Search-R1, arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- **Original IRCoT.** The 2022 paper reports gains of up to 21 retrieval points and 15 QA points, including with Flan-T5 models, in open retrieval — [Trivedi et al., arXiv:2212.10509](https://arxiv.org/abs/2212.10509). These results predate 2024 and are not the distractor setting.

### Inferences
- In a fixed-paragraph distractor setting like the researcher's, IRCoT/ReAct-style *retrieval* loops have little to act on, since there is no corpus to query. The closest analogue is "select-then-reason" (filter the paragraph set, then run CoT) or "reason-attend-reinject" (Attrieval). The ICR/SelfElicit scorers can serve directly as the filter.
- Two results support the researcher's mediation finding (that harm flows through the regenerated chain): Attrieval's diagnosis that second-hop facts are attended but not verbalized, and the MFAI result that small models fail at recognition rather than synthesis. Both predict that filtering *before* chain generation should recover much of the 3–6 point loss. The best-case ceiling is roughly the gold-only score.
- For Qwen2.5-3B the evidence is weakest: Attrieval gives +2 on MuSiQue and −3 on HotpotQA. Gains for ≤3B Qwen models should be treated as uncertain until replicated.

### Gaps
- No paper found evaluates Attrieval, SelfElicit or ICR on Qwen2.5-1.5B, or on MuSiQue with exactly gold + 4 distractors.
- No training-free paper found reports downstream MuSiQue EM after ICR-style re-ranking or attention-based paragraph pruning.
- No head-to-head comparison found of filter-then-CoT versus self-consistency on the same distractor-setting MuSiQue split for ≤8B models.
- System 2 Attention (Weston & Sukhbaatar 2023) — regenerate the context, then answer — was not verified for ≤8B multi-hop in this pass.

## Q2. Is "self-filter then reason" (the model selects relevant paragraphs first) effective for small models, or do small models fail at selection?

### Takeaway
The evidence is mixed. Generative self-selection by 7–8B models helps on HotpotQA (gold context) and on MuSiQue (+7 to +16 EM for Qwen2.5-7B with scoring-based filtering). However:
- it costs about 4–7× more inference;
- small models lose cross-document dependencies when each paragraph is judged in isolation;
- a cheap attention-based selector usually matches or beats generative selection.

### Cited Findings
- **PromptElicit (generative self-selection) on HotpotQA EM/F1 (gold context), compared with SelfElicit (attention-based):**

  | Model | PromptElicit | SelfElicit | Base |
  |---|---|---|---|
  | Llama-3.1-8B | 66.3/66.3 (≈7.5× time: 1672 ms vs 224 ms) | 68.5/69.5 | 58.9/57.7 |
  | Mistral-7B | 71.0/64.1 | 74.3/61.6 | — |
  | Qwen2.5-7B | 64.7/67.7 | 69.1/71.4 | 65.2/65.8 |

  For Qwen2.5-7B, generative selection is *below* base EM. Across 48 model-task-metric pairs, SelfElicit beats PromptElicit in 40, at a fraction of the cost — [arXiv:2502.08767](https://arxiv.org/abs/2502.08767)
- **Scoring by the model itself works on MuSiQue for Qwen2.5-7B.** 3-agent scoring gives +16 EM (14 → 30). Per-document extraction, where each paragraph is processed in isolation, gives only +10. Random extraction gives −2. The paper predicts that isolation-based extraction structurally fails for cross-document chains — [arXiv:2606.25191](https://arxiv.org/abs/2606.25191)
- **Selection quality is the bottleneck.** On MuSiQue (18 docs), given oracle "focus" indices, small models gain about 5–12 points in low-visibility positions, while wrong indices hurt. Llama-3.1-8B's response length triples under misleading instructions (~331 → ~1,052 characters) — [ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)
- **Noise filtering is the main bottleneck for instruction-following models.** Gold-only versus 18-document EM for Llama-3.1-8B on MuSiQue: 40.45% versus an average of 23.28%. Qwen3-8B in thinking mode matched its gold-only score even with noise — [ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)
- **RankGPT-style generative ranking breaks for weaker instruction followers.** Mistral-7B gives correctly formatted rankings only 20.5% of the time, and its MuSiQue R@2 falls *below* ColBERTv2 (36.2 vs 37.9). Attention-based ICR with the same model reaches 100% success and 39.5 R@2 — [arXiv:2410.02642](https://arxiv.org/abs/2410.02642)
- **LongRAG's CoT-guided filter used fine-tuned components for ≤8B models.** Its training-free results are only for GPT-3.5/GLM-4 — [arXiv:2410.18050](https://arxiv.org/abs/2410.18050)

### Inferences
- For 1.5B–3B models, generative paragraph selection is likely fragile. The evidence: format failures at 7B, verbose outputs from Qwen, and second-hop paragraphs that look irrelevant until the bridge entity is known. A safer design is either attention-based scoring (ICR/SelfElicit-style) or *iterative* selection: select hop 1, then select hop 2 conditioned on the bridge entity.
- Soft filtering — highlighting or reordering rather than deleting — is lower risk than hard pruning when recall of the selector is uncertain. SelfElicit's highlighting does not drop any context.

### Gaps
- No direct study found of paragraph self-selection accuracy (precision/recall versus MuSiQue `is_supporting` labels) for Qwen2.5-1.5B/3B.
- No study found comparing hard pruning with reordering (gold-likely paragraphs placed at the edges) for ≤3B models on MuSiQue.

## Q3. Attention-based or retrieval-head-based evidence selection — results?

### Takeaway
This is the most promising family for small models. Aggregated attention from query or CoT tokens to each paragraph is a strong zero-shot relevance signal at 3–8B: ICR beats generative RankGPT, and SelfElicit and Attrieval give downstream gains. It works best with calibration (subtract a content-free-query baseline), aggregation over all query tokens, and deep layers or selected heads. Raw attention from a tiny 0.5B proxy without a trained probe is weaker than keeping the full context.

### Cited Findings
- **ICR ablation on Llama-3.1-8B, Recall@5:**

  | Variant | MuSiQue | 2Wiki |
  |---|---|---|
  | Full ICR | 55.9 | 72.6 |
  | − calibration | 52.1 | 64.4 |
  | − aggregation | 41.9 | 52.6 |
  | − both (≈ attention sorting) | 41.1 | 50.3 |

  The paper also reports strong position bias toward the beginning and end of the context, and bias toward titles, entities and punctuation. On 2Wiki, entity-overlap distractors hurt ICR — [arXiv:2410.02642](https://arxiv.org/abs/2410.02642)
- **Why ICR helps on multi-hop.** In token-level analysis, the bridge entity (e.g., "Frederick Douglass") lights up the second-hop document, attributed to LLM "knowledge consolidation" — [arXiv:2410.02642](https://arxiv.org/abs/2410.02642)
- **SelfElicit.** Default: last 50% of layers. Deeper layers separate evidence from non-evidence sentences across the Llama, Mistral and Qwen families on HotpotQA — [arXiv:2502.08767](https://arxiv.org/abs/2502.08767)
- **Attrieval.** Uses last-layer attention from generated CoT tokens. Second-hop statements receive high attention ranks in early CoT tokens even when they are never verbalized. Attention sinks (prompt start, recent tokens) must be filtered out — [arXiv:2503.09819](https://arxiv.org/abs/2503.09819)
- **QRHead (query-focused retrieval heads).** QRRetriever uses a small set of detected heads (e.g., 16 for Llama-3.1-8B) to score passages. It improves performance by over 10% versus full-context generation with Llama-3.1-8B on LongMemEval and CLIPPER. It is also tested as a BEIR re-ranker and against ICR. Models covered: Llama-3.2-3B, Llama-3.1-8B/70B and Qwen2.5-7B. It was *not* evaluated on MuSiQue/HotpotQA QA in the main tables — [Wu et al. 2025, arXiv:2506.09944](https://arxiv.org/abs/2506.09944)
- **Sentinel (attention from a 0.5B proxy plus a small trained logistic-regression probe; not strictly training-free).**
  - With 5× compression, LongBench Multi-Doc QA reaches 47.4 (Qwen2.5-0.5B proxy) versus 38.7 for the original prompt.
  - Under a 2K constraint with Qwen2.5-7B-Instruct as the reader, Multi-Doc QA scores are:

    | Method | Multi-Doc QA |
    |---|---|
    | Raw attention (Qwen2.5-0.5B) | 38.96 |
    | Sentinel | 46.16 |
    | Original prompt | 44.74 |
    | Random | 30.68 |

  - Raw attention from a tiny proxy loses to the full context; a lightweight probe recovers the gap — [arXiv:2505.23277](https://arxiv.org/abs/2505.23277)
- **Retrieval-head-style analysis on LongBench HotpotQA/2Wiki/MuSiQue.** MuDAF identifies retrieval heads of Llama-3.1-8B and then *trains* them contrastively. Its baselines show Llama-3.1-8B-Instruct-128k at HotpotQA 54.7, 2Wiki 44.0 and MuSiQue 32.8 F1 (LongBench). MuDAF itself (trained) reaches 69.6/66.2/48.2 — [arXiv:2502.13963](https://arxiv.org/abs/2502.13963). This shows head-level focusing has large headroom, but that headroom needs training.
- **Attention shift under correct instructions.** Under matched MFAI, attention mass shifts toward gold documents, "concentrated in deep layers", for Qwen2.5-7B — [ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)

### Inferences
- A concrete low-cost recipe for the researcher's setting:
  1. Run one prefill over the question and all paragraphs with eager attention.
  2. Score each paragraph by query-token→paragraph attention mass, summed over the last half of the layers, minus a content-free-query baseline (ICR calibration).
  3. Keep the top-k, or move the top-k to the end next to the question.
  4. Run greedy CoT.

  Optionally, add an Attrieval-style second pass that uses CoT-token attention to recover second-hop paragraphs.
- Qwen2.5-1.5B/3B have fewer layers and heads, so the attention signal may be noisier. The Attrieval Qwen2.5-3B results (small or negative gains) hint at this. Calibrating the layer range on a small dev split is advisable.

### Gaps
- No published ICR, QRHead or SelfElicit numbers found for Qwen2.5-1.5B/3B on MuSiQue paragraph selection.
- No published downstream EM for "attention-filter → CoT" on MuSiQue-Ans in the distractor setting.

## Q4. Reported EM/F1 of Qwen2.5-3B/7B and Llama-3-8B on MuSiQue (distractor setting) and related benchmarks, to calibrate expectations

### Takeaway
No standard, widely reported "MuSiQue-Ans distractor (20 paragraphs), greedy CoT" EM exists for Qwen2.5-3B/7B. Published numbers vary wildly with prompt, metric and context size. Realistic anchors:
- **18–20 paragraphs, strict EM:** roughly 14–25% for 7–8B instruct models.
- **Gold-only:** about 20% (Qwen2.5-7B, strict EM, verbose) to about 40% (Llama-3.1-8B).
- **LongBench-MuSiQue F1:** roughly 13 (Qwen2.5-3B) to 29–33 (Qwen2.5-7B / Llama-3.1-8B).
- **Open retrieval EM:** only 2–15%.

The researcher's ~25% (1.5B) / ~39% (3B) CoT EM with gold + 4 distractors is plausible — on the high side, given the small context.

### Cited Findings
- **MuSiQue, Qwen2.5-7B-Instruct, 20 paragraphs, 100 queries:** No-Filter EM 14.0 / F1 21.3 — [arXiv:2606.25191](https://arxiv.org/abs/2606.25191)
- **MuSiQue strict EM, gold-only (2 docs; 2-hop / 3-hop / 4-hop):**

  | Model | 2-hop | 3-hop | 4-hop |
  |---|---|---|---|
  | Llama-3.1-8B-Instruct | 40.45 | 35.54 | 32.84 |
  | Ministral-8B | 37.40 | 25.76 | 34.81 |
  | Qwen2.5-7B-Instruct | 19.58 | 15.19 | 17.78 |
  | Qwen2.5-14B | 41.57 | — | — |
  | Qwen3-8B | 42.46 | — | — |

  With no documents, EM is about 0. For Llama-3.1-8B with 18 documents, the average is 23.28% versus 40.45% gold-only. The authors note that F1 or contains would add about 8–15 points — [ACL 2026](https://aclanthology.org/2026.acl-long.1937.pdf)
- **MuSiQue, RULER-style padded contexts, accuracy-type metric, CoT:**

  | Model | Shortest context | 32K tokens |
  |---|---|---|
  | Qwen2.5-3B-Instruct | 69 | 23 |
  | Llama-3.1-8B-Instruct | 78 | 29 |
  | Llama-3.2-3B-Instruct | 78 | 18 |

  These scores are not comparable to strict EM — [arXiv:2503.09819](https://arxiv.org/abs/2503.09819)
- **HotpotQA, RULER-style, CoT, shortest → 32K:** Qwen2.5-3B 64 → 54; Llama-3.1-8B 71 → 68 — [arXiv:2503.09819](https://arxiv.org/abs/2503.09819)
- **LongBench-MuSiQue, single-sample F1** (200 items, ~11K words): Llama-3.1-8B 20.9; Llama-3.2-3B 20.9; Qwen2.5-7B 28.7; Qwen2.5-3B 12.5; Qwen2.5-72B 50.5 — [arXiv:2411.01101](https://arxiv.org/abs/2411.01101)
- **The same benchmark, scored in a different paper:** Llama-3.1-8B-Instruct-128k gets HotpotQA 54.7, 2Wiki 44.0, MuSiQue 32.8 F1 — [arXiv:2502.13963](https://arxiv.org/abs/2502.13963). This conflicts with the 20.9 above for the same model and benchmark, which shows how much prompts and truncation matter.
- **LongBench standardized for RAG (vanilla RAG F1):** Llama3-8B-8k gets HotpotQA 48.25, 2Wiki 43.47, MuSiQue 19.66; Qwen-1.5-7B gets 45.70/34.69/25.08 — [arXiv:2410.18050](https://arxiv.org/abs/2410.18050)
- **HotpotQA, gold context, EM/F1:** Llama-3.1-8B 58.9/57.7; Qwen2.5-7B 65.2/65.8; Mistral-7B 70.4/45.6 — [arXiv:2502.08767](https://arxiv.org/abs/2502.08767)
- **Open retrieval (E5, top-3, EM), Qwen2.5-7B-Instruct, HotpotQA / 2Wiki / MuSiQue:**

  | Method | HotpotQA | 2Wiki | MuSiQue |
  |---|---|---|---|
  | RAG | 0.299 | 0.235 | 0.058 |
  | IRCoT | 0.133 | 0.149 | 0.072 |
  | Search-R1 (RL-trained) | 0.433 | 0.382 | 0.196 |

  The best MuSiQue entry is Search-R1-base. For Qwen2.5-3B, RAG gets 0.255/0.226/0.047 — [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)

### Inferences
- **What these anchors mean for the researcher.** For Qwen2.5-3B at gold + 4 distractors, ~39% CoT EM implies the gold-only ceiling is probably mid-40s. The 3–6 point distractor loss is therefore the realistic recovery target. A good filter could plausibly recover +3 to +6 EM in the +4-distractor condition, and perhaps +1 to +3 in the base condition. Claims above gold-only performance would be suspicious.
- **Qwen2.5 strict-EM results are depressed by verbosity.** Qwen2.5-7B scores only 19.6% gold-only EM, versus Llama-3.1-8B's 40.5%. The researcher should report F1 alongside EM and use a tight answer-extraction prompt, or gains may be extraction artifacts.
- **Self-consistency as a target.** On LongBench-MuSiQue, small Qwen models gain about +0 to +1 F1 from USC. A filtering method that gives +3 or more EM on the researcher's split would already be competitive with or better than self-consistency, at lower cost.

### Gaps
- No leaderboard of prompting-only Qwen2.5-3B/7B on the official MuSiQue-Ans distractor dev set (20 paragraphs) was found. The MuSiQue leaderboard mostly contains fine-tuned models.
- No 2WikiMultiHopQA distractor-setting (10 paragraphs) prompting numbers for Qwen2.5-3B were found in this pass.

## Q5. Cost on 2×T4 (latency, extra forward passes)

### Takeaway
Attention-based scoring costs about one extra prefill: ICR needs 2 forward passes for N documents; SelfElicit adds about 18% time at 8B. Attrieval doubles generation cost. Generative self-selection costs 5–8×, and multi-agent scoring about 4×. On T4s the main constraint is that attention-weight extraction requires eager attention: no FlashAttention, and attention maps are materialized.

### Cited Findings
- **ICR.** Two forward passes (O(1)), versus at least O(N) for generative re-ranking. Over 60% lower latency than RankGPT, and more than 2× faster even though ICR ran in HF Transformers while RankGPT ran in vLLM. Attention weights are not exposed in vLLM — [arXiv:2410.02642](https://arxiv.org/abs/2410.02642)
- **SelfElicit vs PromptElicit vs CoT, per-sample inference time:**

  | Model | Base | SelfElicit | PromptElicit |
  |---|---|---|---|
  | Llama-3.1-8B | 224.1 ms | 264.1 ms | 1672.0 ms |
  | Qwen2.5-7B | 245.2 ms | 289.4 ms | 1165.1 ms |

  For Qwen2.5-7B, CoT takes 421.5 ms. Across models, PromptElicit raises average inference time by 878% — [arXiv:2502.08767](https://arxiv.org/abs/2502.08767)
- **Attrieval.** "Approximately doubles the inference costs", since it runs two generation rounds — [arXiv:2503.09819](https://arxiv.org/abs/2503.09819)
- **Multi-agent scoring.** About 4× cost multiplier versus No-Filter — [arXiv:2606.25191](https://arxiv.org/abs/2606.25191)
- **Self-consistency.** Costs k× generation, with little or negative benefit in long contexts for small models — [arXiv:2411.01101](https://arxiv.org/abs/2411.01101)

### Inferences (not directly sourced; engineering reasoning)
- T4 (Turing, 16 GB, fp16 only — no bf16, no FlashAttention-2). Qwen2.5-1.5B/3B in fp16 (~3/6 GB) fit on one T4. For gold + 4–8 MuSiQue paragraphs (~1.5–3K tokens), materializing attention for a single prefill is feasible if you keep only the needed layers. Practical trick: compute query→paragraph attention mass per layer with hooks, then discard the maps, instead of using `output_attentions=True` for all layers at once.
- On a T4, the ICR-style scorer adds roughly one prefill plus one content-free-query prefill per example. That is likely well under the cost of CoT decoding (hundreds of tokens), so the overhead is plausibly about 10–30% over greedy CoT. The two T4s can run the conditions in parallel, one model per GPU.

### Gaps
- No source reports T4-specific latency for any of these methods; the T4 estimates above are inference only.
