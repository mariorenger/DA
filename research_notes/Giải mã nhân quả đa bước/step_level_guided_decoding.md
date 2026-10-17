# Step-level / process-guided decoding for CoT reasoning with small LLMs (training-free or lightweight), 2023–2026

Scope note: every number below was read from the paper's abstract or full text (arXiv HTML/PDF, ACL Anthology) unless marked "snippet-only". "Compute-matched" means the paper itself compares against self-consistency (SC) or best-of-N (BoN) at an equal sample or token budget. Where a paper does not do this, it is flagged **[NO COMPUTE-MATCHED SC]**.

## Q1. Method inventory: mechanism, models, benchmarks, gain over SC at matched compute, cost

### Takeaway
Most step-level methods are evaluated on math (GSM8K/MATH). The ones that report compute-matched wins over SC at ≤8B are CoT-decoding (k=10 vs SC@10), self-evaluation-guided beam search (arithmetic only; it loses on StrategyQA), confidence-weighted SC (CISC, self-certainty, DeepConf) and PRM search for <7B policies on hard math. rStar shows large gains over SC@128 at ≤8B, but at much higher token cost. On StrategyQA its gain over SC is +3 to +8 points.

### Cited Findings

**A. SC and compute-efficient variants (whole-chain, no step intervention)**
- ESC (Early-stopping SC, ICLR 2024) stops sampling once a window of samples agrees. It cuts the average number of samples by 80.1% on GSM8K, 33.8% on MATH, 76.8% on StrategyQA, 78.5% on CommonsenseQA, 84.2% on Coin Flip and 67.4% on Last Letters, "while attaining comparable performances". It is a cost reducer, not an accuracy gain. — [Li et al. 2024, arXiv 2401.10480](https://arxiv.org/abs/2401.10480)
- Adaptive-Consistency (EMNLP 2023) uses a lightweight stopping criterion based on agreement so far. It cuts sample budget "by up to 7.9 times with an average accuracy drop of less than 0.1%" across 17 reasoning and code datasets and 3 LLMs. — [Aggarwal et al., arXiv 2305.11860](https://arxiv.org/abs/2305.11860)
- Difficulty-Adaptive SC (DSC) adds prior difficulty estimates for batches of queries. It "consistently surpasses the strong baseline ASC and ESC in terms of costs by a significant margin, while attaining comparable performances" on 6 arithmetic, commonsense and symbolic benchmarks. — [Wang et al., arXiv 2408.13457](https://arxiv.org/abs/2408.13457)
- RASC (Reasoning-Aware SC, NAACL 2025) scores both the rationale quality and the answer consistency of each sample. It uses criteria-based early stopping plus weighted voting. It reduces sample usage by ~70% "while maintaining accuracy". Models: LLaMA2-7B, Vicuna-13B, GPT-3.5/4. Main evaluation set: 16,725 samples across commonsense, math and symbolic tasks (Date Understanding, CommonsenseQA, etc.). There are no multi-hop QA datasets in the main table. — [Wan et al., arXiv 2408.17017](https://arxiv.org/abs/2408.17017) (Sec. 5, Table 2)
- Path-consistency uses the confidence of early answers to pick a promising prefix, then branches from it (a step-level reuse of prefixes). It reports "up to 40.5%" latency improvement "while maintaining task accuracy" on math, commonsense and symbolic tasks. **[No accuracy gain over SC claimed]** — [Zhu et al., arXiv 2409.01281](https://arxiv.org/abs/2409.01281)
- Universal Self-Consistency (USC) asks the LLM itself to pick the most consistent answer among candidates. It "matches the standard self-consistency performance" on math and makes SC applicable to free-form answers. This matters for open-answer multi-hop QA, where exact-match voting fragments. — [Chen et al., arXiv 2311.17311](https://arxiv.org/abs/2311.17311)
- CISC (Confidence-Informed SC, 2025) runs a weighted majority vote, with weights from model-internal confidence (P(True), verbal 0/1 or 0–100, sequence probability). It "outperforms self-consistency in nearly all configurations, reducing the required number of reasoning paths by over 40% on average". Setup: 9 models (Gemma2 2B/9B/27B; Qwen2.5 3B/14B/72B; others) on GSM8K, MATH, MMLU-Pro plus one more dataset. Key finding: "the most calibrated confidence method proved to be the least effective for CISC". Within-question discrimination is what matters. — [Taubenfeld et al., arXiv 2502.06233](https://arxiv.org/abs/2502.06233) (Secs. 4.1–4.3)
- Self-certainty is a reward-free BoN selector. It scores responses by the distributional confidence of the output tokens and "scales effectively with increasing sample size N, akin to reward models but without the computational overhead". It "generalizes to open-ended tasks where traditional self-consistency methods fall short". — [Kang et al., arXiv 2502.18581](https://arxiv.org/abs/2502.18581)
- DeepConf filters low-confidence traces *during* generation, using local token-confidence windows, or after it. On AIME 2025, DeepConf@512 reaches up to 99.9% accuracy and cuts generated tokens by up to 84.7% versus full parallel thinking. Evaluated on Qwen3 and GPT-OSS reasoning models, not ≤7B non-reasoning instruct models. — [Fu et al., arXiv 2508.15260](https://arxiv.org/abs/2508.15260)
- Soft-SC replaces discrete voting with a continuous likelihood-based score. At a fixed number of samples it gains over SC by +1.3 (bash), +6.6 (WebShop) and +4.7 (ALFWorld) absolute. It needs "half as many samples as SC for comparable or better performance". — [Wang et al. ACL 2024, arXiv 2402.13212](https://arxiv.org/abs/2402.13212)

**B. CoT-decoding (Wang & Zhou, NeurIPS 2024): branch at the first token, select by answer-confidence margin**
- Mechanism: branch over the top-k alternatives for the *first* decoding token, continue each branch greedily, then pick the path with the largest probability margin Δ between the top-1 and top-2 tokens over the answer span ("max path"). Alternatively, sum Δ per answer across paths ("agg path"). — [arXiv 2402.10200](https://arxiv.org/abs/2402.10200)
- Mistral-7B (pretrained) on GSM8K, no prompt (Table 4): greedy 9.9%; top-k sampling (k=10) 4.9%; nucleus (p=0.9) 6.4%; beam search (b=10) 6.7%; temperature (T=0.7) 7.5%; SC without CoT prompt (10 paths) 12.9%; CoT-decoding (k=10) 25.1%. — [arXiv 2402.10200, Table 4](https://arxiv.org/abs/2402.10200)
- **Compute-matched against SC (k=10 for all multi-path methods), GSM8K, Table 7:** Mistral-7B: zero-shot CoT 17.5%; SC with zero-shot CoT 39.4%; CoT-decoding (max) + zero-shot CoT 40.2%; CoT-decoding (agg) + zero-shot CoT **48.4%**. PaLM-2 L: 75.1 / 85.3 / 78.6 / **87.0**. So agg-path CoT-decoding beats SC@10 by +9.0 points on a 7B model, but the max-path variant alone does not (+0.8 for Mistral, −6.7 for PaLM-2 L). — [arXiv 2402.10200, Table 7](https://arxiv.org/abs/2402.10200)
- Benchmarks: GSM8K, MultiArith, year parity, and Big-Bench sports/coin-flip-style tasks. Models: PaLM-2 family, Mistral-7B, Gemma-7B, plus instruction-tuned variants. No multi-hop QA. — [arXiv 2402.10200, Sec. 3](https://arxiv.org/abs/2402.10200)
- The paper notes the "optimal branching point may vary with the task (e.g., year parity)". Branching at token 0 is a heuristic. — [arXiv 2402.10200, Sec. 2](https://arxiv.org/abs/2402.10200)

**C. Self-evaluation guided (stochastic) beam search (Xie et al., NeurIPS 2023)**
- Mechanism: at each step the LLM scores its own step for correctness (an evaluation-confidence prompt). The score is combined with generation probability, and stochastic beam search with temperature keeps a diverse set of beams. — [arXiv 2305.00633](https://arxiv.org/abs/2305.00633)
- Codex backbone: +6.34 (GSM8K), +9.56 (AQuA) and +5.46 (StrategyQA) few-shot accuracy over the corresponding baselines. These are not compute-matched; the paper notes the single-chain PAL version "costs about [several] times more than the self-consistency baseline on GSM8K". — [arXiv 2305.00633, abstract & Sec. 4](https://arxiv.org/abs/2305.00633)
- **Compute-matched (Table 3, Llama-2 13B, # tokens):** "Our method consistently outperforms self-consistency on the arithmetic reasoning tasks even when benchmarked for relatively less computational cost. … However, our approach lags behind the CoT baseline on commonsense reasoning [StrategyQA]". The authors attribute this to short chains: "decreasing the number of intermediate steps weakens the effect of stepwise self-evaluation". — [arXiv 2305.00633, Sec. 4.2/Table 3](https://arxiv.org/abs/2305.00633)
- On StrategyQA, self-evaluation scores "do not align well with whether the prediction is correct". LLMs are less confident on commonsense chains than on GSM8K chains. — [arXiv 2305.00633, Sec. 5 / App. Fig. 8](https://arxiv.org/abs/2305.00633)

**D. MCTS at inference with small models: rStar (ICLR 2025)**
- Mechanism: MCTS with 5 human-like actions (one-step thought, propose remaining steps, sub-question + answer, re-answer sub-question, rephrase). The terminal reward is SC-majority confidence. A second SLM (Phi3-mini) acts as the *discriminator*: it is given a random prefix of each trajectory, completes the rest, and the trajectory is kept only if both SLMs reach the same answer ("mutual consistency"). — [Qi et al., arXiv 2408.06195](https://arxiv.org/abs/2408.06195) (Sec. 3)
- GSM8K: LLaMA2-7B 12.51 → 63.91; Mistral-7B 36.46 → 81.88; LLaMA3-8B-Instruct 74.53 → 91.13. — [arXiv 2408.06195, abstract](https://arxiv.org/abs/2408.06195)
- **StrategyQA (Table 2; columns LLaMA2-7B / Mistral-7B / LLaMA3-8B / LLaMA3-8B-Inst / Phi3-mini):**
  - Few-shot CoT: 58.82 / 65.65 / 64.05 / 68.41 / 63.61
  - SC@maj8: 59.10 / 65.50 / 63.76 / 68.26 / 64.34
  - SC@maj64: 58.51 / 63.61 / 63.46 / 67.39 / 62.74
  - SC@maj128: 58.37 / 62.01 / 63.31 / 66.67 / 59.53
  - ToT: 45.27 / 55.75 / 57.64 / 60.41 / 40.47
  - RAP: 59.68 / 64.48 / 63.32 / 68.71 / 60.26
  - rStar (generator @maj): 61.57 / 69.43 / 65.50 / 71.47 / 65.50
  - rStar (with discriminator): **67.25 / 70.31 / 67.69 / 71.57 / 67.25**
  - Source: [arXiv 2408.06195, Table 2](https://arxiv.org/abs/2408.06195)
- "SC with more sampling can even lower the score on StrategyQA". ToT is *below* few-shot CoT for every SLM on StrategyQA. — [arXiv 2408.06195, Sec. 4.2](https://arxiv.org/abs/2408.06195)
- On GSM8K, rStar's generator alone (majority vote) is 1.69–7.34 points above SC. — [arXiv 2408.06195, Sec. 4.2](https://arxiv.org/abs/2408.06195)
- **Cost (Table 7):** with 32 rollouts on GSM8K, the average per question is 166.81 calls and 367.1k generated tokens (LLaMA2-7B), or 148.90 calls and 348.6k tokens (Mistral). The full GSM8K test set takes "about 4.5 days on a single A100 GPU per model". **[NO TOKEN-MATCHED SC; SC@128 is the largest SC baseline]** — [arXiv 2408.06195, App. Table 7](https://arxiv.org/abs/2408.06195)
- Swapping the discriminator for stronger or weaker LLMs "generally does not affect the effectiveness". — [arXiv 2408.06195, Sec. 4.3 / Table 5](https://arxiv.org/abs/2408.06195)
- rStar-Math (follow-up) is *not* training-free. It trains a policy SLM and a process preference model over 4 self-evolution rounds on 747k problems, taking Qwen2.5-Math-7B from 58.8 to 90.0 on MATH. — [Guan et al., arXiv 2501.04519](https://arxiv.org/abs/2501.04519)

**E. Step-level verifiers / PRMs (lightweight = a trained verifier, frozen policy)**
- Math-Shepherd builds a PRM from Monte Carlo (MC) rollouts, so no human step labels are needed. Mistral-7B with PRM verification reaches 89.1 (GSM8K) and 43.5 (MATH), versus 77.9 and 28.6 for the base policy. — [Wang et al., arXiv 2312.08935](https://arxiv.org/abs/2312.08935)
- PAV (process advantage verifiers) measure step "progress" under a prover policy distinct from the base. Test-time search against PAVs is ">8% more accurate, and 1.5–5× more compute-efficient" than against ORMs. "Weak prover policies can substantially improve a stronger base policy". — [Setlur et al., arXiv 2410.08146](https://arxiv.org/abs/2410.08146)
- GenRM (generative verifiers, next-token "Yes/No" with CoT verification, majority vote over verifications) for BoN: 5% → 45.3% on algorithmic tasks, 73% → 93.4% on GSM8K; easy-to-hard 28% → 44.6% on MATH. — [Zhang et al., arXiv 2408.15240](https://arxiv.org/abs/2408.15240)
- SelfCheck: zero-shot step checking (regenerate each step from extracted context and compare it with the original), then weighted voting. Tested only on GSM8K, MathQA and MATH. — [Miao et al., arXiv 2308.00436](https://arxiv.org/abs/2308.00436)

**F. Multi-chain meta-reasoning (multi-hop QA specific)**
- MCR (Multi-Chain Reasoning, EMNLP 2023): sample several retrieval-augmented decomposition chains, then a meta-reasoner LLM reads all of them (their intermediate steps, not just their answers) and writes one explanation and answer. — [Yoran et al., arXiv 2304.13007](https://arxiv.org/abs/2304.13007)
- code-davinci-002, 5 chains, Table 2 (SC@5 → MCR): HotpotQA 51.3 → 57.0; 2WikiMQA 65.4 → 67.9; StrategyQA 72.2 → 73.6; Bamboogle 65.0 → 66.5; Fermi 38.3 → 38.9; Quartz 77.6 → 81.6; Feverous 67.9 → 69.4. — [arXiv 2304.13007, Table 2](https://arxiv.org/abs/2304.13007)
- With 15 chains (Table 3: SC@15 / MCR / MCR+SC@3): HotpotQA 54.1 / 57.0 / 59.2; 2WikiMQA 65.8 / 67.9 / 68.6; StrategyQA 74.6 / 73.6 / 76.4. — [arXiv 2304.13007, Table 3](https://arxiv.org/abs/2304.13007)
- With open-source Vicuna-13B, MCR "still outperforms SC@5 on all 7" datasets (e.g., StrategyQA +0.5). — [arXiv 2304.13007, Sec. 5](https://arxiv.org/abs/2304.13007)

### Inferences
- The only clean compute-matched win over SC at 7B in this inventory is CoT-decoding (agg path) plus a CoT prompt: +9.0 points on GSM8K for Mistral-7B at k=10. It is not tested on multi-hop QA. Branching at the *first* token is a cheap way to diversify chains (k greedy continuations), and it fits a 2×T4 budget.
- Self-evaluation guided beam search loses to plain CoT on StrategyQA under matched compute. Its benefit grows with chain length. Multi-hop QA chains of 2–4 hops look more like StrategyQA than GSM8K, so it is a weak candidate unless the chain is decomposed into many steps.
- rStar's gain on StrategyQA (+3 to +8 over SC@8) comes at roughly an order of magnitude more tokens than SC@128. That is our estimate: ~367k tokens per question, versus 128 chains × a few hundred tokens. It is likely impractical on 2×T4 at scale.

### Gaps
- No found paper reports CoT-decoding, ESC/ASC/CISC or self-eval beam search on MuSiQue, HotpotQA, 2Wiki or CLUTRR with ≤7B models.
- The exact Table-3 token counts and accuracies from Xie et al. were in MathML that did not extract. The qualitative statements are verified; the numbers are not recorded here.

## Q2. Which methods have results on multi-hop QA or relational reasoning (CLUTRR) with ≤7B models?

### Takeaway
Multi-hop QA results at ≤8B exist almost only for *retrieval-augmented, partly trained* step-level search: RAG-Star (Llama-3.1-8B, trained RM) and SiGIR (7B, trained). None compares with compute-matched SC. In SiGIR, removing the step-level guided search costs only ~2 F1, so most of the gain comes from training. For CLUTRR, the 2026 step-level methods are neurosymbolic (ILP or symbolic step verifiers), not pure decoding.

### Cited Findings
- **RAG-Star:** MCTS over (sub-query, sub-answer) steps. Each node gets a retrieval-augmented *answer-aware reward*: consistency between the LLM's internal sub-answer and the retrieved documents, with refinement on conflict. It also gets a query-aware reward for logical consistency of the plan. Policy is Llama-3.1-8B-Instruct; the reward model is either GPT-4o-mini or a *fine-tuned* Llama-3.1-8B. — [Jiang et al., arXiv 2412.12881](https://arxiv.org/abs/2412.12881)
- RAG-Star, Table 1, Llama-3.1-8B-Instruct (EM/CoverEM/F1), **on only 100 random validation samples per dataset**:
  - HotpotQA: direct 14/25/26.0; CoT 20/38/26.3; standard RAG 40/48/52.8; RAG-Star with Llama RM 42/44/54.4.
  - 2Wiki: direct 9/29/21.9; CoT 4/32/7.1; standard RAG 17/23/26.1; RAG-Star with Llama RM 34/38/42.0.
  - MuSiQue: direct 2/3/3.9; CoT 4/16/6.6; standard RAG 11/11/15.5; RAG-Star with Llama RM 13/18/22.2.
  - StrategyQA: direct 63/65/63; CoT 55/69/55; RAG-Star with Llama RM 71/72/71.
  - With the GPT-4o-mini RM: HotpotQA 46/49/60.0; 2Wiki EM 38, CEM 43.
  - **[NO SC/BoN BASELINE; RM is trained; n=100]** — [arXiv 2412.12881, Table 1](https://arxiv.org/abs/2412.12881)
- RAG-Star ablation: removing retrieval from reward modeling makes "performance drop significantly for Llama-3.1-8B". Accuracy rises with more simulation rounds, as does time. — [arXiv 2412.12881, Sec. 5.3–5.4](https://arxiv.org/abs/2412.12881)
- **SiGIR (ACL Findings 2025):** iterative decompose → retrieve → reason, with self-critique rewards per step and beam-style top-k selection by cumulative reward. The reasoner and critic are fine-tuned on teacher-annotated trajectories plus self-improvement data. **Not training-free.** — [Chu et al., ACL Findings 2025](https://aclanthology.org/2025.findings-acl.123.pdf)
- SiGIR Table 1 (overall scores; 2WikiMQA / HotpotQA / MuSiQue):
  - CoT: 31.45 / 35.31 / 18.93
  - IRCoT: 46.80 / 49.03 / 21.43
  - ProbTree: 59.90 / 50.40 / 27.00
  - BeamAggR: 63.20 / 55.20 / 32.30
  - RAG-Star: 46.80 / 60.00 / 30.70
  - SiGIR main: 74.47 / 63.09 / 37.15
  - Qwen2.5-7B variant: 70.51 / 60.93 / 33.29
  - LLaMA2-7B variant: 65.82 / 52.92 / 28.56
  - The paper claims "+8.6% average" over the previous SOTA.
  - Source: [ACL Findings 2025, Table 1](https://aclanthology.org/2025.findings-acl.123.pdf)
- **SiGIR ablation (Table 2), the key number for "is step-level search worth it":** full 74.47 / 63.09 / 37.15; without guided search 72.37 / 60.86 / 35.47 (≈ −2 points); without reward 67.23 / 54.94 / 29.45; without self-critique 66.96 / 56.86 / 28.62. — [ACL Findings 2025, Table 2](https://aclanthology.org/2025.findings-acl.123.pdf)
- ProbTree (EMNLP 2023 Findings) builds a query tree. For each leaf sub-question it picks the *more confident* answer between closed-book QA (parametric) and open-book QA (retrieved). It aggregates bottom-up with confidence, which "eliminat[es] the negative retrieval problem". Evaluated on HotpotQA, MuSiQue and 2Wiki (open-domain). — [Cao et al., arXiv 2311.13982](https://arxiv.org/abs/2311.13982)
- MCR (Q1-F) is the multi-hop method with explicit SC@k baselines: +5.7 on HotpotQA and +2.5 on 2Wiki over SC@5 with code-davinci-002. With Vicuna-13B it still beats SC@5 on all 7 datasets. — [arXiv 2304.13007](https://arxiv.org/abs/2304.13007)
- rStar and Xie et al. are evaluated on StrategyQA, the nearest "implicit multi-hop" dataset. Numbers are in Q1-C and Q1-D. — [arXiv 2408.06195](https://arxiv.org/abs/2408.06195); [arXiv 2305.00633](https://arxiv.org/abs/2305.00633)
- Long-context RAG inference scaling (DRAG/IterDRAG; ICLR 2025, Gemini models): with optimal allocation, gains are "nearly linear" in inference compute, with up to 58.9% gains over standard RAG on multi-hop benchmarks. Not ≤7B. — [Yue et al., arXiv 2410.04343](https://arxiv.org/abs/2410.04343)
- Atom of Thoughts (Markov decomposition into self-contained atomic sub-questions, integrable with tree search) claims to beat baselines "as computational budgets increase". The evaluated LLMs (e.g., gpt-4o-mini) are not ≤7B. — [Teng et al., arXiv 2502.12018](https://arxiv.org/abs/2502.12018)
- **CLUTRR, Reason Popper-ly (2026):** learns relation-composition rules via ILP from traces and uses them as an *online step-level verifier*. It diagnoses each violated step, rewrites it symbolically, and regenerates the remaining suffix so the final answer is conditioned on a verified trace. Gains over CoT are "up to 48 percentage points for small models and 15 points for frontier models on the longest chains" (2–10 hops, 5 LMs). — [arXiv 2607.23019](https://arxiv.org/abs/2607.23019)
- **CLUTRR, VeryTrace (2026):** formalizes traces into a compilable DSL, with deterministic checks plus targeted LLM audits for step-level localization and repair. It improves over zero-shot baselines on CLUTRR, AIME 2025 and LLM-BabyBench with state-of-the-art LLMs. Snippet-level detail: a 150-instance CLUTRR ablation scored 69.33 two-stage versus 70.67 direct (not verified in full text). — [arXiv 2606.24124](https://arxiv.org/abs/2606.24124)
- **CLUTRR with SLMs (Gemma 3 1B/4B, Llama 3.2 3B), KG grounding (2026):** RGCN-derived hints give 1.5–2× gains over story-only baselines. The system is limited by an "extraction bottleneck and sequential deductive fragility". The authors report a "distraction effect" where noisy self-generated facts degrade performance even when expert hints are present. — [arXiv 2607.14149](https://arxiv.org/abs/2607.14149)
- Snippet-only (not verified in full text): several 2026 preprints use step-level verification or process rewards on HotpotQA, MuSiQue and 2Wiki. SAFE (LLM-as-verifier per step against passages, KG triples), PRO-Step (PRM-based step preferences, Qwen2.5-7B-Instruct, trained), ARBOR (rubric-based online process rewards for search agents) and StepGap (NLI+LLM evidence-gap checker). — [SAFE 2604.01993](https://arxiv.org/html/2604.01993v2); [PRO-Step 2609.01658](https://arxiv.org/pdf/2609.01658); [ARBOR 2606.03239](https://arxiv.org/pdf/2606.03239); [StepGap 2605.24733](https://arxiv.org/pdf/2605.24733)

### Inferences
- For the researcher's distractor setting with passages in context, the most transferable mechanism is RAG-Star-style *answer-aware step reward*. Concretely: check each written intermediate answer (bridge entity) against the supporting passage before continuing. It needs no retrieval, since the passages are already in context, and it can be done with an NLI or 7B judge rather than a trained RM. That is untested at this configuration.
- SiGIR's ablation suggests that, once a model is trained, step-level search adds only ~2 F1 on multi-hop QA. Training-free search on an *untrained* ≤3B model is unlikely to do better than that, given the self-evaluation failures in Q5.
- On CLUTRR, step-level verification works when the verifier is *symbolic* (composition rules), because kinship composition is checkable. A rule-table verifier (deterministic, cheap) fits 2×T4 easily. This is the strongest "intervene during chain generation" evidence for relational reasoning.

### Gaps
- No found paper reports compute-matched SC versus step-level search on MuSiQue, HotpotQA or 2Wiki with ≤7B models and passages in context (the distractor setting).
- No found paper reports CoT-decoding, CISC, ESC, rStar or PRM search on CLUTRR with Qwen2.5 ≤7B.
- The SiGIR main-row backbone and metric (F1 vs EM) were not confirmed in the extracted text. The paper names Mistral, LLaMA2 and Qwen2.5 (7B).

## Q3. Compute-optimal test-time scaling for small models: when does search/verification beat BoN/SC?

### Takeaway
Search against a PRM beats BoN mainly at *low budgets*, on *harder* problems and for *smaller policies*. At high budgets and on easy problems, beam search over-optimizes and falls below BoN. All of this evidence is on math (MATH, AIME) with math-trained PRMs, and those PRMs generalize poorly across policy models and domains.

### Cited Findings
- Snell et al. (2024; ICLR 2025): "at low generation budgets, beam search performs best, but as we scale the budget further the improvements diminish, falling below the best-of-N baseline. Lookahead-search generally underperforms other methods at the same generation budget." — [arXiv 2408.03314, Fig. 3](https://arxiv.org/abs/2408.03314)
- Snell et al., by difficulty: "On the easier problems (bins 1 and 2), beam search shows signs of over-optimization with higher budgets, whereas best-of-N does not". Revisions suit easy problems; parallel sampling or tree search against a PRM suits harder ones. — [arXiv 2408.03314, Fig. 3 & Sec. 1](https://arxiv.org/abs/2408.03314)
- Snell et al., headline results: compute-optimal allocation gives ">4× efficiency vs best-of-N". FLOPs-matched, a smaller model plus test-time compute beats a 14× larger model on easy and intermediate questions. On "the most challenging questions, we observe very little benefits from scaling up test-time compute". Model: PaLM 2-S*, MATH. — [arXiv 2408.03314](https://arxiv.org/abs/2408.03314)
- Liu et al. (2025) on Qwen2.5 0.5B–72B and Llama 3.x, MATH-500 and AIME24: "For small policy models, search-based methods outperform BoN, while for large policy models, BoN is more effective". The reason given is that "smaller models rely on a verifier to select each step". — [arXiv 2502.06703, Sec. 4.2](https://arxiv.org/abs/2502.06703)
- Liu et al., by difficulty: "for small policy models (<7B), BoN is better for easy problems, while beam search works better for harder problems". DVTS suits easy and medium problems for 7B–32B. BoN is best at every difficulty for 72B. — [arXiv 2502.06703, Sec. 4.3](https://arxiv.org/abs/2502.06703)
- Liu et al., on PRMs: "PRMs are hard to generalize across policy models and tasks". BoN beats search with Math-Shepherd and RLHFlow PRMs. Search wins with Skywork and Qwen2.5-Math PRMs, because OOD policy responses lead to sub-optimal PRM choices. — [arXiv 2502.06703, Sec. 4.1](https://arxiv.org/abs/2502.06703)
- Liu et al., headline: with compute-optimal TTS, a 1B model exceeds a 405B model on MATH-500. A 0.5B model beats GPT-4o and a 3B model beats 405B on MATH-500 and AIME24. — [arXiv 2502.06703](https://arxiv.org/abs/2502.06703)
- Wu et al. (Inference Scaling Laws, ICLR 2025): "smaller models combined with advanced inference algorithms offer Pareto-optimal trade-offs". Llemma-7B with the REBASE tree search "consistently outperforms the Llemma-34B model across all tested inference strategies on MATH". — [arXiv 2408.00724](https://arxiv.org/abs/2408.00724)
- Brown et al. (Large Language Monkeys): coverage scales log-linearly with samples. However, "common methods for picking from a sample collection (majority voting and reward models) plateau beyond several hundred samples". The selector, not the generator, is the bottleneck. — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
- Setlur et al. (2025): verifier-based scaling (RL or search) is provably and empirically superior to verifier-free approaches when the base LLM's distribution over correct traces is heterogeneous. Tested with 3B, 8B and 32B LLMs. — [arXiv 2502.12118](https://arxiv.org/abs/2502.12118)
- Sprague et al. (To CoT or not to CoT): a meta-analysis of 100+ papers plus 20 datasets × 14 models finds CoT "gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks". — [arXiv 2409.12183](https://arxiv.org/abs/2409.12183)

### Inferences
- Applied to the researcher's setting by analogy from math only: for 0.5–3B policies on *hard* multi-hop questions, step-level search with a good verifier is predicted to beat SC. On easy questions, SC or BoN is predicted to win, and search over-optimizes. No compute-optimal TTS study on multi-hop QA or CLUTRR with ≤7B was found to confirm this.
- Because math PRMs don't transfer across domains (VersaPRM in Q5) or across policies (Liu et al.), an off-the-shelf PRM is unlikely to work on MuSiQue or CLUTRR. The verifier would have to be task-specific: a symbolic rule table for CLUTRR, a passage-entailment check for multi-hop QA, or a 7B judge.

### Gaps
- There is no compute-optimal TTS study on knowledge or multi-hop QA with ≤7B policies and passages in context.
- Snell et al. used PaLM 2-S*, not open ≤7B models. Liu et al. cover ≤7B but only MATH and AIME.

## Q4. Methods that check consistency between the chain's stated answer and an independent answer (direct or recomputed)

### Takeaway
Several methods implement "recompute and compare" at the process level: rStar mutual consistency, SelfCheck, self-verification, ProbTree's closed-book versus open-book choice, and RAG-Star's answer-aware reward. rStar is the only one with ≤8B results beyond math (StrategyQA). We found no peer-reviewed study that uses direct-answer versus CoT-answer agreement as a verified gating signal for multi-hop QA.

### Cited Findings
- rStar mutual consistency: a second SLM (Phi3-mini) receives a *partial* trajectory as a hint, completes it independently, and the trajectory is accepted only if the answers match. On StrategyQA it adds +0.1 to +5.7 points over the generator alone (e.g., LLaMA2-7B 61.57 → 67.25; Mistral 69.43 → 70.31). The discriminator's identity matters little. — [arXiv 2408.06195, Table 2 & Table 5](https://arxiv.org/abs/2408.06195)
- SelfCheck regenerates each step from the extracted target and context, compares the result with the original step, and uses the per-step checks for weighted voting. Tested on math only (GSM8K, MathQA, MATH). — [arXiv 2308.00436](https://arxiv.org/abs/2308.00436)
- Self-Verification (Weng et al., EMNLP 2023 Findings) takes "the conclusion obtained by CoT as one of the conditions for solving the original problem" and verifies backward, scoring each candidate answer. Results cover arithmetic, commonsense and logical reasoning. — [arXiv 2212.09561](https://arxiv.org/abs/2212.09561)
- ProbTree answers each leaf sub-question both closed-book and open-book and keeps the more confident answer. This is an explicit parametric-versus-context consistency choice, done per hop. — [arXiv 2311.13982](https://arxiv.org/abs/2311.13982)
- RAG-Star's answer-aware reward checks consistency between the LLM's *internally generated* sub-answer and retrieved documents, and refines the answer when they conflict. Removing retrieval from this check hurts Llama-3.1-8B "significantly". — [arXiv 2412.12881, Secs. 4.4 & 5.3](https://arxiv.org/abs/2412.12881)
- CoT-decoding: "the presence of a CoT in the decoding path correlates with a higher confidence in the model's decoded answer". The answer-token margin Δ separates CoT from non-CoT paths, which makes it an answer-level confidence signal usable inside a step-level search. — [arXiv 2402.10200](https://arxiv.org/abs/2402.10200)
- CISC: within-question confidence (ranking a question's own samples) predicts correctness better than cross-question calibration. "The most calibrated confidence method proved to be the least effective". — [arXiv 2502.06233](https://arxiv.org/abs/2502.06233)
- Caveat on agreement signals: in KG-grounded multi-hop QA there is "a significant disparity between answer accuracy and faithfulness of the CoT", so models "often arrive at correct answers through incorrect reasoning". Answer agreement does not certify the chain. — [Nguyen et al., arXiv 2402.11199](https://arxiv.org/abs/2402.11199)
- Caveat on hinted answers: when an answer is hinted in the prompt, models usually select it and produce a CoT that rationalizes it. This parallels the researcher's "copy the stated wrong answer" finding. — [Barez et al., "Chain-of-Thought Is Not Explainability" (Oxford, 2025)](https://aigi.ox.ac.uk/wp-content/uploads/2025/07/Cot_Is_Not_Explainability.pdf)

### Inferences
- The researcher's finding (≤3B models copy stated wrong answers; 7B vetoes them at a 10% follow rate) suggests a cross-size variant of rStar's mutual consistency. A 0.5–3B generator writes the chain; at each hop a 7B model independently recomputes the sub-answer, from the passage or closed-book; disagreement triggers branching or resampling at that step. This is our inference. rStar shows a peer SLM is enough on math and StrategyQA, but the 7B-veto version is untested.
- Because a stated wrong answer propagates through the chain, the useful comparison is per hop (bridge entity versus passage, or versus an independent recomputation), not at the final readout. This is consistent with the researcher's finding and with RAG-Star's per-node design.

### Gaps
- No peer-reviewed paper found that tests direct-answer versus CoT-answer agreement as a step-level or answer-level gate against ground truth for multi-hop QA or CLUTRR. The only hits were practitioner blogs, which are not cited.
- No paper found that uses a *larger* model's veto as the step verifier for a *smaller* generator in a compute-matched comparison with SC.

## Q5. Negative results: when step-level guidance fails for small models

### Takeaway
Intrinsic self-evaluation by small models is close to random. Tree search needs about 90%-accurate discriminators to beat reranking and costs 10–20× more. Math PRMs fail out of domain. Imperfect verifiers cap resampling gains. Self-evaluation beam search loses to CoT on short commonsense chains. Together these make *self*-guided step search with ≤3B models a poor bet, unless the verifier is external, symbolic or a larger model.

### Cited Findings
- rStar Table 6: SLMs' self-rewarding in RAP-style MCTS is near random. "Replacing the self-evaluated [helpfulness score] to random values does not significantly impact the final reasoning performance." Also: "Majority voting in self-consistency requires most traces to be correct, which is often not the case for SLMs." — [arXiv 2408.06195, Sec. 3.1 & App. Table 6](https://arxiv.org/abs/2408.06195)
- ToT is below few-shot CoT on StrategyQA for all 5 SLMs (e.g., Phi3-mini 40.47 vs 63.61; LLaMA2-7B 45.27 vs 58.82). — [arXiv 2408.06195, Table 2](https://arxiv.org/abs/2408.06195)
- Chen et al. (When is Tree Search Useful?, ACL 2024), on text-to-SQL and math: advanced planning (tree search, iterative correction) needs "discriminators with at least 90% accuracy to achieve significant improvements over re-ranking". Current LLM discriminators don't reach that. "Tree search is at least 10–20 times slower but leads to negligible performance gains." — [arXiv 2402.10890](https://arxiv.org/abs/2402.10890)
- Xie et al.: under matched compute (Llama-2 13B), stepwise self-evaluation beam search "lags behind the CoT baseline on commonsense reasoning". Self-evaluation scores on StrategyQA "do not align well with whether the prediction is correct". — [arXiv 2305.00633, Sec. 4.2 & App.](https://arxiv.org/abs/2305.00633)
- Snell et al.: beam search over-optimizes on easy problems at high budgets, and lookahead (MCTS-like) underperforms at equal budget. — [arXiv 2408.03314](https://arxiv.org/abs/2408.03314)
- VersaPRM: "current PRMs have poor performance in other domains". On MMLU-Pro Law, Qwen2.5-Math-PRM gives only +1.3 over majority voting, versus +7.9 for a multi-domain PRM. — [Zeng et al., arXiv 2502.06737](https://arxiv.org/abs/2502.06737)
- Qwen PRM lessons: MC-estimation PRM data (Math-Shepherd style) "typically yields inferior performance and generalization". BoN evaluation is inflated by correct-answer, flawed-process responses. Many PRMs' minimum scores concentrate on the final answer step, which shifts them from process to outcome assessment. — [Zhang et al., arXiv 2501.07301](https://arxiv.org/abs/2501.07301)
- Liu et al.: PRMs generalize poorly across policy models. With Math-Shepherd and RLHFlow PRMs, BoN beats search. — [arXiv 2502.06703](https://arxiv.org/abs/2502.06703)
- Stroebl et al.: imperfect verifiers with non-zero false-positive rates impose an accuracy ceiling on resampling regardless of budget. "Optimal sampling attempts are often fewer than 10." Weaker models have higher false-positive rates. — [arXiv 2411.17501](https://arxiv.org/abs/2411.17501)
- Huang et al. (ICLR 2024): "LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction." — [arXiv 2310.01798](https://arxiv.org/abs/2310.01798)
- Tyen et al. (ACL Findings 2024): poor self-correction "stems from LLMs' inability to find logical mistakes", not their ability to correct them. Given the mistake location, correction boosts downstream performance on 5 tasks. A small classifier trained out of domain finds mistakes better than prompting a large model. — [arXiv 2311.08516](https://arxiv.org/abs/2311.08516)
- Zhang et al. (ACL Findings 2024): small (≤13B) LMs gain from self-correction "when paired with a strong GPT-4-based verifier", but there are "limitations … when using a weak self-verifier for determining when to correct". — [arXiv 2404.17140](https://arxiv.org/abs/2404.17140)
- CoT-decoding: standard sampling and beam search *hurt* Mistral-7B relative to greedy without a CoT prompt: beam (b=10) 6.7, top-k 4.9, nucleus 6.4, temperature 7.5, versus greedy 9.9. Naive diversity is not free. — [arXiv 2402.10200, Table 4](https://arxiv.org/abs/2402.10200)
- SC can *decrease* with more samples on StrategyQA for SLMs (e.g., Phi3-mini SC@8 64.34 → SC@128 59.53). — [arXiv 2408.06195, Table 2](https://arxiv.org/abs/2408.06195)
- CLUTRR SLM "distraction effect": noisy self-generated facts degrade performance even when expert hints are present (Gemma 3 1B/4B, Llama 3.2 3B). — [arXiv 2607.14149](https://arxiv.org/abs/2607.14149)

### Inferences
- These negatives agree with the researcher's finding that ≤3B models cannot reject their own stated wrong answers. For those models, any step verifier should be external: a symbolic rule table on CLUTRR, passage entailment, or the 7B model that showed veto behaviour. It should not be self-evaluation.
- Given the ≥90%-discriminator threshold (Chen et al.) and the low optimal sample count with imperfect verifiers (Stroebl et al.), the practical design on 2×T4 is probably verifier-gated *reranking or resampling at the step* with small N, rather than deep MCTS.
- The StrategyQA results (SC flat or declining with N; self-eval beam search below CoT) suggest that on short, knowledge-heavy chains, SC itself is a weak baseline to beat, and the step-level gain has to come from fixing specific hop errors (wrong bridge entity, distractor uptake).

### Gaps
- No paper found that isolates distractor-passage harm in multi-hop QA and tests step-level interventions against it with ≤7B models and compute-matched SC.
- No paper found that measures PRM or verifier accuracy on multi-hop QA steps for ≤7B generators, so it is unknown whether any available verifier clears the ~90% threshold there.
