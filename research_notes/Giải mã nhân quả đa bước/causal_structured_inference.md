# Inference-time methods that inject causal or structural information into LLM reasoning (2023–2026), with a focus on MuSiQue / CLUTRR and models of 7B or smaller

Scope note: all numbers below were read directly from the paper PDFs (arXiv or ACL Anthology) unless marked otherwise. "Training-free" means the LLM gets no gradient updates. Auxiliary trained components (encoders, GNNs, rule induction) are flagged separately.

## Q1. Causal-inference-inspired inference methods (front-door/back-door adjustment, deconfounding, counterfactual/contrastive decoding, intervention-based correction, mediation)

### Takeaway
Most "causal" prompting methods are sample-then-reweight wrappers around CoT, not token-level decoders. Causal Prompting (front-door), DeCoT (IV + front-door), ACPS and CaCo-CoT sample many chains, intervene on the prompt or context, and aggregate. They report gains of roughly 3–10 points on MuSiQue/HotpotQA for 7–8B models. None of them guarantees that the answer is computed from the chain. Methods that do enforce chain-to-answer dependence either train the model (FRODO, LIFT, CASE's TCA stage) or structurally block the shortcut. CASE's inference-time attention mask from instruction to answer is the one training-free component found that directly targets the "answer bypasses the chain" failure.

### Cited Findings
**Diagnostic background (why CoT is non-causal)**
- Bao et al. (COLING 2025; first titled "LLMs with Chain-of-Thought Are Non-Causal Reasoners") use interventions to recover the SCM that LLMs implicitly follow. LLMs "often deviate from the ideal causal chain, resulting in spurious correlations and potential consistency errors". In-context learning with examples *strengthens* the causal structure. SFT and RLHF *weaken* it. "The causal structure cannot be strengthened by enlarging the model size only." — [arXiv 2402.16048](https://arxiv.org/abs/2402.16048)
- FRODO (Paul et al., EMNLP Findings 2024) runs causal mediation analysis on 12 LLMs and finds that "LLMs do not reliably use their intermediate reasoning steps". Example of the effect sizes: on StrategyQA, ChatGPT reaches 69.2% CoT accuracy with NIE 15.3 and NDE 9.1, and GPT-4 reaches 93.5% with NIE 40.0 and NDE 22.2. On GSM8K, LLaMA-2-7B-Chat (RLHF) has CIE 1.4 vs CDE 36.7, while in-context LLaMA-2-7B has CIE 45.6 vs CDE 0.9. So RLHF models are dominated by the direct effect. — [arXiv 2402.13950](https://arxiv.org/abs/2402.13950)

**Causal Prompting: front-door adjustment (Zhang et al., AAAI 2025)**
- Mechanism: CoT is the mediator in X→R→A.
  - P(r|do(X)) is estimated by sampling m CoTs, embedding them with an encoder, running K-means, and weighting each cluster centre by cluster size.
  - P(A|do(r)) is estimated with a "prompting version" of the NWGM approximation. Demonstrations are retrieved, each a (question, wrong CoT, correct CoT) triple from the training set nearest to the representative CoT, and the LLM is asked to *improve* that CoT T times.
  - The answer comes from causality-weighted voting. — [arXiv 2403.02738](https://arxiv.org/abs/2403.02738); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/34777)
- Training requirement: the LLM is untouched, but the CoT encoder is fine-tuned with contrastive learning to align its representation space with the LLM. The method also needs a training set of wrong/correct CoT pairs. Front-door adjustment is applied only to samples below a confidence threshold s. Applying it to all samples (s = 1.0) "drops significantly". — [arXiv 2403.02738](https://arxiv.org/abs/2403.02738)
- Compute cost: the reported configuration uses m = 40 sampled CoTs, K = 8 clusters and T = 10 answers per cluster, so K·T = 80 answers plus 40 chains, about 120 LLM generations per question. — [arXiv 2403.02738](https://arxiv.org/abs/2403.02738)
- Results (Table 1). For MuSiQue and HotpotQA, numbers are EM/F1.
  - LLaMA2: MuSiQue CoT 18.71/30.27, CoT-SC 31.16/46.36, Causal Prompting 34.68/48.79. HotpotQA CoT-SC 49.30/68.53 vs CP 52.20/70.88.
  - LLaMA3 (the paper mentions LLaMA3-8B): MuSiQue CoT 38.88/54.38, CoT-SC 41.37/59.78, CP 48.07/64.23. HotpotQA CoT-SC 52.70/75.43 vs CP 58.50/78.18. GSM8K CoT-SC 82.41 vs CP 87.95.
  - GPT-3.5: MuSiQue CoT-SC 46.27/60.83 vs CP 50.13/65.40.
  - The authors summarise the gains on math and multi-hop QA as "approximately 5%–10%". — [arXiv 2403.02738](https://arxiv.org/abs/2403.02738)

**DeCoT: deconfounded CoT via instrumental variable plus front-door adjustment (Wu et al., ACL 2024), training-free**
- Mechanism:
  - Sample N CoTs C_i ~ p(C|E,Q), where E is the context and Q the query.
  - Ask the LLM to extract T factual entities from E and to propose P counterfactual replacements per entity, giving counterfactual contexts E*_{j,k}.
  - For each CoT, compute the average causal effect ACE(C_i) = E[p(A|E,Q,C_i) − p(A|E*,Q,C_i)], i.e. how much answer confidence *drops* when the evidence is counterfactually edited while the chain is held fixed.
  - Select the mediator C* ~ softmax[p(C_i|E,Q)·ACE(C_i)] and answer with p(A|E,Q,C*).
  - External knowledge acts as the instrumental variable, and the CoT is the front-door mediator. — [ACL Anthology 2024.acl-long.758](https://aclanthology.org/2024.acl-long.758)
- Results (Table 1, EM/F1).
  - LLaMA-2-7B: MuSiQue CoT 20.22/27.46, CAD 21.62/28.10, DeCoT 22.75/30.99. HotpotQA CoT 8.86/26.79 vs DeCoT 10.03/31.48. Average F1 across 4 datasets 30.52 → 38.72.
  - Flan-T5-XXL (11B): MuSiQue 19.53/27.61 → 20.56/30.54.
  - GPT-3.5: MuSiQue 22.30/34.22 → 31.28/44.14.
  - Baselines are CoT, CoT without context, and context-aware decoding (CAD). — [ACL Anthology 2024.acl-long.758](https://aclanthology.org/2024.acl-long.758)
- The authors say the method uses "similar numbers of API calls" to multi-sample CoT methods. In practice it needs N CoTs × (1 + T·P) answer-likelihood evaluations. — [ACL Anthology 2024.acl-long.758](https://aclanthology.org/2024.acl-long.758)
- Causal Prompting's authors criticise DeCoT for depending on instrumental variables and counterfactual knowledge, which restricts it to knowledge-intensive tasks. — [arXiv 2403.02738](https://arxiv.org/abs/2403.02738)

**ACPS: Adaptive Causal Prompting with Sketch-of-Thought (Li et al., EACL 2026 Findings)**
- The method uses an SCM to choose adaptively between standard front-door and *conditional* front-door adjustment per query. It replaces verbose CoT with concise Sketch-of-Thought to cut tokens and needs no task-specific retraining. — [ACL Anthology 2026.findings-eacl.234](https://preview.aclanthology.org/credits/2026.findings-eacl.234)
- Results (Table 1, MuSiQue EM/F1):
  - Ministral-3B: CoT 31.05/41.30, CoT-SC 40.00/50.02, CAD 41.25/52.65, DeCoT 40.75/53.66, CP 43.48/51.31, ACPS 51.72/60.65. HotpotQA ACPS 50.00/66.67 vs CP 38.03/48.98.
  - LLaMA-3.1-8B: CoT 38.64/48.23, DeCoT 41.23/54.56, CP 44.94/53.15, ACPS 49.00/59.71.
  - Reasoning traces typically have fewer than 3 steps, vs more than 6 for CoT on HotpotQA. — [ACL Anthology 2026.findings-eacl.234](https://preview.aclanthology.org/credits/2026.findings-eacl.234)
- Caveat: ACPS's numbers for the same baselines do not match the CP paper. For example, GPT-3.5 CoT on MuSiQue is EM 45.94 in ACPS but 41.37 in CP. Round values such as 50.00 and 66.67 suggest small evaluation subsets. — compare [ACPS](https://preview.aclanthology.org/credits/2026.findings-eacl.234) with [CP](https://arxiv.org/abs/2403.02738)

**CaCo-CoT (Tang et al., "Towards CausalGPT", arXiv 2308.11914)**
- Training-free multi-agent framework. "Reasoners" generate chains that mimic human causal reasoning, and an "evaluator" checks the causal consistency of each chain from a non-causal and a counterfactual perspective, then reaches consensus.
- Evaluated on ScienceQA, Com2Sense and BoolQ with GPT-3.5-turbo, Claude and Qwen1.5-32B, not on MuSiQue or CLUTRR. — [arXiv 2308.11914](https://arxiv.org/abs/2308.11914)

**Contrastive / counterfactual decoding**
- Contrastive Decoding for reasoning (O'Brien & Lewis 2023) is training-free. It scores tokens by expert-logit minus β·amateur-logit (LLaMA-65B expert, 1.5B amateur) and improves GSM8K for LLaMA models "by up to 8 absolute" points.
  - Relevant to the copying finding: CD "performs less surface-level copying from the prompt than greedy decoding and misses fewer reasoning steps".
  - Limitations: CD "is worse at arithmetic but better at logical reasoning" and "can harm factual recall". — [arXiv 2309.09117](https://arxiv.org/abs/2309.09117)
- Context-aware decoding (CAD; Shi et al., NAACL 2024) contrasts output probabilities with and without the context. It gives a 2.9x improvement for LLaMA-30B on a knowledge-conflict QA task and a 14.3% factuality gain on summarisation. — [arXiv 2305.14739](https://arxiv.org/abs/2305.14739)
- On multi-hop QA, CAD gives only small gains over CoT for LLaMA-2-7B on MuSiQue (EM 20.22 → 21.62), below DeCoT. — [ACL Anthology 2024.acl-long.758](https://aclanthology.org/2024.acl-long.758)

**Causal faithfulness methods (mostly trained)**
- FRODO (trained). It combines an inference module trained with an implicit causal reward and a reasoning module trained with a counterfactual/causal preference objective (DPO plus a margin-ranking loss).
  - Models: T5-large (770M), T5-3B, LLaMA-2-7B, Mistral-7B. Benchmarks are StrategyQA, QuaRel, OBQA and QASC, not MuSiQue or CLUTRR.
  - LLaMA-2-7B StrategyQA: base 67.2, SFT+CoT 79.4, FRODO 81.5. QASC: 54.6 → 63.4.
  - Mistral-7B StrategyQA: SFT+CoT 78.2 → FRODO 81.9. — [arXiv 2402.13950](https://arxiv.org/abs/2402.13950)
- LIFT (formerly FRIT; arXiv 2509.13334, trained). Each step is scored by its causal influence: replace the step with a domain-matched irrelevant fact and measure the drop in answer log-probability. The model is then fine-tuned on high-scoring traces. Tested on Mistral-7B-Instruct, Qwen3-8B and Qwen3-4B-Instruct. This is the same intervention as a "chain-step intervention" score, but used for data selection and fine-tuning rather than decoding. — [arXiv 2509.13334](https://arxiv.org/abs/2509.13334)
- CASE (arXiv 2607.18820, preprint, no venue found) combines training-time causal alignment (TCA) with *inference-time structural enforcement* (ISE). ISE masks direct attention from instruction tokens to answer tokens (pre-softmax logit set to −∞) during answer generation, so the answer can only reach the question through CoT-token hidden states.
  - Llama-3.1-8B ablation, **ISE alone (no training)**:
    - ARC-Easy: faithfulness score FT 0.042 → 0.610, controlled direct effect CDE 0.358 → 0.018, accuracy 92.2 → 93.4.
    - ARC-Challenge: FT 0.136 → 0.560, accuracy 86.2 → 84.4.
  - Full CASE: G-mean faithfulness 0.646/0.650. Models are Llama-3.1-8B, Qwen3-8B and R1-Distill-Qwen-7B, on ARC-E/C, StrategyQA and LogiQA. — [arXiv 2607.18820](https://arxiv.org/abs/2607.18820)
- Reasoning-rigidity work (arXiv 2603.22816) proposes SLRC, which removes one reasoning sentence at a time, and LC-CoSR, a training intervention that directly rewards step-level causal dependency. The authors report that only RL-trained models show genuine step necessity. — [arXiv 2603.22816](https://arxiv.org/abs/2603.22816)
- FACT-E (ACL 2026 Findings) is a causality-inspired *evaluation* of CoT trustworthiness, not a decoding method. — [arXiv 2604.10693](https://arxiv.org/abs/2604.10693)

### Inferences
- DeCoT's ACE is the closest published analogue to an intervention-scored chain decoder. It intervenes on *evidence entities* with the chain held fixed and uses the result to **select among whole chains**, not to reweight tokens. Its gain at 7B on MuSiQue is small: +2.5 EM / +3.5 F1 over CoT. That is consistent with the researcher's HGCD null result: readout-level reweighting cannot help if the chain carries no causal signal to begin with.
- The large gains reported by CP and ACPS at 7–8B come mostly from heavy self-consistency sampling plus ICL "repair" demonstrations. CoT-SC alone accounts for much of the gain over CoT. So they are not evidence that the small model composes over its chain.
- The researcher found that ≤3B models copy a stated answer and that 7B "vetoes" without using its hops. Both point to a direct Q→A (or stated-answer→A) path. CASE-style attention masking (answer tokens cannot attend to question tokens) is the cheapest training-free way to remove that path. Bao et al.'s result that ICL strengthens and SFT/RLHF weakens the causal structure suggests testing base (non-instruct) checkpoints with few-shot prompts.

### Gaps
- No paper found that applies front-door/IV prompting (CP, DeCoT, ACPS) to CLUTRR.
- No training-free *token-level* causal or counterfactual decoder was found with reported MuSiQue gains for models of 7B or smaller. CAD is the nearest, and its gain is small.
- CASE's ISE-only accuracy on multi-hop QA (MuSiQue) is not reported. CASE is an unreviewed preprint.
- CaCo-CoT's exact numbers were not extracted, because the PDF table was garbled.

## Q2. Faithful-by-construction approaches (LLM translates to a program/logic, a deterministic solver answers) and CLUTRR accuracies

### Takeaway
On CLUTRR, delegating composition to a symbolic engine is the strongest and most reliable lever, and the only one with evidence at 3–4B. Faithful CoT showed +10 points with Codex using a Datalog engine and a kinship transitivity table. Path-of-Thoughts symbolic variants reach 66–68% vs CoT 53–58% on GPT-4-class models. Reason Popper-ly (2026) lifts Qwen-3.5-4B, Gemma-4-E4B and LLaMA-3.2-3B from 14–19% to 42–63% on 8–10-hop chains. The binding constraint for small models is *fact extraction*, not composition.

### Cited Findings
- Faithful CoT (Lyu et al., IJCNLP-AACL 2023). An LLM translates the question into interleaved NL plus a symbolic language. For CLUTRR this is relation statements solved by "a simple relational inference engine" that recursively applies transitivity rules (e.g. son@daughter = grandson) from Zhang et al. (2022).
  - code-davinci-002, greedy decoding, CLUTRR: Standard 42.0, CoT 48.5, LtM 47.2, **Faithful CoT 58.9**.
  - Self-consistency decoding: CoT 45.7, LtM 50.9, **Faithful CoT 71.9**.
  - The relative gain on relational inference is 21.4% with greedy and 41.3% with SC.
  - On StrategyQA, FCoT is *below* CoT (63.0 vs 72.5 greedy). The authors attribute this to the difficulty being knowledge retrieval and to Datalog syntax errors (29% of errors). — [arXiv 2301.13379](https://arxiv.org/abs/2301.13379)
- Path-of-Thoughts (Zhang et al., Huawei/McGill; arXiv 2412.17963 v2, Mar 2026; venue not stated in the PDF).
  - Three stages: graph extraction (entities, relations, attributes), query-relevant path identification, and reasoning by an LLM (PoT-LLM) or a symbolic solver (PoT-Symbolic). "Without requiring fine-tuning or extensive LLM calls."
  - CLUTRR, single trial:
    - GPT-3.5-turbo: CoT 35.6, CoT-SC 37.1, LLM-ASP 32.8, PoT-LLM 35.1, **PoT-Symbolic 54.1**
    - GPT-4-turbo: CoT 53.0 → PoT-Symbolic 66.1
    - GPT-4o: CoT 57.6, LLM-ASP 56.7, PoT-Symbolic 67.7
    - Llama3-70B: CoT 52.8, LLM-ASP 54.8, PoT-Symbolic 65.9
  - Removing path identification drops PoT-LLM on CLUTRR from 61.9 to 50.7.
  - No models of 7B or smaller were tested. — [arXiv 2412.17963](https://arxiv.org/abs/2412.17963)
- Reason Popper-ly (Conference on Neurosymbolic Learning and Reasoning 2026, arXiv 2607.23019).
  - Offline: the Popper ILP system induces a relation-composition rule table from LLM reasoning traces on training stories, validated against a gold kinship table.
  - Online: each inferred CoT step is checked against the table, the violation type is diagnosed, invalid steps are replaced with symbolic repairs, and the remaining suffix is regenerated. The answer is then conditioned on a verified trace.
  - CLUTRR terminal accuracy, 8–10 hops (Endogenous CoT → Exogenous symbolic → Popper-ly):
    - Qwen-3.5-4B: 19.09 → 46.72 → **61.93**
    - Gemma-4-E4B: 14.39 → 42.22 → **62.74**
    - LLaMA-3.2-3B: 18.72 → 33.48 → **42.48**
    - GPT-5.4: 59.91 → 69.53 → 75.09
  - At 2–3 hops: LLaMA-3.2-3B goes 39.48 → 63.93 → 70.36, and Qwen 56.95 → 75.07 → 73.23.
  - The LLM gets no gradient training. Only ILP rule induction on training traces is needed. — [arXiv 2607.23019](https://arxiv.org/abs/2607.23019)
- Hypotheses-to-Theories (HtT; Zhu et al., arXiv 2310.07064). The LLM induces a rule library from training examples (2,000 CLUTRR examples), then deduces with retrieved rules in the prompt.
  - Symbolic CLUTRR, GPT-3.5: 5-shot CoT 21.2 → +HtT 37.6 → +HtT with GPT-4 rules 48.4.
  - GPT-4: 50.6 → 64.3.
  - The supervised EdgeTransformer scores 71.8.
  - On textual CLUTRR, rule gains are "not significant for GPT-3.5, since it often produces errors other than incorrect rules". — [arXiv 2310.07064](https://arxiv.org/abs/2310.07064)
- Small LMs plus KG grounding (Kelesis et al., Causal NeSy workshop at ESWC 2026, arXiv 2607.14149). The SLM extracts triplets, and an RGCN trained on CLUTRR k ∈ {2,3,4} supplies a relation "hint".
  - Story-only accuracy: Gemma-3-1B 7.54, Llama-3.2-3B 16.13, Gemma-3-4B 13.45.
  - Story + GNN hint from self-extracted facts: 12.60 / 20.32 / 19.75.
  - Oracle hint: 34.16 / 62.79 / 59.16.
  - Adding self-extracted facts to the story *hurts* Gemma-4B (13.45 → 1.64), which the authors call a "distraction effect".
  - Extraction recall is about 71%. The RGCN reaches 60.69% on oracle facts and collapses on SLM facts. — [arXiv 2607.14149](https://arxiv.org/abs/2607.14149)
- Logic-LM (Pan et al., EMNLP 2023 Findings): LLM → symbolic formulation → deterministic solver, plus self-refinement from solver error messages. It averages +39.2% over standard prompting and +18.4% over CoT on ProofWriter, PrOntoQA, FOLIO, LogicalDeduction and AR-LSAT. CLUTRR was not evaluated. — [arXiv 2305.12295](https://arxiv.org/abs/2305.12295)
- LINC (Olausson et al., EMNLP 2023, Outstanding Paper): LLM → FOL plus a Prover9 prover with majority vote. StarCoder+ (15.5B) with LINC beats GPT-3.5 and GPT-4 CoT on ProofWriter by 38 and 10 absolute points. — [arXiv 2310.15164](https://arxiv.org/abs/2310.15164)
- SatLM (Ye et al., NeurIPS 2023): declarative specification plus an SMT solver. It beats program-aided LMs by 23% on a hard GSM subset and sets SoTA on LSAT and BoardgameQA. — [arXiv 2305.09656](https://arxiv.org/abs/2305.09656)
- Selection-Inference (Creswell et al., arXiv 2205.09712): alternates a *selection* step (pick facts) and an *inference* step that sees only the selected facts. A 7B LLM, 5-shot with no fine-tuning, gains "over 100%" vs a vanilla baseline on 10 logical reasoning tasks and beats a 280B baseline. — [arXiv 2205.09712](https://arxiv.org/abs/2205.09712)
- Original CLUTRR: GNNs on symbolic inputs far outperform text models (BERT, MAC), which motivates neuro-symbolic decomposition. — [CLUTRR, Sinha et al. 2019](https://arxiv.org/abs/1908.06177v1)

### Inferences
- For CLUTRR with ≤7B models, a "parse to kinship triples, then compose with a fixed composition table" pipeline is a hard baseline that any causal decoder should beat or match. Faithful CoT's solver and Popper-ly's "Exogenous" setting are both of this kind. Popper-ly shows that the exogenous route alone gives +15 to +28 points at 8–10 hops for 3–4B models.
- The Kelesis et al. results (oracle hint 62.79 vs self-extracted 20.32 for Llama-3B) and HtT's note that GPT-3.5's errors go beyond rules both locate the small-model bottleneck in extraction and grounding of facts. Composition is not the bottleneck. A decoder design should therefore constrain *extraction* (e.g. force each hop to name an entity pair from the story) more than composition.
- Popper-ly's verify-repair-regenerate loop is effectively an "intervention-based self-correction": it intervenes on wrong hops and regenerates downstream. It is the only method found with large, measured gains on CLUTRR at 3–4B.

### Gaps
- Faithful CoT's CLUTRR results for the alternative LMs in its Appendix C.3 were not extracted. No CLUTRR accuracy for Faithful CoT with a model of 7B or smaller was found.
- Logic-LM, LINC and SatLM do not report CLUTRR.
- Scallop / "relational programming with foundation models" results on CLUTRR were not retrieved.
- Popper-ly and Kelesis et al. are 2026 venue papers (NeSy conference, ESWC workshop), not ACL/NeurIPS main track.

## Q3. Structured decoding for multi-hop QA (decomposition with sub-answer tracking, entity/triple chains, graph plans, hop-level grounding) and gains for ≤7B models

### Takeaway
Explicit decomposition with sub-answer substitution, sub-question retrieval or triple-chain evidence gives consistent but moderate gains on MuSiQue/2Wiki/HotpotQA. Examples: Self-Ask +1–10 EM with GPT-3; IRCoT up to +15 F1, and it works with Flan-T5 0.7–11B; TRACE +14 EM on average with LLaMA3-8B; PER-QA +1.6–3.9 F1 with Llama-3.1-8B executors. Absolute MuSiQue scores for ≤8B models stay low, around 20–50 EM depending on setting. Most of these methods are training-free.

### Cited Findings
- Self-Ask (Press et al., EMNLP 2023 Findings) introduces the "compositionality gap": models answer the sub-facts but not the composition, and the gap does not narrow with scale. Self-Ask asks explicit follow-up questions and answers each one, optionally with a search engine.
  - davinci-002, EM:
    - 2Wiki: Direct 25.4, CoT 29.8, Self-Ask 30.0, Self-Ask+Search 40.1
    - MuSiQue: Direct 5.6, CoT 12.6, Self-Ask 13.8, +Search 15.2
    - Bamboogle: CoT 46.4 → 57.6 / 60.0
  - For smaller models, ending the prompt with "Are follow up questions needed here: Yes. Follow up:" improves performance. — [arXiv 2210.03350](https://arxiv.org/abs/2210.03350)
- IRCoT (Trivedi et al., ACL 2023) interleaves retrieval with CoT, using each CoT sentence as the next query. This is training-free.
  - With GPT-3, retrieval improves by up to 21 points and QA by up to 15 points on HotpotQA, 2Wiki, MuSiQue and IIRC.
  - It also works with Flan-T5 at 11B, 3B and 0.7B. Flan-T5-XL (3B) with IRCoT beats 58x-larger GPT-3 with one-step retrieval. — [arXiv 2212.10509](https://arxiv.org/abs/2212.10509)
- ProbTree (Cao et al., EMNLP 2023 Findings) decomposes the question into a query tree and solves leaf to root. Each node picks the more confident of closed-book and open-book answers, and parents reason over child answers to recover from local errors. The authors report it "outperforms SOTA methods significantly" on HotpotQA, MuSiQue and 2WikiMQA. The exact numbers were not extracted (garbled table). — [arXiv 2311.13982](https://arxiv.org/abs/2311.13982)
- TRACE (Fang et al., EMNLP 2024 Findings) turns retrieved documents into KG triples with LLaMA3-8B-Instruct prompted in context, then autoregressively builds a *reasoning chain of linked triples* with an e5-mistral-7b bi-encoder ranker and an LLM selector.
  - The reader answers from the triples (TRACE-Triple) or from their source documents (TRACE-Doc), in zero-shot ICL.
  - Average EM gain over using all retrieved documents on HotpotQA, 2Wiki and MuSiQue: **+14.03% (Triple)** and **+13.46% (Doc)**.
  - Readers tested: LLaMA3-8B, Mistral-7B, Gemma-7B. No LLM fine-tuning is described in the setup. — [arXiv 2406.11460](https://arxiv.org/abs/2406.11460); [ACL Anthology](https://preview.aclanthology.org/new-sigs/2024.findings-emnlp.496/)
- PER-QA (Liu et al., ACL 2025) uses a Planner-Executor-Reasoner. The planner builds a graph-structured plan with sub-answer placeholders (e.g. p2: "How many campuses does <a*1> have?"), and the executor solves sub-questions in topological order.
  - The planner was GPT-4o and the other modules Llama-3.1-8B-Instruct.
  - F1 gains: +3.9 / +1.7 / +1.6 over vanilla Llama models, and +0.4 / +3.6 / +2.6 over RAG.
  - Gains appear in only 3 of 10 question types vs GPT-4o-mini, which the authors attribute to "memory" answers rather than reasoning.
  - The paper also identifies "fortuitous reasoning continuance" (correct answers reached through wrong intermediate steps) in RAG systems. — [ACL Anthology 2025.acl-long.1142](https://aclanthology.org/2025.acl-long.1142.pdf)
- ChainRAG (Zhu et al., ACL 2025) identifies "lost-in-retrieval": sub-questions lose their key entities. It sequentially completes missing entities in each sub-question with the previous sub-answers and retrieves from a sentence graph. It was evaluated on MuSiQue, 2Wiki and HotpotQA, but only with large models (GPT-4o-mini, Qwen2.5-72B, GLM-4-Plus). — [ACL Anthology 2025.acl-long.1089](https://aclanthology.org/2025.acl-long.1089.pdf)
- HippoRAG (NeurIPS 2024) combines an LLM-extracted KG with Personalized PageRank. It reports up to 20% gains on multi-hop QA. Single-step HippoRAG is comparable to or better than IRCoT while "10–20 times cheaper and 6–13 times faster", and combining it with IRCoT adds further gains. — [arXiv 2405.14831](https://arxiv.org/abs/2405.14831)
- EfficientRAG (arXiv 2408.04259) generates the next-hop query without an LLM call per iteration, using a small trained labeler/filter, and filters irrelevant chunks. It is evaluated on HotpotQA, 2Wiki and MuSiQue with Llama-3-8B among the generators. The training data was built with Llama-3-70B-Instruct, so this method is *not* training-free. — [arXiv 2408.04259](https://arxiv.org/abs/2408.04259)
- Think-on-Graph (ICLR 2024) runs LLM-guided beam search over a KG. It targets KGQA, not MuSiQue. — [arXiv 2307.07697](https://arxiv.org/abs/2307.07697)
- MuSiQue was built so that a single-hop model loses about 30 F1, making it resistant to shortcuts. Its DAG annotation marks which sub-question depends on which earlier answer. — [MuSiQue, TACL 2022](https://aclanthology.org/2022.tacl-1.31.pdf)

### Inferences
- In decomposition methods with placeholder substitution (Self-Ask, PER-QA's <a*1>, ChainRAG's entity completion), hop k+1 is literally conditioned on the string of hop k's answer. A wrong bridge entity therefore changes the downstream question. This is the "dependence by construction" property the researcher's ground-truth interventions found missing in free-form CoT.
- TRACE's finding that compact triple chains are "often sufficient" as reader context suggests a decoder for MuSiQue could restrict the reader's view to the extracted chain, as Selection-Inference does. That removes the question/passages→answer shortcut.
- Most reported gains are with retrieval in open-domain settings. In MuSiQue's distractor setting, used by CP and DeCoT, absolute EM for 7–8B models stays in roughly the 20–50 range across papers. Numbers are not comparable across papers because of different subsets and settings.

### Gaps
- No paper found on token-level *constrained decoding* that forces each hop to cite a passage or entity ID, with MuSiQue results for ≤7B models. Graph-constrained reasoning over KG tries exists for KGQA but was not verified here.
- Exact ProbTree, HippoRAG and EfficientRAG numbers for ≤7B on MuSiQue were not extracted.
- "Graph of Thoughts" style methods were not found with MuSiQue/CLUTRR evaluations at ≤7B.
- StructRAG, HippoRAG 2, GenGround and BeamAggR were not checked.

## Q4. Which methods make the answer depend on the chain by construction (compute rather than copy)?

### Takeaway
Only methods where the final answer is *produced from the chain's content by something other than the free-running LM*, or where the LM is *structurally blocked* from seeing the shortcut, guarantee dependence. Examples:
- a deterministic solver or composition table (Faithful CoT, Logic-LM, LINC, SatLM, PoT-Symbolic, Popper-ly "Exogenous")
- an answer step that only sees the selected facts (Selection-Inference, TRACE-Triple reader)
- sub-answer placeholder substitution (Self-Ask, PER-QA, ChainRAG)
- attention masking from the question to the answer (CASE-ISE)

Causal-prompting wrappers (CP, DeCoT, ACPS, CaCo-CoT) and contrastive decoders (CD, CAD) only *select or reweight*. The LLM still reads the question when answering.

### Cited Findings
- Faithful CoT explicitly "guarantees" that the reasoning chain causes the answer: the solver derives the answer only from the translated symbolic chain. — [arXiv 2301.13379](https://arxiv.org/abs/2301.13379)
- Logic-LM: a "deterministic symbolic solver performs inference on the formulated problem". — [arXiv 2305.12295](https://arxiv.org/abs/2305.12295)
- SatLM: offloading reasoning to a theorem prover "can guarantee the correctness of the answer with respect to the parsed specification". — [arXiv 2305.09656](https://arxiv.org/abs/2305.09656)
- PoT-Symbolic and LLM-ASP use LLMs only for semantic parsing, with symbolic solvers for reasoning. — [arXiv 2412.17963](https://arxiv.org/abs/2412.17963)
- Popper-ly's Exogenous mode does symbolic composition with the learned rule table. Popper-ly itself repairs violated steps and regenerates, so the final answer is LLM-produced but conditioned on a verified chain. It beats pure Exogenous on harder instances "by preserving the model's successful reasoning while correcting only verifiable reasoning failures". — [arXiv 2607.23019](https://arxiv.org/abs/2607.23019)
- Selection-Inference: the inference step sees only the selected facts, which yields a "causal natural-language-based reasoning trace". — [arXiv 2205.09712](https://arxiv.org/abs/2205.09712)
- CASE-ISE removes the explicit Z→Y attention path, so the answer "relies more on the generated CoT, though it may slightly affect accuracy". With Llama-3.1-8B, ISE alone cuts CDE from 0.358 to 0.018 on ARC-Easy. — [arXiv 2607.18820](https://arxiv.org/abs/2607.18820)
- PER-QA plans with explicit sub-answer variables (<a*1>) resolved in topological order. — [ACL Anthology 2025.acl-long.1142](https://aclanthology.org/2025.acl-long.1142.pdf)
- DeCoT, in contrast, answers with p(A|E,Q,C*), so the LLM still conditions on the full question and context. The chain is selected, not enforced. — [ACL Anthology 2024.acl-long.758](https://aclanthology.org/2024.acl-long.758)
- Causal Prompting's final answer also comes from LLM queries with the full prompt, aggregated by weighted voting. — [arXiv 2403.02738](https://arxiv.org/abs/2403.02738)
- CD empirically *reduces* prompt copying in CoT but gives no guarantee. — [arXiv 2309.09117](https://arxiv.org/abs/2309.09117)

### Inferences
Design implications for a ≤7B "causal" decoder on MuSiQue and CLUTRR, synthesised from the above:
1. **Cut the shortcut structurally.** At answer time, mask attention from the question/context to the answer tokens (CASE-ISE), or feed the reader only the hop chain (Selection-Inference, TRACE-Triple). This directly targets the observed copying and veto behaviour.
2. **Make hops typed and substitutable.** Emit (head, relation, tail) per hop, with placeholders resolved from the previous tail (Self-Ask/PER-QA style). Then hop k+1's input is a function of hop k's output, and the researcher's ground-truth interventions become well-defined, observable edits.
3. **For CLUTRR, compose exogenously, or verify and repair per step against a composition table** (Faithful CoT, Popper-ly). This yields +23 to +48 points at 3–4B on long chains. The decoding/LLM budget should then go to extraction accuracy, the dominant error at small scale per Kelesis et al.
4. **Use interventions to select, not to reweight tokens.** DeCoT-style ACE over whole chains is cheap to add on top of steps 1–3 and is training-free. Readout-level contrast without these steps gave no gain in the researcher's HGCD, consistent with DeCoT's small 7B gains.
5. **Prefer base or ICL checkpoints over instruct/RLHF ones** when the goal is CoT→answer dependence (Bao et al.; FRODO's LLaMA-2-7B-Chat CIE 1.4 vs CDE 36.7).

### Gaps
- No study found that measures, for the "dependence by construction" methods, whether ground-truth interventions on intermediate hops flip the answer in ≤7B models on MuSiQue. Such a measurement would directly test the researcher's hypothesis.
- No study found that combines attention masking (CASE-ISE) with multi-hop decomposition, or reports its effect on MuSiQue/CLUTRR accuracy.
- Compute cost comparisons across methods are not reported on a common basis. CP needs about 120 generations per question. HippoRAG reports 10–20x cheaper than IRCoT. PoT claims few LLM calls. Popper-ly needs one CoT plus suffix regeneration per violation.
