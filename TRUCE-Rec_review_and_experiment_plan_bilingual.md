# TRUCE-Rec 顶会级项目审阅与正式实验方案  
# TRUCE-Rec Top-Conference Review and Formal Experiment Plan

**Repository / 仓库**: <https://github.com/appleweiping/TRUCE-Rec.git>  
**Prepared by / 审阅角色**: LLM4Rec top-conference reviewer + artifact/reproducibility reviewer  
**Date / 日期**: 2026-05-02  
**Status / 状态**: R2 passed as full single-dataset paper-candidate protocol; real-LLM evidence still pending.

---

## 0. Scope and Evidence / 范围与证据边界

### 中文

本文档整理当前 TRUCE-Rec 项目的已有实现、文件结构、功能、研究 idea、贡献边界、Storyflow 关系，以及从现在开始应如何进入正式实验。  
我没有在本地重新运行代码；结论基于：

1. GitHub 公开仓库页面和文档；
2. 你提供的阶段性 Gate 报告；
3. 当前项目约定：Phase 1–7 已完成，R1 pilot PASS，R2 full single-dataset PASS；
4. 代码仍需通过真实 LLM / 多数据集 / 候选敏感性实验才能形成论文主结论。

**重要边界**：

- Smoke/mock 不是论文证据。
- MockLLM 不是 real LLM evidence。
- R1 是 pilot / non-paper evidence。
- R2 是 full single-dataset paper-candidate analysis，但如果仍使用 MockLLM，则只能作为非 LLM 真实链路和 protocol evidence。
- 真正论文主结果至少需要 R2-real-LLM subgate + multi-dataset 或多域验证。

### English

This document summarizes the current TRUCE-Rec project: implemented modules, file-level functionality, research idea, contribution boundary, Storyflow relationship, and a concrete next-step experiment plan.  
I did not re-run the code locally; this review is based on:

1. the public GitHub repository and documentation;
2. your reported Gate outputs;
3. the current project convention that Phases 1–7 are complete, R1 passed, and R2 passed;
4. the requirement that real LLM / multi-dataset / candidate-sensitivity experiments are still needed before paper-level claims.

**Evidence boundary**:

- Smoke/mock outputs are not paper evidence.
- MockLLM outputs are not real LLM evidence.
- R1 is pilot / non-paper evidence.
- R2 is full single-dataset paper-candidate analysis, but MockLLM paths remain diagnostic.
- Paper-level main results require at least R2-real-LLM and broader dataset validation.

---

## 1. Project One-Sentence Summary / 项目一句话定位

### 中文

TRUCE-Rec 是一个面向 **uncertainty-aware generative LLM4Rec** 的研究级代码库，核心问题不是“LLM 能不能推荐”，而是：

> 当 LLM 以生成 item title 的方式推荐时，它的 confidence 是否真正反映正确性、grounding validity、用户效用，而不是 popularity、familiarity、exposure bias 或 echo-chamber reinforcement？

### English

TRUCE-Rec is a research-grade codebase for **uncertainty-aware generative LLM4Rec**.  
The core question is not merely whether an LLM can recommend, but:

> When an LLM recommends by generating an item title, does its confidence reflect correctness, grounding validity, and user utility, rather than popularity, familiarity, exposure bias, or echo-chamber reinforcement?

---

## 2. Current Repository Status / 当前仓库状态

### 中文

GitHub README 当前显示：

- 仓库身份：`https://github.com/appleweiping/TRUCE-Rec.git`
- active branch: `main`
- active implementation package: `src/llm4rec/`
- historical / older Storyflow references are no longer the active Phase 6/7 package contract.
- 已实现：dataset preprocessing、baseline、LLM mock/interface、unified evaluator、ranking/validity/confidence/calibration/beyond-accuracy metrics、Phase 6 OursMethod、Phase 7 paper support and real-experiment templates。
- 未完成：approved paper-result experiment suite、OursMethod effectiveness claim、new real API run、HF download、real LoRA/QLoRA training、final paper conclusions。

### English

The README currently states that:

- repository identity: `https://github.com/appleweiping/TRUCE-Rec.git`;
- active branch: `main`;
- active implementation package: `src/llm4rec/`;
- older Storyflow references are no longer the active Phase 6/7 package contract;
- implemented components include preprocessing, baselines, LLM mock/interface, unified evaluator, ranking/validity/confidence/calibration/beyond-accuracy metrics, Phase 6 OursMethod, and Phase 7 paper support / real-experiment templates;
- not completed: approved paper-result suite, effectiveness claims for OursMethod, new real API run, HF download, real LoRA/QLoRA training, and final paper conclusions.

---

## 3. Research Idea / 研究 Idea

### 中文

当前研究 idea 已经从泛泛的 LLM4Rec 变成了清晰的方向：

> **Uncertainty-aware Generative Recommendation**  
> 用户历史 → LLM 生成推荐 title → title ground 到 catalog item → 评估 correctness / uncertainty / grounding / popularity / long-tail / echo risk。

关键不是“让 LLM 输出一个 confidence score”，而是建立一个推荐系统特有的不确定性观察和决策框架：

1. **LLM 生成正确 item title 时，它是否真的有信心？**
2. **LLM 错时，是低置信错误，还是高置信 hallucination？**
3. **high-confidence 是否偏向热门 item？**
4. **长尾 item 是否即使正确也 under-confident？**
5. **high-confidence 是否强化历史偏好，导致 echo chamber？**
6. **uncertainty 是否能用于 abstention、fallback、rerank、过滤 noisy pseudo-label？**

### English

The research idea has been sharpened into:

> **Uncertainty-aware Generative Recommendation**  
> user history → LLM generates a recommendation title → title is grounded to catalog item → evaluate correctness / uncertainty / grounding / popularity / long-tail / echo risk.

The contribution is not simply asking the LLM for a confidence score. The project defines recommendation-specific uncertainty observation and decision logic:

1. Is the LLM confident when it generates a correct item title?
2. When it is wrong, is it uncertain or confidently hallucinating?
3. Does high confidence concentrate on popular items?
4. Are correct tail recommendations under-confident?
5. Does high confidence reinforce past user preferences and echo chambers?
6. Can uncertainty guide abstention, fallback, reranking, or noisy pseudo-label filtering?

---

## 4. Why This Is Not Generic Calibration / 为什么这不是普通 calibration

### 中文

Bryan Hooi 组的 *Can LLMs Express Their Uncertainty?* 是启发而不是要复制。该工作关注 black-box LLM confidence elicitation：prompting、multi-sampling、aggregation/consistency，并评估 calibration 与 failure prediction。TRUCE-Rec 的不同点是将这些思想迁移到 **生成式推荐**：

| Generic LLM calibration | TRUCE-Rec adaptation |
|---|---|
| answer confidence | generated item title confidence |
| answer correctness | generated title grounding + next-item hit |
| self-consistency | multi-sample grounded-item consistency |
| failure prediction | hallucination / invalid title / wrong grounded item |
| distractor-normalized confidence | candidate-normalized item confidence |
| overconfidence | high-confidence wrong recommendation |
| task difficulty | user-history ambiguity / long-tail uncertainty |

### English

Bryan Hooi’s group’s *Can LLMs Express Their Uncertainty?* is an inspiration, not a template to copy. That work studies black-box LLM confidence elicitation through prompting, multi-sampling, aggregation/consistency, calibration, and failure prediction. TRUCE-Rec adapts these ideas into **generative recommendation**:

| Generic LLM calibration | TRUCE-Rec adaptation |
|---|---|
| answer confidence | generated item title confidence |
| answer correctness | generated title grounding + next-item hit |
| self-consistency | multi-sample grounded-item consistency |
| failure prediction | hallucination / invalid title / wrong grounded item |
| distractor-normalized confidence | candidate-normalized item confidence |
| overconfidence | high-confidence wrong recommendation |
| task difficulty | user-history ambiguity / long-tail uncertainty |

---

## 5. Storyflow and `llm4rec` Relationship / Storyflow 与 `llm4rec` 的关系

### 中文

项目中存在两个层次：

1. **`src/llm4rec/`**  
   当前正式实验和 Phase 6/7 的 active implementation package。所有正式 runner、baseline、method、metrics、evaluation、artifact contract 都应该走这一层。

2. **`src/storyflow/`**  
   历史概念层和早期 observation / diagnostic / confidence / grounding / provider / triage / server scaffold。它记录了项目从 “Storyflow observation” 演化到 “TRUCE-Rec uncertainty-aware generative recommendation” 的过程。  
   它仍然重要，因为它承载了早期思路、diagnostic tooling、server/Qwen/DeepSeek planning 与数据准备经验；但**不是当前正式实验的统一接口层**。

### English

The project has two layers:

1. **`src/llm4rec/`**  
   The active implementation package for formal experiments and Phase 6/7. All official runners, baselines, methods, metrics, evaluation, and artifacts should go through this layer.

2. **`src/storyflow/`**  
   A historical / conceptual / diagnostic layer containing earlier observation, confidence, grounding, provider, triage, server, and simulation scaffolds.  
   It remains valuable as design history and diagnostic tooling, but it is **not the active formal experiment contract**.

---

## 6. File-Level Architecture / 文件级架构

### 6.1 `src/llm4rec/data/`

| File / 文件 | Function / 功能 | Review note / 审阅意见 |
|---|---|---|
| `base.py` | Canonical interaction, item, user-example contracts | 正式实验数据接口基础 |
| `registry.py` | Dataset registration / lookup | 支持多数据集扩展 |
| `preprocess.py` | Preprocessing wrapper | 生成 train/valid/test 和 artifacts |
| `splits.py` | leave-one-out / temporal split | 必须保持无未来泄漏 |
| `candidates.py` | full/sampled/file/retriever candidates | R2/R3 关键协议文件 |
| `text_fields.py` | item title / metadata text policy | 控制 prompt 和 BM25 输入 |
| `sequential.py` | sequence examples and item index mapping | 支持 sequential baseline |

### 6.2 `src/llm4rec/rankers/`

| File | Function | Review note |
|---|---|---|
| `base.py` | Shared ranker contract | 所有 baseline 和 method 输出统一 schema |
| `random.py` | deterministic random baseline | sanity baseline |
| `popularity.py` | train-only popularity baseline | 重要 baseline |
| `bm25.py` | text retrieval/ranking baseline | Ours fallback 的核心对照 |
| `mf.py` | lightweight MF baseline | 目前是 smoke-capable，需加强或标注 |
| `sequential.py` | sequential_last_item / markov / sasrec scaffold | R2 中 sequential_markov 表现强，应认真对照 |
| `llm_generative.py` | LLM generated-title baseline | real LLM subgate 的关键 |
| `llm_reranker.py` | LLM candidate rerank baseline | real LLM 对照 |

### 6.3 `src/llm4rec/llm/`

| File | Function | Review note |
|---|---|---|
| `base.py` | provider request/response contract | 保存 text, usage, latency, metadata |
| `mock_provider.py` | deterministic tests | 不能作为 paper evidence |
| `openai_provider.py` | OpenAI-compatible API interface | R2-real-LLM subgate 使用 |
| `hf_provider.py` | local HF scaffold | R3/R4 可扩展 |
| `cost_tracker.py` | token/cost/latency | 论文 efficiency 必备 |
| `response_cache.py` | response caching | API 成本控制和复现实验关键 |

### 6.4 `src/llm4rec/prompts/`

| File | Function | Review note |
|---|---|---|
| `base.py` | prompt interface | 模板版本化 |
| `templates.py` | generative / rerank / confidence / verification prompts | 必须保证 target exclusion |
| `builder.py` | prompt construction with metadata/hash | leakage audit 核心 |
| `parsers.py` | robust JSON parsing | parse_success 必须进入 metrics |

### 6.5 `src/llm4rec/grounding/` and `generators/`

| File | Function | Review note |
|---|---|---|
| `grounding/title.py` | exact/normalized/token-overlap title grounding | 生成式推荐能否评估的核心 |
| `generators/parser.py` | generated structured-output parser | 需保留 raw output |
| `generators/constrained.py` | constrained generation scaffold | 可用于 future candidate-constrained method |
| `generators/explanation.py` | explanation scaffold | 目前不是主贡献 |

### 6.6 `src/llm4rec/methods/`

| File | Function | Review note |
|---|---|---|
| `ours_method.py` | Calibrated Uncertainty-Guided Generative Recommendation | Phase 6 主方法集成 |
| `uncertainty_policy.py` | accept / fallback / abstain / rerank policy | 核心 method logic |
| `fallback.py` | BM25 / popularity / sequential fallback | 需防止退化为 fallback-only |
| `ablation.py` | config-driven ablation | 论文消融基础 |

### 6.7 `src/llm4rec/metrics/`

| File | Function | Review note |
|---|---|---|
| `ranking.py` | Recall/NDCG/HitRate/MRR | 主准确率指标 |
| `validity.py` | validity/hallucination/candidate adherence | 生成式推荐必备 |
| `confidence.py` | confidence diagnostics | calibration story |
| `calibration.py` | ECE/Brier/reliability | uncertainty 核心 |
| `coverage.py` | item/catalog coverage | beyond-accuracy |
| `diversity.py` | category/token diversity | echo-risk proxy |
| `novelty.py` | train-popularity novelty | 长尾/流行度 bias |
| `long_tail.py` | head/mid/tail metrics | 论文重点 |
| `cold_start.py` | user history slices | sparsity analysis |
| `slicing.py` | per-domain/user/item slices | table export |
| `efficiency.py` | latency/token/cost/cache | API / real LLM 必备 |

### 6.8 `src/llm4rec/evaluation/`

| File | Function | Review note |
|---|---|---|
| `evaluator.py` | shared evaluator | 所有方法必须共用 |
| `prediction_schema.py` | schema validation | artifact integrity |
| `export.py` | JSON/CSV metrics export | run-level output |
| `aggregation.py` | multi-seed aggregation | paper table |
| `significance.py` | paired bootstrap / test scaffold | 不能对 smoke 数据 overclaim |
| `slices.py` | sliced metrics | long-tail/cold-start |
| `table_export.py` | CSV/Markdown/LaTeX tables | paper-ready artifacts |

### 6.9 `src/llm4rec/experiments/`

| File | Function | Review note |
|---|---|---|
| `config.py` | YAML config loading/resolution | experiment reproducibility |
| `runner.py` | unified run_all pipeline | 主入口 |
| `registry.py` | method/dataset/runner registry | extensibility |
| `seeding.py` | deterministic seeds | multi-seed correctness |
| `logging.py` | run logs | artifact audit |

### 6.10 `src/llm4rec/trainers/` and `models/`

| File | Function | Review note |
|---|---|---|
| `trainers/base.py` | trainer contract | train/eval/checkpoint |
| `trainers/sequential.py` | deterministic sequential training | current sequential baseline |
| `trainers/traditional.py` | traditional trainer | MF/traditional hooks |
| `trainers/lora.py` | LoRA dry-run scaffold | no real training yet |
| `trainers/checkpointing.py` | checkpoint manifest | server experiment readiness |
| `models/sequential.py` | sequential model interface | true SASRec not yet implemented |

### 6.11 `src/storyflow/`

| Directory / File | Role / 作用 |
|---|---|
| `analysis/` | early observation-analysis tooling |
| `baselines/` | older baseline observation utilities |
| `confidence/` | CURE/TRUCE confidence feature scaffolds |
| `data/` | legacy data preparation |
| `generation/` | legacy prompts / generation |
| `grounding/` | early title grounding |
| `metrics/` | older calibration/popularity metrics |
| `providers/` | older API/mock provider code |
| `server/` | Qwen/server run plan layer |
| `simulation/` | echo simulation / diagnostic tooling |
| `training/` | Qwen LoRA planning |
| `triage/` | data triage / risk summaries |
| `observation.py` | original observation flow |
| `observation_parsing.py` | original response parser |
| `schemas.py` | early schemas |

**Review decision**: keep Storyflow as conceptual/provenance and diagnostic layer; do not use it as the official paper-run interface unless explicitly wrapped through `src/llm4rec`.

---

## 7. Config and Experiment Files / 配置与实验文件

### 中文

当前 `configs/experiments/` 已经包含：

- smoke configs: `smoke_*`
- R1 pilot: `pilot_movielens_1m_r1.yaml`
- R2 full single-dataset: `r2_movielens_1m_full.yaml`
- R2 real-LLM follow-up: `r2_movielens_1m_real_llm_api_followup.yaml`
- real templates: `real_*_template.yaml`

这些文件表明项目已经具备：

1. smoke pipeline；
2. one-dataset pilot；
3. full single-dataset paper-candidate experiment；
4. API follow-up config；
5. multi-seed / ablation / observation template。

### English

`configs/experiments/` includes:

- smoke configs: `smoke_*`;
- R1 pilot: `pilot_movielens_1m_r1.yaml`;
- R2 full single-dataset: `r2_movielens_1m_full.yaml`;
- R2 real-LLM follow-up: `r2_movielens_1m_real_llm_api_followup.yaml`;
- real templates: `real_*_template.yaml`.

This indicates readiness for:

1. smoke pipeline;
2. one-dataset pilot;
3. full single-dataset paper-candidate experiment;
4. API follow-up;
5. multi-seed / ablation / observation templates.

---

## 8. Contributions / 贡献点

### Contribution 1: Recommendation-specific uncertainty observation  
### 贡献 1：推荐特有的不确定性观察框架

**中文**:  
不是泛泛地问 LLM “are you confident”，而是在生成推荐 title 后进行 catalog grounding，并分析 confidence 与 correctness、hallucination、popularity、tail、history similarity、diversity、echo-risk 的关系。

**English**:  
The project does not merely ask “are you confident?” It grounds generated recommendation titles to catalog items, then analyzes uncertainty against correctness, hallucination, popularity, tail behavior, history similarity, diversity, and echo-risk.

### Contribution 2: Calibrated uncertainty-guided decision policy  
### 贡献 2：校准不确定性驱动的决策策略

**中文**:  
OursMethod 将 generative recommendation 的输出分为 accept、fallback、abstain、rerank；并引入 grounding score、confidence、candidate-normalized confidence、popularity bucket、history similarity 等信号。

**English**:  
OursMethod routes generated recommendations into accept, fallback, abstain, or rerank decisions using grounding score, confidence, candidate-normalized confidence, popularity bucket, and history similarity.

### Contribution 3: Echo-risk and popularity-aware confidence analysis  
### 贡献 3：面向流行度与 echo risk 的置信度分析

**中文**:  
核心分析不是单纯 ECE，而是：高置信是否集中在 head item？tail item 是否 under-confident？high-confidence 是否降低 novelty/diversity、强化历史偏好？

**English**:  
The key analysis goes beyond ECE: does high confidence concentrate on head items? Are correct tail items under-confident? Does high confidence reduce novelty/diversity and reinforce past preferences?

### Contribution 4: Reproducible artifact pipeline  
### 贡献 4：可复现实验产物链

**中文**:  
每个 run 保存 resolved config、environment、logs、predictions、metrics、cost_latency、tables；这对 artifact review 是关键。

**English**:  
Every run saves resolved config, environment, logs, predictions, metrics, cost/latency, and tables; this is essential for artifact review.

---

## 9. Current Reviewer Verdict / 当前审阅结论

### 中文

基于当前公开仓库和你提供的 R1/R2 报告：

- **工程与实验框架**：通过。
- **R1 pilot**：通过。
- **R2 full single-dataset**：通过，可作为 paper-candidate analysis。
- **真实 LLM 证据**：尚未完成。
- **多数据集主表**：尚未完成。
- **最终论文 claim**：不能开始写结论，只能写 protocol 和 TBD 表格。

当前最关键风险：

1. R2 中 LLM/Ours generative 部分仍是 MockLLM。
2. Ours full 与 fallback-only / BM25 在 R1/R2 中可能高度重合，需真实 LLM subgate 判断方法是否有独立信号。
3. candidate size=100 是 sampled-ranking evidence；最终论文需要 candidate sensitivity 或 full-ranking 对照。
4. minimal MF / sasrec_interface 需要明确标注，若目标顶会，需要更强 sequential / graph baseline。
5. Amazon / multi-domain 数据尚未进入 paper-candidate 主实验。

### English

Based on the public repository and your R1/R2 reports:

- **Engineering and experiment framework**: passed.
- **R1 pilot**: passed.
- **R2 full single-dataset**: passed, usable as paper-candidate analysis.
- **Real LLM evidence**: pending.
- **Multi-dataset main table**: pending.
- **Final paper claims**: not yet allowed.

Key risks:

1. LLM/Ours generative paths in R2 still rely on MockLLM.
2. Ours full may collapse to fallback/BM25; real LLM subgate is needed.
3. candidate size=100 is sampled-ranking evidence; final paper needs candidate sensitivity or full-ranking comparison.
4. minimal MF / sasrec_interface must be clearly labeled; stronger sequential/graph baselines may be required.
5. Amazon / multi-domain experiments are not yet paper-candidate results.

---

## 10. Experiment Roadmap / 实验路线图

### Stage R2.5: Real-LLM Subgate on MovieLens  
### 阶段 R2.5：MovieLens 小规模真实 LLM 子门

**Goal / 目标**: Replace MockLLM with real LLM on a small approved subset.  
**Why / 原因**: R2 proves protocol, not real LLM uncertainty.  

**Dataset**: MovieLens 1M R2 subset  
**Sample size**: 200–500 test examples  
**Candidate size**: 100 initially, then 500 if cost allows  
**Seeds**: 13 initially  
**Methods**:

- popularity
- BM25
- sequential_markov
- llm_generative_real
- llm_rerank_real
- llm_confidence_observation_real
- ours_uncertainty_guided_real
- ours_fallback_only
- ours_no_uncertainty
- ours_no_grounding

**Metrics**:

- Recall@10 / NDCG@10 / MRR@10
- validity / hallucination
- parse success / grounding success
- confidence mean
- ECE / Brier
- risk-coverage
- confidence-by-popularity bucket
- latency/token/cost
- fallback/accept/abstain/rerank ratio

**Pass criteria**:

- parse success ≥ 95%
- grounding success ≥ 90%
- target leakage = 0
- raw outputs saved
- cost/latency saved
- Ours has non-trivial non-fallback decisions, or method is marked observation-only.

---

### Stage R2.6: Candidate Sensitivity  
### 阶段 R2.6：候选集敏感性

**Goal / 目标**: Verify whether conclusions hold under candidate size changes.  
**Candidate sizes**: 50, 100, 500, optionally full ranking for non-LLM methods.  

**Methods**:

- popularity
- BM25
- sequential_markov
- Ours full
- Ours fallback-only
- llm_rerank_real if API budget allows

**Report**:

- ranking metric sensitivity
- grounding and hallucination changes
- confidence shifts
- popularity bucket shifts
- cost scaling

---

### Stage R3: Full Single-Dataset Real LLM  
### 阶段 R3：完整单数据集真实 LLM

**Goal / 目标**: Establish one paper-quality dataset result with real LLM.  
**Dataset**: MovieLens 1M full R2 dataset  
**Seeds**: [13, 21, 42]  
**Candidate size**: 100 + sensitivity  
**Methods**: all baselines + Ours + ablations  
**Output**: paper-candidate tables, not final conclusion yet.

---

### Stage R4: Multi-Dataset / Multi-Domain  
### 阶段 R4：多数据集 / 多域

**Datasets**:

- MovieLens 1M
- Amazon Beauty
- Amazon Health_and_Personal_Care
- Amazon Video Games or Digital Music, if metadata quality is acceptable

**Goal**:

- show robustness across domains;
- test whether confidence-popularity-tail observations generalize;
- identify domains where grounding is hard.

---

### Stage R5: Strong Baseline Upgrade  
### 阶段 R5：强 baseline 补强

If paper target is SIGIR/KDD/WWW/RecSys main track, strengthen baselines:

- true BPR-MF or NeuMF
- SASRec or GRU4Rec
- LightGCN if feasible
- sentence-transformer dense retrieval
- real API LLM generative / rerank
- local Qwen / Llama interface if server available

---

### Stage R6: Final Paper Freeze  
### 阶段 R6：论文结果冻结

Freeze:

- commit hash
- configs
- dataset manifests
- run artifacts
- paper tables
- case studies
- failure cases
- limitations

---

## 11. Recommended Codex Prompt: R2-Real-LLM Subgate  
## 11. 给 Codex 的下一步 Prompt：R2 真实 LLM 子门

```text
R2 full single-dataset experiment has PASSed, but it used MockLLM for LLM/Ours generative paths. Now start R2-real-LLM subgate on a small approved MovieLens subset.

You are not adding new research functionality. You are validating whether the existing uncertainty-aware generative recommendation pipeline works with a real LLM provider.

Read:

- AGENTS.md
- README.md
- docs/RESEARCH_IDEA.md
- docs/experiment_protocol.md
- docs/real_experiment_matrix.md
- docs/server_runbook.md
- docs/leakage_fairness_checklist.md
- docs/result_artifact_checklist.md
- docs/r2_movielens_1m_protocol.md
- configs/experiments/r2_movielens_1m_full.yaml
- configs/experiments/r2_movielens_1m_real_llm_api_followup.yaml

Goal:

Run a small real-LLM subgate on MovieLens, replacing MockLLM with an approved OpenAI-compatible provider, while keeping all safety, leakage, artifact, and cost controls.

Hard rules:

- Do not change OursMethod core mechanism.
- Do not run full dataset real API first.
- Do not run multi-dataset experiments.
- Do not download HF models.
- Do not run LoRA.
- Do not write paper conclusions.
- Do not fabricate results.
- If API key or provider config is missing, stop with BLOCKER and list what user must provide.

Step 1: Preflight

Create or update:

configs/experiments/r2_movielens_1m_real_llm_subgate.yaml

Use:

- dataset: MovieLens 1M R2 processed data
- subset_size: 200 or configurable
- candidate protocol: sampled, size 100, include_target true, seed 13
- seeds: [13]
- provider: openai_compatible
- allow_api_calls: false by default
- requires_confirm: true by default
- cache enabled
- resume enabled
- raw output saving enabled
- max requests / concurrency explicitly set
- cost limit field explicitly set

Then run:

python scripts/validate_experiment_ready.py --config configs/experiments/r2_movielens_1m_real_llm_subgate.yaml
python scripts/list_required_artifacts.py --config configs/experiments/r2_movielens_1m_real_llm_subgate.yaml

If validation fails, fix config only.

Step 2: User-confirmation mode

Do not execute API unless config is explicitly changed to:

allow_api_calls: true
requires_confirm: false

and required environment variables exist.

If approved, run:

python scripts/run_all.py --config configs/experiments/r2_movielens_1m_real_llm_subgate.yaml

Step 3: Methods

Run:

- popularity
- bm25
- sequential_markov
- llm_generative_real
- llm_rerank_real
- llm_confidence_observation_real
- ours_uncertainty_guided_real
- ours_fallback_only
- ours_ablation_no_uncertainty
- ours_ablation_no_grounding

Step 4: Artifact checks

Each run must contain:

- resolved_config.yaml
- environment.json
- logs.txt
- predictions.jsonl
- metrics.json
- metrics.csv
- cost_latency.json
- raw LLM outputs
- artifacts/

Step 5: Required analysis

Report:

- parse success
- grounding success
- hallucination rate
- Recall@10 / NDCG@10 / MRR@10
- confidence mean
- ECE / Brier
- high-confidence wrong count
- low-confidence correct count
- risk-coverage artifacts
- confidence-by-popularity artifacts
- fallback/accept/abstain/rerank ratio
- cost and latency
- leakage audit

Step 6: Regression

Run:

python scripts/run_all.py --config configs/experiments/smoke_phase6_all.yaml
python scripts/run_all.py --config configs/experiments/smoke_phase5_all.yaml
python -m pytest
git diff --check

Output verdict:

- PASS: real LLM subgate trustworthy enough to scale
- PASS WITH MINOR FIXES
- MAJOR FIXES REQUIRED
- BLOCKER

If PASS, next action:

Run candidate sensitivity and scale to full single-dataset real LLM experiment.
```

---

## 12. Recommended Codex Prompt: Candidate Sensitivity  
## 12. 给 Codex 的候选集敏感性 Prompt

```text
R2 and R2-real-LLM subgate have passed. Now run candidate sensitivity for MovieLens.

Do not change model code.
Do not change OursMethod.
Do not run multi-dataset experiments.
Do not write paper conclusions.

Create:

configs/experiments/r2_movielens_candidate_sensitivity.yaml

Candidate sizes:

- 50
- 100
- 500
- full ranking for non-LLM methods if feasible

Methods:

- popularity
- bm25
- sequential_markov
- ours_uncertainty_guided
- ours_fallback_only
- llm_rerank_real if API budget allows

Metrics:

- Recall@10
- NDCG@10
- MRR@10
- validity
- hallucination
- grounding success
- confidence/ECE
- coverage
- novelty
- long-tail
- cost/latency

Run:

python scripts/validate_experiment_ready.py --config configs/experiments/r2_movielens_candidate_sensitivity.yaml
python scripts/run_all.py --config configs/experiments/r2_movielens_candidate_sensitivity.yaml
python scripts/export_tables.py --input outputs/runs --output outputs/tables
python scripts/aggregate_runs.py --input outputs/runs --output outputs/tables
python -m pytest
git diff --check

Output:

- candidate-size table
- sensitivity summary
- whether sampled-ranking conclusions are stable
- recommended candidate protocol for paper experiments
```

---

## 13. Recommended Codex Prompt: Full Multi-Dataset Main Experiment  
## 13. 给 Codex 的多数据集主实验 Prompt

```text
R2 full single-dataset, R2-real-LLM subgate, and candidate sensitivity have passed. Now prepare full multi-dataset main experiment plan.

Do not run immediately. First output a costed execution plan.

Datasets:

- MovieLens 1M
- Amazon Beauty
- Amazon Health_and_Personal_Care
- one additional Amazon domain only if data quality gates pass

Methods:

- popularity
- bm25
- mf or strengthened MF
- sequential_markov
- stronger sequential baseline if implemented
- llm_generative_real
- llm_rerank_real
- llm_confidence_observation_real
- ours_uncertainty_guided
- ours_fallback_only
- ours_no_uncertainty
- ours_no_grounding
- ours_no_popularity_adjustment
- ours_no_echo_guard

Metrics:

- Recall@K / NDCG@K / MRR@K
- validity / hallucination / grounding
- confidence / ECE / Brier / risk-coverage
- popularity bucket
- long-tail
- diversity
- novelty
- cost/latency
- significance / multi-seed aggregation

Output before running:

- configs to create
- estimated number of API calls
- estimated token cost
- runtime estimate
- table plan
- artifact plan
- risk checklist
- stopping criteria

Do not execute until user approves.
```

---

## 14. Paper Story After R2 / R2 后论文主线

### 中文

现在论文主线不应该写成：

> 我们提出一个 LLM reranker，提升 NDCG。

这会弱，而且 R2 目前显示 Ours 在 MockLLM 下基本走 fallback/BM25 路径。

更好的顶会故事是：

> 生成式 LLM 推荐的 confidence 并不是普通 QA calibration 问题。它受 catalog grounding、popularity、tail exposure、history similarity 和 fallback decision 共同影响。TRUCE-Rec 提出一套 generative recommendation uncertainty observation + calibrated decision policy，并系统评估什么时候 LLM confidence 有用、什么时候会产生高置信错误和 echo-risk。

### English

The paper should not be framed as:

> We propose an LLM reranker that improves NDCG.

That is weak, and current R2 indicates that Ours under MockLLM largely follows BM25/fallback behavior.

A stronger top-conference story is:

> Confidence in generative LLM recommendation is not generic QA calibration. It is confounded by catalog grounding, popularity, tail exposure, history similarity, and fallback decisions. TRUCE-Rec introduces a generative recommendation uncertainty observation framework and calibrated decision policy, systematically evaluating when LLM confidence is useful and when it causes high-confidence errors or echo-risk.

---

## 15. Final Recommendation / 最终建议

### 中文

当前项目没问题，可以继续。  
但下一步不要直接跑多域主实验。正确顺序是：

1. **R2-real-LLM subgate on MovieLens small subset**
2. **Candidate sensitivity**
3. **Full single-dataset real LLM experiment**
4. **Multi-dataset / Amazon domains**
5. **Stronger baselines**
6. **Final paper tables**

### English

The project is structurally sound.  
Do not jump directly to multi-dataset main experiments. The correct next order is:

1. **R2-real-LLM subgate on MovieLens small subset**
2. **Candidate sensitivity**
3. **Full single-dataset real LLM experiment**
4. **Multi-dataset / Amazon domains**
5. **Stronger baselines**
6. **Final paper tables**

---

## 16. Sources Consulted / 参考来源

- GitHub repository: <https://github.com/appleweiping/TRUCE-Rec>
- README / repository status
- `docs/RESEARCH_IDEA.md`
- `src/llm4rec/` tree
- `src/storyflow/` tree
- `configs/experiments/` tree
- User-provided Gate R1/R2 reports
- Xiong et al., “Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs”
- Negative sampling and sampled metric evaluation literature
- Generative recommendation survey literature
