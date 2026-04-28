# AGENTS.md

## Project identity

This repository implements the research project **Storyflow / TRUCE-Rec**.

The GitHub repository is:

- `https://github.com/appleweiping/uncertainty-llm4rec.git`

The active local project directory is:

- `D:\Research\TRUCE-Rec`

The active branch is:

- `main`

The current clean main branch started from:

- `2e0cfff Start new main`

Old branches exist only for archival/reference purposes and must not influence this new project:

- `origin/archive/old-main`
- `origin/archive/local-before-new-main`
- `origin/codex/week4-confidence-repair`
- `origin/codex/apr12-preserve-local`
- `origin/codex-day4-calibration-rerank`

Do **not** use the old local directory:

- `D:\Research\Uncertainty-LLM4Rec`

Do **not** switch to old branches unless the user explicitly instructs it for a narrowly scoped comparison. The new project must be developed only on `D:\Research\TRUCE-Rec` and `main`.

## Core research goal

Storyflow / TRUCE-Rec is a top-tier research codebase for **uncertainty-aware LLM-based generative recommendation**.

The central research question is:

> When an LLM recommends an item by generating its title, does the model know whether the recommendation is correct, and does its confidence reflect true user preference or merely popularity, familiarity, exposure bias, grounding ease, or training noise?

The central thesis is:

> In generative recommendation, confidence is not just a passive reliability score. It is an exposure-shaping variable that can change what users see, what they click, and what the recommender learns in the future. Therefore, LLM recommender confidence must be calibrated to exposure-counterfactual user utility, not merely offline correctness.

The main conceptual specification is in:

- `Storyflow.md`

Treat `Storyflow.md` as the source of truth.

## What this project is not

This is not a toy project.

This is not a generic top-k recommendation project.

This is not a simple uncertainty add-on.

This is not merely prompt engineering.

This is not just “run LLM API and draw a chart.”

This is not allowed to collapse into an ordinary ranking-only recommender system.

The core task is **title-level generative recommendation**:

1. The model receives user interaction history as item titles and metadata.
2. The model generates item title(s).
3. The generated title must be grounded to a catalog item.
4. The system evaluates correctness, confidence, calibration, popularity coupling, and echo risk.
5. The final framework must improve calibrated generative recommendation using Qwen3-8B + LoRA or comparable small-model training on server.

## Non-negotiable scientific rules

1. Never fabricate experimental results.
2. Never invent tables, plots, metrics, numbers, or conclusions.
3. Never write claims like “our method improves performance” unless produced by actual code and actual logs.
4. Synthetic demos must be explicitly labeled as synthetic.
5. Toy/synthetic data is allowed only for scaffolding and tests. After the scaffold works, transition to real datasets immediately.
6. Do not stay in toy mode.
7. Do not silently skip dataset download or full-data preparation. If a dataset source requires login, license acceptance, or manual intervention, create a clear Chinese report explaining exactly what the user must do.
8. Do not leak API keys.
9. Never commit `.env`, API keys, access tokens, raw API responses containing sensitive content, or large raw datasets.
10. Never commit large PDFs or reference archives such as `recprefer.zip`.
11. Always preserve reproducibility: configs, seeds, command logs, output paths, and exact code versions.
12. Always distinguish local CPU/GPU work from server-only work.
13. Codex cannot access the remote server. If server execution is needed, write scripts and runbooks, but do not claim server results were run.
14. Every finished coding task must end with tests when possible, a git commit, a push to `origin/main`, and a detailed Chinese local report.

## Required git workflow

Before making changes:

1. Check current directory.
2. Verify the repository is `D:\Research\TRUCE-Rec`.
3. Verify current branch is `main`.
4. Run `git status`.
5. If the branch is not `main`, stop and report in Chinese.
6. If there are unexpected uncommitted changes, inspect and report before editing.
7. Do not check out archive branches.
8. Do not merge old branches.
9. Do not copy old branch code unless explicitly requested and scientifically justified.

After making changes:

1. Run relevant tests.
2. Update `README.md` if the user-facing workflow, dependencies, commands, or project status changed.
3. Update docs if architecture, experiment protocol, or assumptions changed.
4. Write a detailed Chinese report under `local_reports/`.
5. Ensure `local_reports/` is gitignored.
6. Run `git status`.
7. Commit changes with a meaningful message.
8. Push to `origin/main`.
9. In the final response, include:
   - files changed;
   - commands run;
   - tests passed/failed/not run;
   - commit hash;
   - push status;
   - remaining TODOs;
   - any user action needed.

## Chinese local report requirement

Every substantial Codex task must create one Chinese report file under:

- `local_reports/YYYYMMDD-HHMMSS-task-name.md`

This report is for internal research management only and must not be committed to GitHub.

The report must include:

1. 本轮目标
2. 实际完成内容
3. 修改文件清单
4. 新增/修改的命令
5. 数据下载/处理状态
6. API 调用状态与缓存状态
7. 实验是否为 synthetic / pilot / full
8. 本地可跑内容
9. 服务器才可跑内容
10. 测试命令与测试结果
11. Git commit hash
12. 是否已 push 到 `origin/main`
13. 和 `Storyflow.md` 的对应关系
14. 目前风险点
15. 下一步建议
16. 需要用户帮助的事项

`local_reports/` must be added to `.gitignore`.

## README requirement

`README.md` must be updated regularly.

Update `README.md` whenever:

- a new module is added;
- a command changes;
- a dataset pipeline is added;
- an experiment script is added;
- an API provider adapter is added;
- training or evaluation workflow changes;
- server run instructions change.

The README should never contain fabricated results.

It can contain “implemented”, “planned”, “not yet run”, “synthetic demo only”, and “server-only” status markers.

## Data policy

The project must support real datasets, not only synthetic demos.

The data pipeline should support at least:

1. MovieLens 1M as a fast local sanity-check real dataset.
2. Amazon Reviews 2023 categories for full-scale generative recommendation.
3. Steam / games data if feasible.
4. Additional Amazon categories such as Beauty, Sports_and_Outdoors, Toys_and_Games, Video_Games, Books, CDs_and_Vinyl, Office_Products, etc., as configs allow.

Use publicly available official or widely accepted sources when possible.

Dataset code must support:

- downloading raw data;
- verifying files where possible;
- caching;
- resuming;
- processing;
- k-core filtering;
- interaction-count filtering;
- chronological splitting;
- leave-last-one split;
- leave-last-two validation/test split;
- rolling/iterative examples where each user can produce multiple training samples;
- history truncation;
- title cleaning;
- item metadata joining;
- popularity computation;
- head/mid/tail bucket assignment.

If a dataset cannot be downloaded automatically due to access restrictions, do not silently skip it. Create a Chinese report explaining:

- dataset name;
- attempted source;
- exact failure;
- required user action;
- expected target path;
- command to resume after the file is placed.

Raw datasets should be stored under `data/raw/` and should usually be gitignored.

Processed small fixtures for tests can be committed under `tests/fixtures/`.

## Reference material policy

The uploaded `recprefer.zip` contains NH/NR recommendation paper PDFs and is reference material.

It should be placed locally under something like:

- `references/recprefer.zip`

or extracted under:

- `references/recprefer/`

But large PDFs and zip files must not be committed.

Instead, commit only:

- `references/README.md`
- `docs/related_work/recprefer_index.md`
- `docs/related_work/baseline_notes.md`

The reference material should be used to identify:

- relevant baseline families;
- NH metrics: NDCG + Hit Ratio;
- NR metrics: NDCG + Recall;
- common preprocessing settings;
- minimal-change baselines useful for reviewer-proofing.

## Baseline policy

Observation must not only be run on one base LLM.

The project should eventually evaluate whether the confidence phenomena appear across:

1. Official large-model APIs:
   - DeepSeek
   - Qwen API
   - Kimi / Moonshot
   - GLM / Zhipu
2. Local or server-side small models:
   - Qwen3-8B
   - Qwen3-8B + LoRA
3. Recommendation baselines:
   - SASRec
   - BERT4Rec
   - GRU4Rec
   - LightGCN or other graph/ranking baselines where appropriate
4. Generative recommendation / LLM4Rec baselines where feasible:
   - P5-like instruction recommendation
   - TIGER / Semantic-ID style if feasible
   - BIGRec / grounding-style if feasible
   - uncertainty-aware baselines if reproducible

Do not implement every baseline at once. Add them in phases. But the architecture must not prevent full baseline coverage.

## API provider policy

Official API keys are available to the user, but Codex must not ask the user to paste keys into source code.

Use environment variables and `.env.example`.

Expected environment variables include:

- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`
- `MOONSHOT_API_KEY`
- `ZHIPUAI_API_KEY`

API adapters should be config-driven.

API calling must support:

- provider selection;
- model name selection;
- rate limit control;
- concurrency control;
- retries with exponential backoff;
- response caching;
- idempotent resume;
- JSONL input/output;
- partial run continuation;
- cost/token accounting if available;
- run manifests;
- clear separation of prompt, request, raw response, parsed prediction, and grounded prediction.

Do not call paid APIs in tests.

Do not commit API cache if it may contain sensitive data unless explicitly sanitized.

For speed, prefer:

- deduplicating identical prompts;
- batching where provider supports it;
- asynchronous/concurrent requests with safe rate limits;
- caching by prompt hash + provider + model + temperature + seed/config;
- incremental JSONL writes;
- resumable failed runs;
- small pilot subset before full run;
- stratified sampling by popularity/user length/category for pilots;
- full run after pilot validation.

Speed must not compromise the scientific design.

## Local versus server execution

Local machine is used for:

- repository editing;
- data downloading where feasible;
- preprocessing;
- synthetic tests;
- small real-data sanity checks;
- API-based observation;
- report generation;
- plotting from completed outputs.

Server is expected for:

- Qwen3-8B full inference if too heavy locally;
- Qwen3-8B + LoRA framework training;
- large-scale baselines;
- large Amazon categories;
- long-running experiments.

When server execution is needed, create:

- `scripts/server/`
- `configs/server/`
- `docs/server_runbook.md`

Codex must not claim server experiments were run unless the user provides logs or results.

## Training policy

The framework stage should use Qwen3-8B + LoRA or comparable small-model training.

Training scripts should follow robust HF/PEFT practices:

- config-driven model path;
- tokenizer handling;
- LoRA config;
- bf16/fp16 support;
- gradient accumulation;
- checkpointing;
- resume;
- eval per epoch or step;
- early stopping where appropriate;
- deterministic seeds where feasible;
- output manifest;
- clean separation of train/eval/generate.

Training should support:

- SFT baseline;
- confidence expression / RecBrier-style objective where feasible;
- risk-aware preference optimization where feasible;
- CURE/TRUCE scoring/reranking module;
- evaluation of calibration and recommendation metrics.

Do not start heavy training in a local Codex run unless explicitly requested.

## Core modules expected

Prefer a clean Python package structure:

- `src/storyflow/`
- `src/storyflow/data/`
- `src/storyflow/grounding/`
- `src/storyflow/generation/`
- `src/storyflow/providers/`
- `src/storyflow/confidence/`
- `src/storyflow/metrics/`
- `src/storyflow/analysis/`
- `src/storyflow/simulation/`
- `src/storyflow/triage/`
- `src/storyflow/models/`
- `src/storyflow/training/`
- `src/storyflow/baselines/`
- `src/storyflow/utils/`
- `configs/`
- `scripts/`
- `scripts/server/`
- `tests/`
- `docs/`
- `references/`

## Core implementation phases

Do not implement everything in one task.

Do not keep saying “continue” forever without closure.

Follow milestones:

### Phase 0: Governance and scaffold

- Verify repo, branch, remote.
- Create project structure.
- Create implementation plan.
- Create experiment protocol.
- Create local report system.
- Create `.gitignore`.
- Update README.
- Commit and push.

### Phase 1: Data and preprocessing

- Dataset downloader.
- Dataset manifest.
- MovieLens 1M real-data pipeline.
- Amazon Reviews 2023 downloader/processor configs.
- K-core and interaction-count filtering.
- Chronological and rolling split options.
- Popularity computation.
- Tests and docs.
- Commit and push.

### Phase 2: Generative observation pipeline

- Prompt templates.
- API provider adapters.
- Mock provider tests.
- JSONL generation format.
- Title grounding.
- Confidence extraction.
- Correctness labels.
- Pilot observation on small real data.
- Commit and push.

### Phase 3: Full observation

- API batch runner with cache and resume.
- Multi-provider observation.
- Local/server Qwen3-8B observation support.
- Baseline observation support.
- Reliability, ECE, Brier, CBU, WBC.
- Popularity-confidence coupling.
- Tail underconfidence.
- Wrong-high-confidence analysis.
- Commit and push.

### Phase 4: Framework

- CURE/TRUCE uncertainty features.
- Calibrator.
- Popularity residual / deconfounding.
- Exposure-aware scoring.
- Reranking.
- Risk-aware preference objective design.
- Qwen3-8B + LoRA training scripts.
- Server runbook.
- Commit and push.

### Phase 5: Echo simulation and data triage

- Confidence-induced exposure simulation.
- Multi-round feedback loop.
- Exposure Gini, tail coverage, category entropy.
- Uncertainty-guided data triage.
- Noise injection experiments.
- Commit and push.

### Phase 6: Full experimental suite and paper artifacts

- Full datasets.
- Full baselines.
- Full model runs.
- Final tables and plots from actual logs.
- Reproducibility package.
- Paper-ready analysis.
- Commit and push.

## Metrics required

Recommendation metrics:

- Recall@K
- NDCG@K
- Hit Ratio@K
- MRR@K where useful
- Coverage
- Tail coverage
- Head/mid/tail performance

Generative/title metrics:

- GroundHit
- Grounding confidence
- Grounding ambiguity
- Out-of-catalog rate
- Duplicate title rate
- Fuzzy/semantic match status

Confidence metrics:

- ECE
- Adaptive ECE if implemented
- Brier score
- CBU_tau: correct but uncertain
- WBC_tau: wrong but confident
- AURC / selective risk if implemented
- Reliability diagram data by popularity bucket

Popularity/echo metrics:

- Popularity-confidence slope
- Tail underconfidence gap
- Head/mid/tail confidence gap
- Exposure Gini
- Tail exposure share
- Category entropy
- Confidence drift across feedback rounds

Noise/triage metrics:

- Prune ratio
- Downweight ratio
- Kept hard-tail-positive ratio
- Noise detection precision/recall on synthetic noise
- Post-triage recommendation/calibration metrics

## Testing policy

Use pytest.

Tests should cover:

- schemas;
- data loading;
- k-core filtering;
- splitting;
- title normalization;
- grounding;
- correctness labels;
- calibration metrics;
- popularity buckets;
- provider parsing without paid API calls;
- cache/resume logic;
- synthetic observation pipeline;
- scoring/reranking;
- triage behavior;
- simulation determinism.

If tests cannot run due to missing dependencies, install or document the missing dependency and add a minimal test where possible.

## Final response policy

At the end of each Codex task, respond in Chinese with:

1. 已完成什么
2. 改了哪些文件
3. 运行了哪些命令
4. 测试结果
5. 是否更新 README
6. 是否写入 local report
7. git commit hash
8. 是否 push 到 origin/main
9. 下一步建议
10. 需要用户做什么

Do not use vague phrases like “should work.” Be concrete.

If something failed, say exactly what failed and why.