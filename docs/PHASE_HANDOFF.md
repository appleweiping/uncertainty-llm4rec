# Phase Handoff

Repository identity:

- Repository: TRUCE-Rec unless the user states otherwise.
- GitHub: `https://github.com/appleweiping/TRUCE-Rec.git`
- Historical remote alias in this checkout:
  `https://github.com/appleweiping/uncertainty-llm4rec.git`
- Local path: `D:\Research\TRUCE-Rec`
- Active branch: `main`

Completed local phases:

- Phase 1: reproducible skeleton and tiny smoke path.
- Phase 2: minimal traditional/text baselines.
- Phase 3: LLM baseline layer and uncertainty observation hooks.
- Phase 4: sequential/training interface layer.
- Phase 5: evaluation/export/aggregation support.
- Phase 6: minimal MockLLM OursMethod integration.
- Phase 7: paper support and real experiment planning.

Current gate:

- Gate R1: server deployment and independent four-domain experiment buildout.

## CRITICAL: Independence from Pony/TGL-Rec

TRUCE-Rec 和 Pony 共享 8 个外部 baseline 和数据 setting（同组论文），但方法/
framework 完全不同。Baseline 分数可复用，方法代码不可混用。
TRUCE-Rec 在服务器上独立部署，不依赖 Pony 的目录结构。

Current roadmap:

- `docs/PROJECT_MEMORY.md` is the durable future-agent memory. Read and update
  it when big direction, baseline policy, server commands, or current status
  changes.
- `docs/submission_roadmap.md` is the primary milestone document.
- `docs/server_execution_matrix.md` is the server command and artifact gate.
- `docs/top_conference_review_plan.md` is the top-conference reviewer defense
  checklist.

Evidence boundaries:

- Smoke outputs are not paper evidence.
- MockLLM outputs are not paper evidence.
- Pilot/API diagnostic outputs are not paper conclusions.
- External-project adapter smoke outputs are not paper evidence. The
  OpenP5 Amazon Beauty adapter smoke only proves that a `candidate_scores.csv`
  can be imported and evaluated by TRUCE.
- Ignored local diagnostics under `outputs/` or `data/processed/` are not paper
  evidence unless explicitly promoted by a later approved protocol.
- Formal paper results must come from approved real experiment configs,
  tracked code, saved configs, logs, raw outputs, predictions, and metrics.

Next allowed Gate R1 actions:

- Deploy TRUCE-Rec to server independently (git clone, venv, install).
- Prepare four-domain same-candidate data from raw Amazon Reviews 2023.
- Run base Qwen3-8B observation on TRUCE's own prepared data.
- Train TRUCE's own baselines (TALLRec, OpenP5, DEALRec, LC-Rec with
  Qwen3-8B + LoRA) independently.
- Run Ours and ablations on the same candidate rows.

Server handoff:

- Server repo path: `~/projects/TRUCE-Rec` (to be deployed).
- Server: pony-rec-gpu (125.71.97.70:15302), user ajifang, GPU RTX 4090.
- Use SSH Git remote: `git@github.com:appleweiping/TRUCE-Rec.git`.
- TRUCE-Rec is fully independent on the server. It does NOT share data or
  environments with ~/projects/pony-rec-rescue-shadow-v6 or any other project.

Next non-toy buildout:

- Convert and validate Week8 same-candidate tasks with strict target-in-candidate
  checks.
- Keep the Pony baseline manifest current as missing/pending evidence arrives.
- Run Ours and ablations on the same candidate rows.

Future-agent update rule:

- After a completed stage, update `docs/PROJECT_MEMORY.md`, this handoff,
  README, server command docs, and relevant baseline/roadmap docs so the next
  agent does not start from stale status.
