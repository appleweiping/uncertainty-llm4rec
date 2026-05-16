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

- Gate R1: server-first four-domain controlled experiment buildout.

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

- Pull latest on the server and inspect Week8 task availability.
- Convert/validate Week8 same-candidate tasks when available.
- Keep Beauty controlled-adapter pilots as legacy diagnostics only.
- Reuse Pony/Uncertainty official-qwen3base same-candidate baseline evidence
  through `docs/pony_official_baseline_reuse.md`.
- Run Ours and ablations on the same candidate rows after data conversion.

Server handoff:

- Server repo path: `~/projects/TRUCE-Rec`.
- Server OpenP5 path: `~/projects/OpenP5`.
- Use SSH Git remote on server:
  `git@github.com:appleweiping/TRUCE-Rec.git`. HTTPS pull failed once with a
  GnuTLS termination error, while `ssh -T git@github.com` authenticated as
  `appleweiping`.
- OpenP5 Amazon Beauty adapter smoke completed on the server at
  `outputs/runs/openp5_adapter_smoke_amazon_beauty_seed13`.
  It used deterministic no-model scores from the OpenP5 bridge template, not an
  official OpenP5 checkpoint.
- Smoke metrics from TRUCE evaluator on 225 test examples:
  Recall@10 0.017778, NDCG@10 0.005872, MRR@10 0.002519.
- OpenP5 upstream/T5 and the old TRUCE Qwen3-LoRA controlled paths are
  appendix/reference or legacy diagnostics unless explicitly reopened.
- Main external-framework comparison now reuses Pony/Uncertainty
  official-qwen3base same-candidate evidence. See
  `docs/pony_official_baseline_reuse.md`.
- Reused Pony baselines must keep their copied evidence package, sha256,
  status label, and score schema provenance. Pending rows stay out of main
  tables.
- Do observation analysis on Ours and reused strong baselines where the needed
  score/prediction artifacts are available.
- Current Amazon Beauty data is acceptable for pipeline and early diagnostics,
  but final top-conference-strength TRUCE claims should use the four-domain
  same-candidate artifacts.
- Large-scale target data is being produced in
  `~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/` for
  beauty/books/electronics/movies when available, with same-candidate
  10k-user, 1-positive+100-negative protocol. See
  `docs/week8_large_same_candidate_protocol.md`.
- First controlled adapter-pilot suite:
  `TALLRec-Qwen3-LoRA`, `OpenP5-style-Qwen3-LoRA`,
  `DEALRec-Qwen3-LoRA`, and `LC-Rec-Qwen3-LoRA`. This suite is legacy/pilot
  only and should not be run as the default paper-baseline route.

Latest server status as of 2026-05-06:

- Legacy Main4 smoke is complete for all four controlled baselines:
  `TALLRec-Qwen3-LoRA`, `OpenP5-style-Qwen3-LoRA`,
  `DEALRec-Qwen3-LoRA`, and `LC-Rec-Qwen3-LoRA`.
- Completed smoke artifact directories:
  `outputs/server_training/controlled_baselines/tallrec_qwen3_lora_amazon_beauty`,
  `outputs/server_training/controlled_baselines/openp5_style_qwen3_lora_amazon_beauty`,
  `outputs/server_training/controlled_baselines/dealrec_qwen3_lora_amazon_beauty`,
  and
  `outputs/server_training/controlled_baselines/lc_rec_qwen3_lora_amazon_beauty`.
- TALLRec/DEALRec/LC-Rec smoke scoring is fast after switching generic
  baselines to pairwise `Yes.` likelihood. OpenP5-style smoke works but is too
  slow for full scoring in the current runner: two score rows took about 763
  seconds. Do not start full runs unless the user explicitly reopens the
  legacy controlled-adapter lane.
- Next recommended baseline action: import/copy Pony official baseline evidence
  with `scripts/import_pony_official_baselines.py`, then build TRUCE status
  tables with `scripts/build_pony_baseline_comparison.py`.

Next non-toy buildout:

- Convert and validate Week8 same-candidate tasks with strict target-in-candidate
  checks.
- Keep the Pony baseline manifest current as missing/pending evidence arrives.
- Run Ours and ablations on the same candidate rows.

Future-agent update rule:

- After a completed stage, update `docs/PROJECT_MEMORY.md`, this handoff,
  README, server command docs, and relevant baseline/roadmap docs so the next
  agent does not start from stale status.
