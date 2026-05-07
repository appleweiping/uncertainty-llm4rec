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

- Gate R0: pre-experiment reviewer gate.

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

Next allowed action after Gate R0 pass:

- Run real pilot experiments on one dataset.

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
- Next OpenP5 step: replace the deterministic smoke scorer with a real
  OpenP5/T5 candidate scorer or training/evaluation run, then import with
  `scripts/import_external_predictions.py --split test` and evaluate with
  TRUCE only.
- Main external-framework comparison should now use controlled Qwen3-8B base
  model baselines rather than mixing different upstream backbones. See
  `docs/qwen3_lora_controlled_baselines.md`.
- Stricter baseline fidelity rule: after fairness controls are fixed, final
  main-table baselines must reuse official project implementation as much as
  possible. Current TRUCE-side Qwen3 adapters are controlled-adapter
  pilots unless a fidelity audit promotes them to official-native controlled
  baselines. See `docs/controlled_baseline_fidelity_audit.md`.
- These baselines are selected from the recommended/reference LLM4Rec project
  families. The main claim should be framework-vs-framework under the same
  Qwen3-8B base model and TRUCE protocol, not copied official metrics with
  mismatched backbones and not generic prompt baselines mislabeled as official
  methods. LoRA/adapter training should follow each baseline's official
  algorithm.
- Do observation analysis on controlled baselines too: check whether the
  motivating TRUCE/CU-GR observation phenomena appear in TALLRec/OpenP5/
  DEALRec/LC-Rec outputs, not only in weak/base-model outputs.
- Current Amazon Beauty data is acceptable for pipeline and early controlled
  comparison, but final top-conference-strength experiments should rerun the
  same suite on the larger dataset being generated on the same server.
- Large-scale target data is being produced in
  `~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/` for
  books/electronics/movies with same-candidate 10k-user, 1-positive+100-negative
  protocol. See `docs/week8_large_same_candidate_protocol.md`.
- First controlled adapter-pilot suite:
  `TALLRec-Qwen3-LoRA`, `OpenP5-style-Qwen3-LoRA`,
  `DEALRec-Qwen3-LoRA`, and `LC-Rec-Qwen3-LoRA`.
- Two additional official baseline families are now selected for follow-up:
  `LLaRA` for recommendation-signal alignment and `LLM-ESR` for long-tail
  sequential robustness.
- Prepare the suite with `python scripts/prepare_controlled_baseline_suite.py`,
  then run the generated server smoke queue before removing limits for the full
  long-running experiments.

Latest server status as of 2026-05-06:

- Main4 smoke is complete for all four controlled baselines:
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
  seconds. Do not start a full OpenP5-style run until scoring is batched or
  otherwise optimized.
- Next recommended server action: full-run the fast three controlled baselines
  first: TALLRec, DEALRec, and LC-Rec. Use `~/projects/TALLRec/.venv_tallrec`
  for Qwen/torch/peft execution, not `.venv_truce`.
- After each full run, import with
  `scripts/import_external_predictions.py --split test` and evaluate with
  `scripts/evaluate_predictions.py`. Final paper metrics must come from TRUCE
  evaluator outputs only.
