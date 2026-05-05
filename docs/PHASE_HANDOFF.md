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
- Main external-framework comparison should now use controlled Qwen3-8B LoRA
  baselines rather than mixing different upstream backbones. See
  `docs/qwen3_lora_controlled_baselines.md`.
- First controlled main-table suite:
  `TALLRec-Qwen3-LoRA`, `OpenP5-style-Qwen3-LoRA`,
  `DEALRec-Qwen3-LoRA`, and `LC-Rec-Qwen3-LoRA`.
- Prepare the suite with `python scripts/prepare_controlled_baseline_suite.py`,
  then run the generated server smoke queue before removing limits for the full
  long-running experiments.
