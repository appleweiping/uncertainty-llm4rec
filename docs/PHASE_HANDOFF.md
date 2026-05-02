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
- Ignored local diagnostics under `outputs/` or `data/processed/` are not paper
  evidence unless explicitly promoted by a later approved protocol.
- Formal paper results must come from approved real experiment configs,
  tracked code, saved configs, logs, raw outputs, predictions, and metrics.

Next allowed action after Gate R0 pass:

- Run real pilot experiments on one dataset.
