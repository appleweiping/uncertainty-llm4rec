# Self Review 2026-05-01

This self-review follows the latest DeepSeek Health / Video_Games prompt-gate
documentation, the candidate-leakage guard, and the API-free CURE/TRUCE gate
diagnostic documentation. It is a governance and research-quality artifact, not
an experimental result.

## Current Phase

Storyflow / TRUCE-Rec is now between Phase 3, early Phase 4, and scaffolded
Phase 5.

- Phase 0 governance/scaffold is complete.
- Phase 1 data support is beyond MovieLens. Amazon Beauty is the first full
  e-commerce category; local processed and observation-input artifacts are
  ignored and must not be treated as paper results.
- Phase 2/3 observation infrastructure is broad: mock, API dry-run, real
  DeepSeek gate artifacts, analysis, case review, candidate diagnostics, run
  comparison, grounding diagnostics, and local ignored run registry are
  implemented.
- Lightweight baseline observation is implemented for popularity,
  train-split co-occurrence, and ranking-JSONL-to-title adaptation. It is a
  contract for later trained rankers, not trained-baseline evidence.
- Qwen3-8B server observation has a plan/execution contract only. Codex has
  not run Qwen3 inference, server inference, LoRA training, or large baseline
  training.
- Phase 4 CURE/TRUCE is implemented as API-free scaffold code: feature schema,
  grounded-observation feature builder, deterministic scoring, histogram
  calibration, popularity residualization, reranking, and compact
  selective-risk diagnostics. These are not learned method results.
- Phase 5 is scaffolded through synthetic exposure simulation and diagnostic
  data triage over CURE/TRUCE feature rows. These are synthetic/scaffold
  contracts, not real feedback-loop evidence.

## Recent Closure Check

Recent commits changed the project state since the previous self-review:

- `692b5d6` added selective-risk observation analysis.
- `a24033e` connected compact selective-risk diagnostics to rerank and triage
  manifests.
- `ea3d35b` updated the selective-risk phase review.
- `02eb086`, `a3539b7`, and `57bedc1` recorded local Amazon / DeepSeek
  observation artifacts and prompt-gate documentation.
- `7798760` guarded against observation candidate leakage.
- `83f4de3` recorded CURE/TRUCE gate diagnostic documentation.

The working tree is clean at the time of this review. The ignored
`docs/reviewer_report.md` from 2026-04-29 is now stale in several details: its
baseline blocker was closed by later commits. It should still be read for risk
style, but it must not be committed.

## Mainline Check

The project still matches the Storyflow core task:

```text
user history item titles
  -> generated or selected item title
  -> catalog grounding
  -> correctness + confidence + popularity + grounding + head/mid/tail analysis
  -> exposure-aware confidence framework
```

The project has not collapsed into ordinary top-k recommendation. Even
ranking baselines must produce or select catalog titles, pass through grounding,
and enter the same correctness/confidence/popularity schema before analysis.

## Grounding Check

All current observation paths require grounding before correctness:

- API and mock observation write generated titles, parsed predictions, and
  grounded predictions as separate layers.
- Baseline observation converts selected item IDs into catalog titles and then
  grounds those titles.
- Qwen3 server observation is a plan-compatible writer for the same layered
  schema.
- Analysis, case review, and CURE/TRUCE feature builders consume grounded
  predictions rather than raw text completions.

The main remaining risk is not a missing grounding step. The risk is that
free-form title generation can fail grounding, while retrieval-context and
catalog-constrained prompt gates can make target correctness hard to interpret
because candidates may intentionally exclude the held-out target. These prompt
gates must remain prompt/candidate/grounding diagnostics unless the candidate
policy supports ordinary recommendation-accuracy interpretation.

## Confidence Analysis Check

Confidence is currently analyzed jointly with:

- correctness;
- GroundHit and grounding status;
- target and generated popularity where available;
- head/mid/tail buckets;
- wrong-high-confidence and correct-low-confidence cases;
- Tail Underconfidence Gap where the slice supports it;
- ECE, Brier, AURC/selective-risk summaries, and reliability bins;
- repeat-target slices for e-commerce duplicate/repeat behavior;
- candidate-set adherence, selected candidate rank, target-excluded status,
  and history-copying for constrained prompt diagnostics.

For baseline observation, confidence is explicitly a non-calibrated proxy. For
CURE/TRUCE diagnostics on test-only gate artifacts, same-split calibration and
residualization are explicitly leakage-marked contract checks. They must not be
reported as learned calibration, causal deconfounding, or method improvement.

## Dataset Route

The data route remains coherent but still needs broader real execution:

1. Synthetic fixtures: tests only.
2. MovieLens 1M: sanity and low-cost API substrate only.
3. Amazon Beauty: first full e-commerce category and the active local
   protocol substrate.
4. Health_and_Personal_Care and Video_Games: recently used prompt-gate
   diagnostic categories; useful for grounding/prompt-shape stress, not yet a
   complete paper suite.
5. Sports_and_Outdoors / Video_Games / Books: next likely title-rich
   robustness paths after approval, disk/network planning, and server/manual
   data handling.

The project is no longer toy-only, but it can still be attacked if it spends
too long on small local gates without a clear approved path to title-rich
Amazon categories, Qwen/server observation, and trained baseline artifacts.

## Baseline Route

Current baseline support is a good contract layer, not final reviewer-proof
coverage.

- Implemented: popularity title baseline.
- Implemented: train-split co-occurrence title baseline.
- Implemented: ranking-JSONL-to-title adapter.
- Implemented: source run-manifest and ranking-artifact validation gates.
- Implemented: analysis registry semantics for baseline confidence proxies.
- Missing: at least one trained SASRec/BERT4Rec/GRU4Rec/LightGCN-style
  artifact entering the adapter after validation.
- Missing later: generative baseline coverage such as P5-like, TIGER /
  Semantic-ID-like, BIGRec/grounding-style, and uncertainty-aware baselines
  where reproducible.

The next baseline step should use an actual approved trained ranking artifact
or server-side baseline run. Local Codex should not claim trained baseline
results without logs, manifests, and artifact validation.

## Qwen3 / Server Route

No server execution has been performed by Codex. The Qwen3 route is currently
a contract:

- `configs/server/qwen3_8b_observation.yaml` describes model/runtime/input
  expectations.
- `scripts/server/run_qwen3_observation.py` writes plan artifacts by default.
- Actual execution requires explicit user approval, a server path, GPU/model
  environment details, artifact-return rules, logs, manifests, and output
  inspection.

This is acceptable at the current stage, but the project cannot make
cross-model claims until Qwen API, Qwen3/server, Kimi, GLM, or other provider
artifacts actually exist.

## Framework Coherence

The unified object remains:

```text
C(u, i) ~= P(user accepts item i | user u, do(exposure=1))
```

Current implementation is coherent because each scaffold maps to that object:

- verbal confidence is treated as noisy evidence;
- generation evidence and grounding confidence remain separate fields;
- generated-item popularity is tracked separately from target popularity;
- popularity residualization is a split-audited confounding contract;
- reranking uses the selected confidence proxy with explicit fallback
  provenance;
- selective risk connects observation, rerank, and triage diagnostics;
- triage reason codes separate likely noise, hard-tail positives, grounding
  uncertainty, and popularity/echo overconfidence.

The main risk is overclaiming. These scaffolds still optimize offline
correctness/proxy confidence contracts, not exposure-counterfactual user
acceptance from real exposure logs.

## Reviewer-Attack Risks

1. Single-provider dependence.
   Priority: high. DeepSeek diagnostics cannot establish general model
   behavior. Mitigation: next real expansion must target another provider or
   Qwen3/server only after explicit approval.

2. Candidate-gate correctness confusion.
   Priority: high. Retrieval/context and catalog-constrained gates may exclude
   the held-out target. Mitigation: keep target correctness marked as
   non-recommendation-accuracy when candidate policy says so.

3. Same-split scaffold overclaim.
   Priority: high. Test-only gate artifacts force `--allow-same-split-eval`
   for contract diagnostics. Mitigation: manifests and docs must keep these as
   leakage-marked diagnostics, never learned calibrators.

4. Grounding-failure confounding.
   Priority: high. Free-form over/underconfidence may partly reflect title
   grounding ease. Mitigation: always report GroundHit, grounding status,
   grounding ambiguity, ungrounded-high-confidence, and generated popularity.

5. Baseline insufficiency.
   Priority: high. Lightweight baselines are not enough for a top-tier paper.
   Mitigation: feed the first approved trained ranking artifact through the
   existing validation and grounding adapter.

6. Qwen/server absence.
   Priority: high. The framework target names Qwen3-8B + LoRA, but only
   server contracts exist. Mitigation: keep current wording plan-only until
   real server artifacts are provided.

7. Full-data breadth risk.
   Priority: medium-high. Beauty and small gates are useful, but not enough.
   Mitigation: use Amazon readiness/run packets for Video_Games, Sports, or
   Books after user approval.

8. Implicit-feedback ambiguity.
   Priority: medium-high. Next item is not the only valid preference. Mitigation:
   use future-window or graded relevance where the data and protocol support it.

9. Popularity residual interpretation.
   Priority: medium. Popularity can be relevance signal and confounder.
   Mitigation: treat residualization as a split-audited diagnostic until a
   causal or counterfactual evaluation is available.

10. Governance drift.
    Priority: medium. The project now has many safe gates; adding more gates
    without execution can slow progress. Mitigation: choose concrete
    non-blocking engineering work, or ask for the minimal approvals needed for
    one real expansion.

## Priority Decision

Go forward with a bias toward execution-readiness, not more toy diagnostics.

Recommended next non-blocking work without user approval:

- build a concrete next-expansion packet for the most likely approved track
  without executing it;
- tighten Amazon title-rich full-prepare readiness documentation for
  Video_Games / Books if raw placement or server requirements are missing;
- add small API-free safeguards only where they directly protect grounding,
  split separation, baseline artifact validation, or CURE/TRUCE claim scope.

Do not run real API expansion, Qwen3/server inference, LoRA training, large
baseline training, or additional full-data processing without explicit user
approval and concrete run gates.

