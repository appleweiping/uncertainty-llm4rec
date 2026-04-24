# Uncertainty-Aware Recommendation

> Defining, calibrating, and operationalizing task-grounded uncertainty signals for recommendation.

This repository studies **uncertainty for recommendation**: how uncertainty signals should be defined, calibrated, and used inside candidate scoring, ranking, reranking, and exposure-aware recommendation decisions.

Local `8B + LoRA` models are used as a scalable execution carrier. They make the experiments cheaper, faster, and more reproducible than relying only on official API calls, while keeping the method focus on uncertainty-aware recommendation.

## Core Claim

The project starts from verbalized confidence, uses it to expose where recommendation uncertainty components succeed or fail, and then develops more task-grounded signals. The current paper direction is:

**Recommendation systems should not blindly rely on verbalized self-confidence. They should use task-grounded uncertainty signals, calibrate them, and choose suitable signal-to-decision modules under different domain conditions.**

In practice, the project is organized around three layers:

- `light`: the completed local replay and diagnostic line for verbalized confidence. It validates the original uncertainty pipeline and identifies which components of yes/no self-confidence, calibration, reranking, and exposure analysis are reliable or fragile.
- `shadow`: the task-grounded signal extension derived from the issues exposed by `light`, including relevance posterior, top-k inclusion posterior, preference strength, rank-position distribution, and intent-prototype match.
- `system`: the domain-adaptive signal-to-decision layer that decides which uncertainty signal should influence ranking, reranking, fallback, and exposure behavior.

## Development Story

The project started from a narrow but important uncertainty question: when an LLM gives a recommendation together with a confidence value, is that confidence a useful signal or just a stylistic artifact of generation? The first implementation therefore focused on a pointwise recommendation diagnosis pipeline: build user-item samples, query an LLM, parse a yes/no decision and verbalized confidence, evaluate correctness, and measure calibration through `ECE`, `Brier`, `AUROC`, and confidence-bin behavior.

That first stage was useful because it established the evidence vocabulary for the whole project. It made confidence measurable instead of anecdotal. It also introduced the leakage-aware calibration protocol that remains important later: fit a calibrator on `valid`, apply it to `test`, and report raw-vs-calibrated reliability separately. At this point, the project was still close to an old-style pointwise yes/no formulation, but it had already clarified the key methodological principle: uncertainty should not be trusted merely because the model verbalizes it; it has to be diagnosed, calibrated, and stress-tested.

The second stage moved uncertainty from observation into recommendation decision-making. The project added candidate ranking, pairwise preference, uncertainty-aware reranking, and exposure-side metrics. This was the first major shift toward recommendation as the real center of the work. Instead of asking only whether a single item should be recommended, the pipeline began asking how uncertainty changes the ordering of candidates, whether high-risk candidates should be penalized, and whether the resulting top-k list changes head exposure or long-tail coverage. `structured risk` emerged from this stage as a hand-crafted uncertainty-to-rerank baseline: it is not the final method, but it proves that calibrated or estimated uncertainty can become a decision variable in ranking.

The third stage addressed execution scale. Early experiments used official API models heavily, which was helpful for observation and reference evidence but too slow, expensive, and brittle for sustained four-domain experimentation. The local `Qwen3-8B + LoRA` route was introduced for this reason. Its role is practical and methodological: it gives the project a controllable, server-runnable execution carrier so the uncertainty pipeline can be replayed and extended without depending on API throughput. This is the origin of the `light` line. `light` is important because it completes the local execution path and makes the behavior of the original uncertainty components visible at larger scale.

Those component-level findings became the fourth stage. Once `light` showed that yes/no verbalized confidence can be calibrated but may still have poor sample-level discrimination, the project gained a concrete map of what needs to be improved: the signal should be closer to recommendation semantics, less dependent on self-reported confidence, and easier to connect to ranking decisions. This is where the `shadow` direction comes from. The point is not to replace `light`; without the light-side diagnosis, there would be no clear reason for the shadow-side design. A relevance posterior asks whether a candidate is likely to match the user. A top-k inclusion posterior asks whether it belongs in the recommendation frontier. A preference-strength signal asks how strongly it aligns with user history. A rank-position distribution asks where it should appear in a list. An intent-prototype match signal asks whether the candidate matches the user's inferred preference prototype.

The fifth stage is the current paper-facing consolidation. The story is now circular in a good way: the project begins with confidence diagnosis, uses `light` to make the strengths and weaknesses of the original components visible, moves uncertainty into ranking and exposure decisions, replaces API-only execution with local LoRA replay, and then uses those findings to motivate task-grounded signal design. The final system is therefore not a loose collection of experiments. It is a path from uncertainty observation to uncertainty signal design to uncertainty-aware recommendation decision-making.

The paper story can be told as:

1. Old verbalized confidence is measurable and sometimes useful, but not reliable enough to be the final uncertainty signal for recommendation.
2. Calibration can repair probability scale, but it cannot by itself create strong discrimination if the underlying signal is weak.
3. Uncertainty becomes more meaningful when it is connected to candidate ranking, reranking, exposure, and robustness.
4. Local `8B + LoRA` execution makes the pipeline scalable enough to test these questions beyond API-limited compact runs.
5. The main method direction is therefore task-grounded uncertainty signal design and domain-adaptive signal-to-decision integration.

## Research Questions

The early `light` pipeline answers four foundation questions:

- Is LLM confidence informative in recommendation, or merely stylistic?
- How miscalibrated is verbalized confidence, and can it be corrected post hoc?
- Can calibrated uncertainty influence downstream ranking behavior in a controlled way?
- Does uncertainty-awareness change not only ranking metrics, but also exposure patterns across head and long-tail items?

These are treated as foundation questions, not the final method claim. The current direction uses the observed limits of verbalized confidence to motivate task-grounded uncertainty signals for recommendation.

## Method Stack

The implemented workflow is:

```text
Data processing
  -> pointwise diagnosis / calibration
  -> candidate ranking
  -> uncertainty-aware reranking
  -> exposure and robustness analysis
  -> summary / comparison tables
```

The repository currently includes:

- Pointwise uncertainty diagnosis with `accuracy`, `Brier`, `ECE`, `MCE`, `AUROC`, confidence bins, and calibration figures.
- Leakage-aware post-hoc calibration: fit on `valid`, apply on `test`.
- Candidate ranking evaluation with `HR@10`, `NDCG@10`, `MRR`, `coverage@10`, `head_exposure_ratio@10`, and `longtail_coverage@10`.
- Structured-risk reranking as a hand-crafted uncertainty-to-decision baseline.
- Pairwise preference and pairwise-to-rank mechanism experiments.
- Local Hugging Face backend support for server-side Qwen3-8B execution.
- LoRA ranking training and evaluation paths used as scalable local execution routes.
- Summary builders for light replay, SRPD-style variants, robustness, and compare tables.

`shadow` signal variants are being designed as task-grounded extensions of the issues found by `light`. They should not be read as completed benchmark results unless corresponding output tables are present.

## Project Positioning

`official API` routes remain useful for historical observation, external reference evidence, and case studies.

`local 8B + LoRA` is the scalable execution carrier. It exists so the uncertainty-for-recommendation pipeline can run without depending on slow or expensive API-only execution.

`structured risk` is the strongest hand-crafted uncertainty-aware rerank baseline. It tests whether calibrated or estimated uncertainty can affect ranking decisions without additional training.

`light` is the confidence replay and diagnostic line. It is valuable because it completes the local execution path and shows which original uncertainty components are reliable, fragile, or insufficient for recommendation decisions.

`shadow` is the task-grounded extension line: it turns the component issues surfaced by `light` into more recommendation-native uncertainty signals and signal-to-decision modules.

`SOTA proxy` refers to fair, protocol-aware approximations of external literature baselines when exact reproduction is not possible. Proxy comparisons must record metric family, candidate setting, confidence availability, and whether the result is same-schema, executable proxy, or related-work-only.

## Repository Layout

```text
configs/
  data/                 # domain preprocessing configs
  exp/                  # pointwise, ranking, rerank, replay configs
  lora/                 # LoRA training configs
  model/                # API and local-HF model configs

prompts/                # task-specific prompt templates

src/
  analysis/             # summary and comparison builders
  data/                 # preprocessing and sample construction
  eval/                 # pointwise, ranking, pairwise metrics
  llm/                  # API/local-HF backends, prompt builder, parser
  methods/              # rerankers, baselines, pairwise aggregation
  training/             # LoRA training utilities

main_*.py               # CLI entry points
outputs/                # generated predictions, tables, figures, summaries
artifacts/              # adapters and training logs
```

`Paper/` contains local planning and paper notes. It is intentionally not part of the main GitHub-facing experiment interface.

## Main Entry Points

Pointwise diagnosis:

```bash
python main_infer.py --config configs/exp/<pointwise_config>.yaml --input_path <valid_or_test_jsonl> --split_name test
python main_eval.py --exp_name <exp_name> --output_root outputs
python main_calibrate.py --exp_name <exp_name> --output_root outputs --method isotonic
```

Candidate ranking:

```bash
python main_rank.py --config configs/exp/<rank_config>.yaml
python main_eval_rank.py --exp_name <rank_exp_name> --output_root outputs
```

Uncertainty-aware reranking:

```bash
python main_rank_rerank.py --config configs/exp/<rerank_config>.yaml
```

LoRA ranking training:

```bash
python main_lora_train_rank.py --config configs/lora/<lora_config>.yaml --startup_check
python main_lora_train_rank.py --config configs/lora/<lora_config>.yaml
python main_eval_lora_rank.py --config configs/lora/<lora_config>.yaml --overwrite --resume_partial
```

Week7.8 light replay summaries:

```bash
python main_compare_teacher_requested_line.py --mode light_pointwise
python main_compare_teacher_requested_line.py --mode light_rerank
python main_compare_teacher_requested_line.py --mode light_robustness
python main_compare_teacher_requested_line.py --mode light_final
```

## Environment

Use Python 3.12 locally when possible. On the server, use the project environment already configured for Qwen3-8B, PEFT, and Transformers.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For local-HF execution, model paths are configured under `configs/model/`. Server credentials, temporary machine links, and local model cache paths should not be committed.

## Outputs

The main generated artifacts are:

- `outputs/<exp_name>/predictions/*.jsonl`
- `outputs/<exp_name>/tables/*.csv`
- `outputs/<exp_name>/figures/*`
- `outputs/<exp_name>/calibrated/*.jsonl`
- `outputs/summary/*.csv`
- `outputs/summary/*.md`
- `artifacts/adapters/<run_name>/`
- `artifacts/logs/<run_name>/`

Typical summary files include pointwise diagnostics, calibration comparisons, ranking metrics, structured-risk rerank results, robustness summaries, and final light/shadow/system compare tables.

## Evaluation Philosophy

Utility metrics alone are not enough. The project reports ranking quality together with calibration, parse reliability, candidate coverage, head exposure, long-tail coverage, and robustness under noisy or shifted inputs.

The comparison protocol is intentionally conservative:

- Same-schema comparisons are preferred whenever data, split, candidate space, and metrics are aligned.
- Proxy baselines are explicitly marked when exact external reproduction is not available.
- NH and NR literature protocols are audited separately: NH usually means `NDCG + Hit Ratio`, while NR usually means `NDCG + Recall`.
- `HR@K` and `Recall@K` are treated as equivalent only in one-positive candidate evaluation settings.

## Current Status

The project currently has:

- A completed `light` replay route for verbalized-confidence diagnosis under local execution.
- Evidence that old self-confidence can be calibrated but remains weakly discriminative in some settings.
- Candidate ranking and structured-risk rerank pipelines with utility and exposure metrics.
- SRPD-style light variants and LoRA ranking adapters as local execution evidence.
- A planned `shadow` route that extends `light` findings into task-grounded uncertainty signals and domain-adaptive signal-to-decision modules.

The next research emphasis is not to make the README a full weekly log, but to keep the repository aligned with the paper claim: **uncertainty-aware recommendation through task-grounded signal design, calibration, and decision integration**.

## Notes

Generated outputs, adapters, synced server archives, and temporary transfer directories can be large. Keep them out of source commits unless a specific artifact is intentionally tracked.

Paper planning notes may live locally under `Paper/`, but they are separate from the GitHub-facing code and experiment interface.

## License

MIT, unless noted otherwise.
