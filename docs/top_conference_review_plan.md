# Top-Conference Review Plan

This plan records the internal review standard before paper writing. It should
be used by future Codex agents as a reviewer checklist, not as a claim that the
paper is ready.

## Likely Rejection Reasons

1. **Novelty looks like generic LLM reranking.**
   Defense: tie every method component to the observation object: grounded
   title uncertainty, popularity residuals, long-tail under-confidence, and
   exposure-aware routing.

2. **Novelty looks stitched from senior-recommended or recent projects.**
   Defense: use those projects only for reproduction fidelity and inspiration.
   The method contribution must be an original observation-to-policy framework
   with TRUCE-specific targets, diagnostics, conservative fallback, and
   ablations. Reject paper text that merely renames another project's training
   objective, prompt format, or pipeline.

3. **Baselines are not official or are too weak.**
   Defense: use official-native controlled baselines for TALLRec, OpenP5,
   DEALRec, LC-Rec, LLaRA, and LLM-ESR where feasible; keep adapter pilots out
   of final main tables.

4. **Data is too toy.**
   Defense: move from tiny/MovieLens/Beauty debugging to four-domain Week8
   same-candidate experiments with up to 10k users and 100 negatives.

5. **Candidate protocol is unfair.**
   Defense: all methods score the exact same candidate rows and preserve event
   IDs for paired testing.

6. **Uncertainty metrics are unconvincing.**
   Defense: report ECE, Brier, selective risk, wrong-high-confidence cases,
   correct-low-confidence cases, and head/mid/tail reliability.

7. **Long-tail or diversity benefit is unclear.**
   Defense: include tail Recall/NDCG/MRR, coverage, novelty, diversity, and
   popularity-stratified analysis.

8. **Ablation does not isolate the proposed framework.**
   Defense: run component ablations for grounding, uncertainty policy,
   candidate-normalized confidence, popularity residuals, echo/history guard,
   and fallback-only routing.

9. **Cost/latency is ignored.**
   Defense: record runtime, GPU memory where available, API tokens/cost where
   applicable, and throughput.

## Reviewer-Implementation Loop

For complex method work, do not let a single implementation pass define the
final design. Run an explicit loop:

```text
implementation agent proposes or edits method/code
  -> reviewer agent critiques originality, fairness, leakage, and toy risk
  -> implementation agent revises code/configs/tests
  -> main agent updates server commands and docs
```

If subagents are unavailable, perform these as separate named passes in the
main agent. The task is not done until the reviewer pass has a clear verdict,
the remaining risks are documented, and the next server command is explicit.

## Required Reviewer Tables

- Main four-domain ranking table.
- Official-native controlled baseline table.
- Ours ablation table.
- Observation/reliability table.
- Long-tail/popularity-stratified table.
- Validity/hallucination/candidate-adherence table.
- Efficiency/cost table.
- Statistical significance table or appendix.

## Required Case Studies

- wrong-high-confidence generated title;
- correct-low-confidence tail item;
- hallucinated title that cannot be grounded;
- popularity-biased high-confidence recommendation;
- Ours reroutes away from risky generated item;
- Ours failure where uncertainty policy hurts.

## Internal Reviewer Agent Prompt

Use this prompt after each full experiment phase:

```text
Act as a SIGIR/WWW/RecSys/NeurIPS reviewer. Given the current metrics,
artifacts, configs, and docs, identify fatal flaws in originality, baseline
fairness, data scale, leakage, ablations, statistical testing, efficiency,
long-tail analysis, and reproducibility. Do not praise the work unless the
artifact evidence supports it. List missing experiments before paper claims.
Also check whether the proposed method is a genuinely original TRUCE/CURE
framework or only a stitched combination of senior-recommended papers.
```

## Literature Agent Prompt

Use this prompt before freezing baselines:

```text
Act as an LLM4Rec literature auditor. Check whether each selected baseline is
representative, official-native where claimed, and comparable under the shared
Qwen3-8B base-model and TRUCE candidate/evaluator protocol. Flag missing
recent baselines and classify them as main, appendix, long-tail, sequential, or
efficiency baselines.
```
