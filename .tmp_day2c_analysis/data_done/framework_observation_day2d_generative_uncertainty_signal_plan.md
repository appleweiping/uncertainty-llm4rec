# Framework-Observation-Day2d Generative Uncertainty Signal Plan

Status: plan only. Do not run until Day2c confirms clean label-first candidate-grounded output.

## Motivation

Day2 exposed placeholder generation failure. Day2b fixed candidate-title validity but still had explanatory text and unusable raw verbalized confidence. Day2c tests whether label-first generation gives clean candidate-grounded output. If Day2c output control succeeds but confidence remains low-variance or uninformative, the next observation should stop tuning verbalized confidence and extract non-verbal uncertainty signals.

## Candidate Signals

1. Label logprob:
   - Extract probabilities for labels A-F at the `selected_label` position.
   - Use top-1 label probability, top-1/top-2 margin, entropy over A-F, and calibrated selected-label confidence.

2. Title logprob:
   - Compute average token logprob of the generated `recommended_title`.
   - Separate title likelihood from recommendation correctness.

3. Retrieval margin:
   - Ground generated title to candidate/catalog.
   - Use match-score gap between top-1 and top-2 retrieved candidates as an uncertainty proxy.

4. Self-consistency title agreement:
   - Sample multiple label-first generations for the same user-candidate pool.
   - Use selected-label frequency, title agreement rate, vote entropy, and majority confidence.

5. Calibration:
   - Fit calibrators on valid split.
   - Evaluate on test split.
   - Targets: matched-title correctness, label correctness, and generation validity.

## Decision Rule

If non-verbal uncertainty beats raw verbalized confidence on AUROC/ECE/Brier and remains stable under candidate shuffling, use it as the next confidence observation line. If all signals remain weak, treat base Qwen candidate-grounded generation as valid but modest, and investigate stronger recommendation context or model adaptation before any CEP/evidence integration.
