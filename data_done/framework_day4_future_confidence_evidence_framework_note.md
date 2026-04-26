# Future Confidence + Evidence Framework Note

The future CEP framework combines two observation lines.

1. Confidence line: Week1-Week4 verbalized confidence, calibration, robustness, and multi-model observation from `Paper/version1`, `Paper/version2`, and older week1-week4 outputs.
2. Evidence line: Day9+ evidence decomposition, calibrated relevance posterior, evidence risk, external backbone plug-in, and robustness from `output-repaired`.

Qwen-LoRA is a fixed local recommendation baseline. It should learn relevance/ranking ability, not calibrated probability or final CEP risk. Confidence calibration, evidence risk, and CEP fusion are decision-stage framework modules fitted or selected on valid data and fixed for test evaluation.

Framework-Day4 intentionally does not implement confidence module, evidence module, or CEP fusion. It only verifies that the Qwen-LoRA baseline training pipeline is clean.
