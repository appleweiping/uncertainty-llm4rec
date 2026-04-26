# Framework-Day7 Qwen-LoRA Eval Diagnosis Report

## 1. Day6 Recap

Day6 completed train -> save adapter -> infer -> parse -> evaluate. However, performance was weak: Qwen-LoRA was only slightly above random on NDCG/MRR, while HR@3/NDCG@5 were lower than random. Parse success was `0.8672`.

## 2. Parse Failure Diagnosis

- prediction source: `None`
- parse success rate: `NA`
- schema valid rate: `NA`
- non-JSON rate: `NA`
- missing key rate: `NA`
- too-few-items rate: `NA`

## 3. Ranking Case Study

See `data_done/framework_day7_ranking_case_study.csv` for hit@1, rank 2-3, rank >3, and parse failure examples.

## 4. Eval Sample Sensitivity

512/large-subset eval status: `skipped`. If completed, see `data_done/framework_day7_eval_512_summary.csv`.

## 5. Base vs LoRA

Base comparison status: `skipped_due_to_runtime_or_not_requested`. If skipped, this is due to runtime control; run with `--run_base_comparison` on the server if needed.

## 6. Prompt / Parser Repair Plan

See `data_done/framework_day7_prompt_parser_repair_plan.md`.

## 7. Day8 Recommendation

If parse failures are mainly schema/key/extra-text issues, first do prompt/parser repair and re-evaluate the same raw outputs. If parse improves but ranking remains weak, run a 500-1000 step Beauty listwise train. If base and LoRA look similar, 100 steps is likely insufficient. Do not enter CEP/confidence/evidence framework until the Qwen-LoRA baseline is stable.
