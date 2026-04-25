# Day29 Movies medium_5neg_2000 Runtime Monitor

- expected_valid_rows: `12000`
- expected_test_rows: `12000`
- output_dir: `output-repaired/movies_deepseek_relevance_evidence_medium_5neg_2000/`
- metric_note: `HR@10 is trivial because candidate_pool_size=6.`

## Row Count Checks

```powershell
(Get-Content output-repaired\movies_deepseek_relevance_evidence_medium_5neg_2000\predictions\valid_raw.jsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
(Get-Content output-repaired\movies_deepseek_relevance_evidence_medium_5neg_2000\predictions\test_raw.jsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
```

## Resume Commands

```powershell
py -3.12 main_infer.py --config configs/exp/movies_deepseek_relevance_evidence_medium_5neg_2000.yaml --split_name valid --concurrent --resume
py -3.12 main_infer.py --config configs/exp/movies_deepseek_relevance_evidence_medium_5neg_2000.yaml --split_name test --concurrent --resume
```