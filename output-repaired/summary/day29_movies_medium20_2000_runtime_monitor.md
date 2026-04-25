# Day29 Movies medium_20neg_2000 Runtime Monitor

- expected_valid_rows: `42000`
- expected_test_rows: `42000`
- output_dir: `output-repaired/movies_deepseek_relevance_evidence_medium_20neg_2000/`

## Row Count Checks

```powershell
(Get-Content output-repaired\movies_deepseek_relevance_evidence_medium_20neg_2000\predictions\valid_raw.jsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
(Get-Content output-repaired\movies_deepseek_relevance_evidence_medium_20neg_2000\predictions\test_raw.jsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
```

## Tail Logs

```powershell
Get-Content output-repaired\summary\day29_movies_medium20_2000_valid_inference.log -Tail 40 -Wait
Get-Content output-repaired\summary\day29_movies_medium20_2000_test_inference.log -Tail 40 -Wait
```

## Resume Commands

```powershell
py -3.12 main_infer.py --config configs/exp/movies_deepseek_relevance_evidence_medium_20neg_2000.yaml --split_name valid --concurrent --resume
py -3.12 main_infer.py --config configs/exp/movies_deepseek_relevance_evidence_medium_20neg_2000.yaml --split_name test --concurrent --resume
```

## Rolling Parse Success

```powershell
py -3.12 -c "import json, pathlib; p=pathlib.Path('output-repaired/movies_deepseek_relevance_evidence_medium_20neg_2000/predictions/valid_raw.jsonl'); rows=[json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]; print(len(rows), sum(r.get('parse_success') is True for r in rows)/max(len(rows),1))"
py -3.12 -c "import json, pathlib; p=pathlib.Path('output-repaired/movies_deepseek_relevance_evidence_medium_20neg_2000/predictions/test_raw.jsonl'); rows=[json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]; print(len(rows), sum(r.get('parse_success') is True for r in rows)/max(len(rows),1))"
```