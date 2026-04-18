param(
    [string]$ExpConfig = "configs/exp/beauty_qwen_local_rank.yaml",
    [string]$ValidInput = "",
    [string]$TestInput = "",
    [int]$MaxSamples = 20
)

$ErrorActionPreference = "Stop"

if (-not $ValidInput) {
    $ValidInput = "data/processed/amazon_beauty/valid.jsonl"
}
if (-not $TestInput) {
    $TestInput = "data/processed/amazon_beauty/test.jsonl"
}

$expName = (Get-Content $ExpConfig | Select-String "^exp_name:" | ForEach-Object { $_.Line.Split(":")[1].Trim() })
if (-not $expName) {
    throw "Cannot resolve exp_name from $ExpConfig"
}

Write-Host "[Version3] Running local rank pipeline for $expName"

py -3.12 main_infer.py `
  --config $ExpConfig `
  --input_path $ValidInput `
  --split_name valid `
  --max_samples $MaxSamples `
  --overwrite

py -3.12 main_infer.py `
  --config $ExpConfig `
  --input_path $TestInput `
  --split_name test `
  --max_samples $MaxSamples `
  --overwrite

py -3.12 main_eval.py `
  --exp_name $expName `
  --input_path ("outputs/{0}/predictions/test_ranking_raw.jsonl" -f $expName) `
  --task_type candidate_ranking `
  --output_root outputs `
  --k 10

py -3.12 main_calibrate.py `
  --exp_name $expName `
  --valid_path ("outputs/{0}/predictions/valid_ranking_raw.jsonl" -f $expName) `
  --test_path ("outputs/{0}/predictions/test_ranking_raw.jsonl" -f $expName) `
  --task_type candidate_ranking `
  --method isotonic

py -3.12 main_uncertainty_compare.py `
  --exp_name ("{0}_compare" -f $expName) `
  --task_type candidate_ranking `
  --input_path ("outputs/{0}/predictions/test_ranking_raw.jsonl" -f $expName) `
  --output_root outputs `
  --k 10

py -3.12 main_rerank.py `
  --exp_name ("{0}_rerank" -f $expName) `
  --task_type candidate_ranking `
  --input_path ("outputs/{0}/predictions/test_ranking_raw.jsonl" -f $expName) `
  --output_root outputs `
  --k 10 `
  --lambda_penalty 0.5 `
  --ranking_score_source raw_score `
  --ranking_uncertainty_source inverse_probability

Write-Host "[Version3] Local rank pipeline completed for $expName"
