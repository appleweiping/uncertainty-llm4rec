param(
    [int]$ValidPid = 0
)

$ErrorActionPreference = "Stop"
$ExpName = "movies_deepseek_relevance_evidence_medium_20neg_2000"
$Config = "configs/exp/movies_deepseek_relevance_evidence_medium_20neg_2000.yaml"
$OutDir = "output-repaired/movies_deepseek_relevance_evidence_medium_20neg_2000"
$SummaryDir = "output-repaired/summary"
$ValidRaw = Join-Path $OutDir "predictions/valid_raw.jsonl"
$TestRaw = Join-Path $OutDir "predictions/test_raw.jsonl"
$ValidLog = Join-Path $SummaryDir "day29_movies_medium20_2000_valid_inference.log"
$ValidErr = Join-Path $SummaryDir "day29_movies_medium20_2000_valid_inference.err.log"
$TestLog = Join-Path $SummaryDir "day29_movies_medium20_2000_test_inference.log"
$TestErr = Join-Path $SummaryDir "day29_movies_medium20_2000_test_inference.err.log"
$CalLog = Join-Path $SummaryDir "day29_movies_medium20_2000_calibration.log"
$AnalysisLog = Join-Path $SummaryDir "day29_movies_medium20_2000_analysis.log"

function Count-Lines($Path) {
    if (Test-Path $Path) {
        return (Get-Content $Path | Measure-Object -Line).Lines
    }
    return 0
}

function Run-Inference($Split, $LogPath, $ErrPath) {
    & py -3.12 main_infer.py --config $Config --split_name $Split --concurrent --resume --max_workers 4 --requests_per_minute 120 1> $LogPath 2> $ErrPath
}

New-Item -ItemType Directory -Force -Path $SummaryDir | Out-Null

if ($ValidPid -gt 0) {
    try {
        Wait-Process -Id $ValidPid
    } catch {
        "Valid PID $ValidPid was not running when pipeline checked it." | Add-Content $ValidLog
    }
}

$validRows = Count-Lines $ValidRaw
if ($validRows -lt 42000) {
    Run-Inference "valid" $ValidLog $ValidErr
    $validRows = Count-Lines $ValidRaw
}
if ($validRows -ne 42000) {
    throw "Valid inference incomplete: $validRows / 42000"
}

$testRows = Count-Lines $TestRaw
if ($testRows -lt 42000) {
    Run-Inference "test" $TestLog $TestErr
    $testRows = Count-Lines $TestRaw
}
if ($testRows -ne 42000) {
    throw "Test inference incomplete: $testRows / 42000"
}

& py -3.12 main_calibrate_relevance_evidence.py --exp_name $ExpName --output_root output-repaired 1> $CalLog 2>&1
& py -3.12 main_day29_movies_medium20_2000_analysis.py --run 1> $AnalysisLog 2>&1

"Day29 Movies medium_20neg_2000 pipeline completed." | Add-Content $AnalysisLog
