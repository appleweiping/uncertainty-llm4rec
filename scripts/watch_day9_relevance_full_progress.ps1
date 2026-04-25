param(
    [string]$Root = "D:\Research\Uncertainty-LLM4Rec",
    [string]$ExpName = "beauty_deepseek_relevance_evidence_full"
)

$predDir = Join-Path $Root "output-repaired\$ExpName\predictions"
$validPath = Join-Path $predDir "valid_raw.jsonl"
$testPath = Join-Path $predDir "test_raw.jsonl"
$logPath = Join-Path $Root "output-repaired\summary\beauty_relevance_evidence_full_pipeline.log"

function Show-JsonlStatus {
    param([string]$Path, [string]$Name, [int]$Expected)
    if (-not (Test-Path $Path)) {
        Write-Output "$Name: missing / expected=$Expected"
        return
    }
    $rows = Get-Content $Path | ForEach-Object {
        try { $_ | ConvertFrom-Json } catch { $null }
    } | Where-Object { $_ -ne $null }
    $count = ($rows | Measure-Object).Count
    $ok = ($rows | Where-Object { $_.parse_success -eq $true } | Measure-Object).Count
    $fail = $count - $ok
    Write-Output "$Name: rows=$count / expected=$Expected parse_success=$ok failed_or_parse_failed=$fail"
}

Set-Location $Root
Write-Output "== python processes =="
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, WorkingSet, StartTime, Path

Write-Output ""
Write-Output "== prediction progress =="
Show-JsonlStatus -Path $validPath -Name "valid" -Expected 5838
Show-JsonlStatus -Path $testPath -Name "test" -Expected 5838

Write-Output ""
Write-Output "== log tail =="
if (Test-Path $logPath) {
    Get-Content $logPath -Tail 30
} else {
    Write-Output "log missing: $logPath"
}
