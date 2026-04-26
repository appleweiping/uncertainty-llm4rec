param(
    [string]$Python = "py -3.12",
    [int]$MaxWorkers = 4,
    [int]$RequestsPerMinute = 120
)

$ErrorActionPreference = "Stop"

$settings = @(
    @{ noise_type = "history_dropout"; noise_level = "0.1" },
    @{ noise_type = "history_dropout"; noise_level = "0.2" },
    @{ noise_type = "history_dropout"; noise_level = "0.3" },
    @{ noise_type = "candidate_text_dropout"; noise_level = "0.1" },
    @{ noise_type = "candidate_text_dropout"; noise_level = "0.2" },
    @{ noise_type = "candidate_text_dropout"; noise_level = "0.3" },
    @{ noise_type = "history_swap_noise"; noise_level = "0.1" },
    @{ noise_type = "history_swap_noise"; noise_level = "0.2" },
    @{ noise_type = "history_swap_noise"; noise_level = "0.3" }
)

$summaryDir = "output-repaired\summary"
$monitorPath = Join-Path $summaryDir "day30_cep_robustness_runtime_monitor.md"
$progressCsv = Join-Path $summaryDir "day30_cep_robustness_progress.csv"
New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null

function Get-LineCount {
    param([string]$Path)
    if (Test-Path $Path) {
        return (Get-Content $Path | Measure-Object -Line).Lines
    }
    return 0
}

function Get-ParseSuccessRate {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0.0
    }
    $total = 0
    $ok = 0
    Get-Content $Path | ForEach-Object {
        if (-not [string]::IsNullOrWhiteSpace($_)) {
            $total += 1
            try {
                $record = $_ | ConvertFrom-Json
                if ($record.parse_success -eq $true) {
                    $ok += 1
                }
            } catch {
            }
        }
    }
    if ($total -eq 0) {
        return 0.0
    }
    return [double]$ok / [double]$total
}

function Write-Monitor {
    param(
        [array]$Rows
    )
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $lines = @()
    $lines += "# Day30 CEP Robustness Runtime Monitor"
    $lines += ""
    $lines += "Last update: $now"
    $lines += ""
    $lines += "| noise_type | noise_level | expected_rows | current_rows | rolling_parse_success | status | resume_command |"
    $lines += "|---|---:|---:|---:|---:|---|---|"
    foreach ($row in $Rows) {
        $rate = "{0:N4}" -f ([double]$row.rolling_parse_success)
        $resume = $row.resume_command.Replace("|", "\|")
        $lines += "| $($row.noise_type) | $($row.noise_level) | $($row.expected_rows) | $($row.current_rows) | $rate | $($row.status) | ``$resume`` |"
    }
    Set-Content -Path $monitorPath -Value $lines -Encoding UTF8
    $Rows | Export-Csv -Path $progressCsv -NoTypeInformation
}

function Invoke-Cmd {
    param([string]$Command)
    Write-Host $Command
    $previousPreference = $ErrorActionPreference
    try {
        # Python/tqdm writes progress bars to stderr. Treat that as normal native
        # process output and rely on the actual exit code instead of PowerShell's
        # NativeCommandError wrapping.
        $ErrorActionPreference = "Continue"
        Invoke-Expression $Command
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: ${Command}"
    }
}

$rows = @()
foreach ($setting in $settings) {
    $noiseType = $setting.noise_type
    $noiseLevel = $setting.noise_level
    $settingName = "${noiseType}_${noiseLevel}"
    $config = "configs\exp\beauty_robustness_500_${noiseType}_${noiseLevel}.yaml"
    $smokeInput = "data\processed\amazon_beauty_robustness_500\noisy_${settingName}\smoke_20.jsonl"
    $smokeOutput = "output-repaired\beauty_robustness_500\${settingName}\predictions\smoke_verify_raw.jsonl"
    $testOutput = "output-repaired\beauty_robustness_500\${settingName}\predictions\test_raw.jsonl"
    $logPath = "output-repaired\summary\day30_${settingName}_inference.log"
    $errPath = "output-repaired\summary\day30_${settingName}_inference.err.log"
    $resumeCommand = "$Python main_infer.py --config $config --split_name test --concurrent --resume --max_workers $MaxWorkers --requests_per_minute $RequestsPerMinute"

    $row = [pscustomobject]@{
        noise_type = $noiseType
        noise_level = $noiseLevel
        expected_rows = 3000
        current_rows = Get-LineCount $testOutput
        rolling_parse_success = Get-ParseSuccessRate $testOutput
        status = "pending"
        resume_command = $resumeCommand
        last_update_time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
    $rows += $row
    Write-Monitor -Rows $rows

    $predDir = "output-repaired\beauty_robustness_500\${settingName}\predictions"
    $successfulSmoke = $null
    if (Test-Path $predDir) {
        Get-ChildItem $predDir -Filter "smoke_verify*_raw.jsonl" -File | Sort-Object LastWriteTime -Descending | ForEach-Object {
            if ($null -eq $successfulSmoke) {
                $candidateRows = Get-LineCount $_.FullName
                $candidateRate = Get-ParseSuccessRate $_.FullName
                if (($candidateRows -ge 20) -and ($candidateRate -ge 0.95)) {
                    $successfulSmoke = $_.FullName
                }
            }
        }
    }
    if ($null -ne $successfulSmoke) {
        $smokeOutput = $successfulSmoke
    }
    $smokeRows = Get-LineCount $smokeOutput
    $smokeRate = Get-ParseSuccessRate $smokeOutput
    if (($smokeRows -ge 20) -and ($smokeRate -lt 0.95)) {
        $stamp = Get-Date -Format "yyyyMMddHHmmss"
        $smokeOutput = "output-repaired\beauty_robustness_500\${settingName}\predictions\smoke_verify_${stamp}_raw.jsonl"
        $smokeRows = 0
        $smokeRate = 0.0
    }
    if (($smokeRows -lt 20) -or ($smokeRate -lt 0.95)) {
        $row.status = "running_smoke"
        Write-Monitor -Rows $rows
        $smokeCommand = "$Python main_infer.py --config $config --input_path $smokeInput --output_path $smokeOutput --split_name smoke --concurrent --resume --max_workers $MaxWorkers --requests_per_minute $RequestsPerMinute"
        Invoke-Cmd $smokeCommand
        $smokeRows = Get-LineCount $smokeOutput
        $smokeRate = Get-ParseSuccessRate $smokeOutput
        if (($smokeRows -lt 20) -or ($smokeRate -lt 0.95)) {
            $row.current_rows = Get-LineCount $testOutput
            $row.rolling_parse_success = Get-ParseSuccessRate $testOutput
            $row.status = "blocked_smoke_parse_success_${smokeRate}_rows_${smokeRows}"
            Write-Monitor -Rows $rows
            throw "Smoke failed for ${settingName}: rows=$smokeRows parse_success=$smokeRate"
        }
    }

    $row.status = "running_full"
    Write-Monitor -Rows $rows
    $fullCommand = "$resumeCommand *> $logPath"
    # Keep stderr/stdout in one transcript because tqdm uses stderr heavily.
    Invoke-Cmd $fullCommand
    $row.current_rows = Get-LineCount $testOutput
    $row.rolling_parse_success = Get-ParseSuccessRate $testOutput
    if ($row.current_rows -ge 3000 -and $row.rolling_parse_success -ge 0.95) {
        $row.status = "complete"
    } else {
        $row.status = "incomplete_rows_$($row.current_rows)_parse_$($row.rolling_parse_success)"
        Write-Monitor -Rows $rows
        throw "Full inference incomplete for ${settingName}"
    }
    $row.last_update_time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Monitor -Rows $rows
}

$analysisCommand = "$Python main_day30_cep_robustness.py --analyze *> output-repaired\summary\day30_cep_robustness_analysis.log"
Invoke-Cmd $analysisCommand
Write-Monitor -Rows $rows
