param(
    [string]$ProgressPath = "outputs/summary/week7_8_replay_v2_runtime_progress.csv",
    [string]$SummaryPath = "outputs/summary/week7_8_replay_v2_runtime_materialization.csv",
    [string]$Domain = ""
)

while ($true) {
    Clear-Host
    Get-Date

    Write-Host "`n[progress csv]"
    if (Test-Path $ProgressPath) {
        $progress = Import-Csv $ProgressPath
        if ($Domain) {
            $progress = $progress | Where-Object { $_.domain -like "*$Domain*" }
        }
        $progress | Select-Object -Last 12 | Format-Table -AutoSize
    } else {
        Write-Host "No progress file yet: $ProgressPath"
    }

    Write-Host "`n[summary csv]"
    if (Test-Path $SummaryPath) {
        Import-Csv $SummaryPath | Format-Table -AutoSize
    } else {
        Write-Host "No summary file yet: $SummaryPath"
    }

    if ($Domain) {
        $domainDir = Join-Path "data/processed" $Domain
        if (Test-Path $domainDir) {
            Write-Host "`n[$domainDir files]"
            Get-ChildItem $domainDir | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
        }
    }

    Start-Sleep -Seconds 20
}
