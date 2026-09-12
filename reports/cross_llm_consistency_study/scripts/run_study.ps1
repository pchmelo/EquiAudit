# run_study.ps1 - EquiAudit consistency study batch runner
#
# Resumable: any run whose output file already exists is skipped automatically.
# Re-run the same command after a rate-limit pause and it picks up from where
# it left off without re-running completed experiments.
#
# Usage:
#   .\experiments\run_study.ps1
#   .\experiments\run_study.ps1 -Models @("config_qwen", "config_deepseek")
#   .\experiments\run_study.ps1 -Models @("config_openrouter") -Runs 10
#   .\experiments\run_study.ps1 -Runs 3   # quick smoke test

param(
    [string[]]$Models = @("config_qwen", "config_deepseek", "config_openrouter"),
    [string[]]$Datasets = @("adult-all.csv", "GermanCredit.csv"),
    [int]$Runs = 10,
    [string]$OutputDir = "experiments/results"
)

$ErrorActionPreference = "Continue"

# Silently load .env (project root or examples/) if key not already set
$_root = Split-Path $PSScriptRoot -Parent
foreach ($dotenvPath in @("$_root\.env", "$_root\examples\.env")) {
    if ((Test-Path $dotenvPath) -and (-not $env:OPENROUTER_API_KEY)) {
        Get-Content $dotenvPath | ForEach-Object {
            if ($_ -match '^\s*OPENROUTER_API_KEY\s*=\s*(.+)$') {
                $env:OPENROUTER_API_KEY = $Matches[1].Trim().Trim('"').Trim("'")
            }
        }
    }
}

$total   = $Models.Count * $Datasets.Count * $Runs
$done    = 0
$skipped = 0
$failed  = 0

Write-Host "========================================"
Write-Host "EquiAudit Consistency Study (resumable)"
Write-Host "Models:   $($Models -join ', ')"
Write-Host "Datasets: $($Datasets -join ', ')"
Write-Host "Runs:     $Runs per combination"
Write-Host "Total:    $total slots (skips existing files)"
Write-Host "========================================"

foreach ($model in $Models) {
    $configPath = "experiments/configs/$model.yml"

    if (-not (Test-Path $configPath)) {
        Write-Host "SKIP model: config not found: $configPath"
        $done += $Datasets.Count * $Runs
        continue
    }

    # Guard: OpenRouter needs an API key
    if ($model -like "*openrouter*") {
        if (-not $env:OPENROUTER_API_KEY) {
            Write-Host ""
            Write-Host "ERROR: OPENROUTER_API_KEY is not set. Set it with:"
            Write-Host "  `$env:OPENROUTER_API_KEY = 'sk-or-...'"
            Write-Host "Then re-run the script. Already-completed runs will be skipped."
            Write-Host ""
            $done += $Datasets.Count * $Runs
            continue
        }
    }

    $configStem = [System.IO.Path]::GetFileNameWithoutExtension($configPath)

    foreach ($dataset in $Datasets) {
        $datasetStem = [System.IO.Path]::GetFileNameWithoutExtension($dataset)

        for ($i = 1; $i -le $Runs; $i++) {
            $done++
            $runStr = $i.ToString("00")
            $expectedFile = "$OutputDir/${configStem}__${datasetStem}__run${runStr}.json"

            # Resume: skip runs whose output already exists
            if (Test-Path $expectedFile) {
                $skipped++
                $pct = [int](($done / $total) * 100)
                Write-Host "[$done/$total ($pct%)] SKIP (done): $configStem | $dataset | run $i"
                continue
            }

            $pct = [int](($done / $total) * 100)
            Write-Host "[$done/$total ($pct%)] $configStem | $dataset | run $i"

            python experiments/consistency_runner.py --config $configPath --dataset $dataset --run-id $i --output $OutputDir

            if ($LASTEXITCODE -ne 0) {
                $failed++
                Write-Host "  FAILED (exit $LASTEXITCODE) - continuing"
            }
        }
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "Done: $done slots, $skipped skipped, $failed failed"
Write-Host "Results saved to: $OutputDir"
Write-Host ""
Write-Host "Next step - run the analysis:"
Write-Host "  python experiments/analyze_consistency.py --results $OutputDir --csv experiments/summary.csv"
