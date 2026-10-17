# Runs every experiment of the project locally (seed 0 for everything first).
# Usage:  powershell -File run_all.ps1 [-Python path\to\python.exe]
param([string]$Python = "python")
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force results | Out-Null

$runs = @()
foreach ($seed in 0, 1, 2) {
    foreach ($m in 'perframe', 'late', 'early', '3d_scratch', '3d_inflated') {
        $runs += , @($m, 'leaky', $seed)
        $runs += , @($m, 'noleak', $seed)
    }
    $runs += , @('flow', 'noleak', $seed)
}

foreach ($r in $runs) {
    $name = "$($r[1])_$($r[0])_s$($r[2])"
    if (Test-Path "results/runs/$name/test_probs.npy") { Write-Host "skip $name"; continue }
    & $Python train.py --model $r[0] --dataset $r[1] --seed $r[2] *>> "results/log.txt"
}
Write-Host "ALL DONE"
