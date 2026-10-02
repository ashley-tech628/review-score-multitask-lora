param(
    [string]$DataPath = 'D:\assignment2\ratebeer.json',
    [int]$Limit = 2000
)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$runPath = Join-Path $projectPath ('runs\lora-smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $DataPath)) { throw "Dataset not found: $DataPath" }
Set-Location -LiteralPath $projectPath
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $runtimeCandidates = @(
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe')
    )
    $runtimePath = $null
    foreach ($candidate in $runtimeCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { $runtimePath = $candidate; break }
        }
    }
    if (-not $runtimePath) { throw 'No runnable Python 3.10-3.13 found in known locations.' }
    Write-Host "Creating environment with: $runtimePath"
    & $runtimePath -m venv (Join-Path $projectPath '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating the project Python environment failed.' }
}
Invoke-CheckedPython -Arguments @('-m','pip','install','torch','--index-url','https://download.pytorch.org/whl/cpu')
Invoke-CheckedPython -Arguments @('-m','pip','install','-r','requirements-neural.txt')
# Explicitly require dependencies, so missing imports cannot count as skipped success.
Invoke-CheckedPython -Arguments @('-c','import torch, transformers, peft, safetensors; print(torch.__version__, transformers.__version__, peft.__version__)')
Invoke-CheckedPython -Arguments @('-m','unittest','discover','-s','tests','-v')
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$baseModel = 'distilbert-base-uncased'
$cachedSnapshots = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--distilbert-base-uncased\snapshots'
$cachedModel = Get-ChildItem -LiteralPath $cachedSnapshots -Directory -ErrorAction SilentlyContinue |
    Where-Object { (Test-Path -LiteralPath (Join-Path $_.FullName 'model.safetensors')) -and (Test-Path -LiteralPath (Join-Path $_.FullName 'tokenizer.json')) } |
    Select-Object -First 1
$trainingArguments = @('-m','reviewscore.train_neural','--data',$DataPath,'--base',$baseModel,'--out',$runPath,'--limit',"$Limit",'--epochs','1','--batch-size','8','--device','cpu')
if ($cachedModel) {
    $trainingArguments[5] = $cachedModel.FullName
    $trainingArguments += '--local-only'
}
Invoke-CheckedPython -Arguments $trainingArguments
Invoke-CheckedPython -Arguments @('-m','reviewscore.predict_neural','--bundle',(Join-Path $runPath 'bundle'),'--text','Floral aroma with a smooth texture and balanced finish')
Invoke-CheckedPython -Arguments @('-m','pip','freeze')
Write-Host "Completed tests, training, reload validation and prediction. Results: $runPath"
