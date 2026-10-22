param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe" }
if (-not (Test-Path $python)) { throw "Run setup.ps1 first" }
& $python -m vqa_lab @Arguments
exit $LASTEXITCODE
