$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 required" }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -e .
if ($LASTEXITCODE -ne 0) { throw "Dependencies install failed" }
