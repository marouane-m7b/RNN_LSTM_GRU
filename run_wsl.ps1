# Run this project's local .venv with GPU support through WSL.
$ErrorActionPreference = 'Stop'
if ($args.Count -eq 0) {
    Write-Output 'Usage: .\run_wsl.ps1 script.py [arguments]'
    exit 1
}
& wsl.exe -d Ubuntu-24.04 --cd $PSScriptRoot -- env 'UV_PROJECT_ENVIRONMENT=.venv' /home/marouane/.local/bin/uv run --offline --extra gpu python run_gpu.py @args
exit $LASTEXITCODE
