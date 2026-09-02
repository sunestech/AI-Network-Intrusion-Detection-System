param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,

    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Find-ProjectRoot {
    $candidates = @(
        $PSScriptRoot,
        "C:\Users\esumi\AI-NIDS-Project",
        (Get-Location).Path
    ) | Select-Object -Unique

    foreach ($candidate in $candidates) {
        if (
            -not [string]::IsNullOrWhiteSpace($candidate) -and
            (Test-Path -LiteralPath (Join-Path $candidate "dashboard.py") -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $candidate ".venv\Scripts\python.exe") -PathType Leaf)
        ) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw @"
The AI-NIDS project could not be found.

Expected project:
C:\Users\esumi\AI-NIDS-Project

The folder must contain:
  dashboard.py
  .venv\Scripts\python.exe
"@
}

$ProjectRoot = Find-ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Dashboard = Join-Path $ProjectRoot "dashboard.py"
$InferenceModule = Join-Path $ProjectRoot "src\dashboard_inference.py"
$CrossFlowPage = Join-Path $ProjectRoot "pages\4_Cross_Flow_Correlation.py"
$LocalUrl = "http://localhost:$Port"

$requiredFiles = @(
    $Python,
    $Dashboard,
    $InferenceModule
)

$missingFiles = @(
    $requiredFiles |
    Where-Object {
        -not (Test-Path -LiteralPath $_ -PathType Leaf)
    }
)

if ($missingFiles.Count -gt 0) {
    Write-Host "Missing required files:" -ForegroundColor Red
    $missingFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    throw "The dashboard cannot start until the missing files are restored."
}

if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
    $listener = Get-NetTCPConnection `
        -State Listen `
        -LocalPort $Port `
        -ErrorAction SilentlyContinue

    if ($listener) {
        throw @"
Port $Port is already in use.

Run the launcher on another port, for example:
.\Start-AI-NIDS-Dashboard.ps1 -Port 8502
"@
    }
}

Set-Location $ProjectRoot

Write-Host ""
Write-Host "AI-NIDS Streamlit Dashboard Launcher" -ForegroundColor Cyan
Write-Host ("=" * 72)
Write-Host "Project root : $ProjectRoot"
Write-Host "Python       : $Python"
Write-Host "Dashboard    : $Dashboard"
Write-Host "Local URL    : $LocalUrl"

if (Test-Path -LiteralPath $CrossFlowPage -PathType Leaf) {
    Write-Host "Cross-flow   : Available"
}
else {
    Write-Warning "The Cross Flow Correlation page was not found."
}

Write-Host ""
Write-Host "Checking Python packages and dashboard syntax..."

& $Python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit could not be imported from the project virtual environment."
}

$compileFiles = @(
    $Dashboard,
    $InferenceModule
)

if (Test-Path -LiteralPath $CrossFlowPage -PathType Leaf) {
    $compileFiles += $CrossFlowPage
}

& $Python -m py_compile @compileFiles
if ($LASTEXITCODE -ne 0) {
    throw "A dashboard Python file failed the syntax check."
}

Write-Host "PASS: Environment and syntax checks completed." -ForegroundColor Green
Write-Host ""
Write-Host "Starting Streamlit..."
Write-Host "Press Ctrl+C in this terminal to stop the dashboard."
Write-Host ""

$streamlitArguments = @(
    "-m",
    "streamlit",
    "run",
    $Dashboard,
    "--server.address",
    "localhost",
    "--server.port",
    "$Port",
    "--browser.gatherUsageStats",
    "false"
)

if ($NoBrowser) {
    $streamlitArguments += @(
        "--server.headless",
        "true"
    )
}
else {
    $streamlitArguments += @(
        "--server.headless",
        "false"
    )
}

& $Python @streamlitArguments

if ($LASTEXITCODE -ne 0) {
    throw "Streamlit exited with code $LASTEXITCODE."
}
