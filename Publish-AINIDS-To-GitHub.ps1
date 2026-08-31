[CmdletBinding()]
param(
    [string]$ProjectRoot = "C:\Users\esumi\AI-NIDS-Project",
    [string]$GitHubOwner = "sunestech",
    [string]$RepositoryName = "AI-NIDS-Project",
    [ValidateSet("private", "public")]
    [string]$Visibility = "private"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Require-Command {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is not installed. $InstallHint"
    }
}

Require-Command `
    -Name "git" `
    -InstallHint "Install Git for Windows, reopen PowerShell, and rerun this script."

Require-Command `
    -Name "gh" `
    -InstallHint "Run: winget install --id GitHub.cli --exact"

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project folder not found: $ProjectRoot"
}

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath ".\README.md" -PathType Leaf)) {
    throw "README.md is missing from the project root."
}

Write-Host ""
Write-Host "Project root: $ProjectRoot"
Write-Host "GitHub target: $GitHubOwner/$RepositoryName"
Write-Host "Initial visibility: $Visibility"
Write-Host ""

# Add safe publication exclusions without deleting an existing .gitignore.
$marker = "# AI-NIDS GitHub publish safeguards"
$ignoreBlock = @'
# AI-NIDS GitHub publish safeguards

# Python environments and caches
.venv/
venv/
env/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Editors and operating-system files
.vscode/
.idea/
*.swp
.DS_Store
Thumbs.db
desktop.ini

# Secrets and local configuration
.env
.env.*
!.env.example
.streamlit/secrets.toml
*.pem
*.key
*.pfx
*.p12

# Raw, large, or reproducible datasets
data/raw/
data/processed/
data/live_lab/raw/
*.pcap
*.pcapng
*.parquet

# Large generated prediction outputs; retain the small live-lab ledgers
reports/predictions/*
!reports/predictions/live_lab/
!reports/predictions/live_lab/benign_prediction_ledger.csv
!reports/predictions/live_lab/portscan_prediction_ledger.csv
!reports/predictions/live_lab/cross_flow_scan_alerts.csv

# Generated archives, logs, and temporary files
*.log
*.tmp
*.zip
*.tar
*.tar.gz
~$*
'@

if (-not (Test-Path -LiteralPath ".\.gitignore")) {
    Set-Content `
        -LiteralPath ".\.gitignore" `
        -Value $ignoreBlock `
        -Encoding utf8
}
elseif (-not (Select-String `
        -LiteralPath ".\.gitignore" `
        -SimpleMatch `
        -Pattern $marker `
        -Quiet)) {
    Add-Content `
        -LiteralPath ".\.gitignore" `
        -Value ("`r`n" + $ignoreBlock) `
        -Encoding utf8
}

if (-not (Test-Path -LiteralPath ".\.git" -PathType Container)) {
    git init
    if ($LASTEXITCODE -ne 0) {
        throw "git init failed."
    }
}

git branch -M main
if ($LASTEXITCODE -ne 0) {
    throw "Could not set the main branch."
}

$authorName = git config user.name
if ([string]::IsNullOrWhiteSpace([string]$authorName)) {
    $authorName = Read-Host "Enter the commit author name"
    if ([string]::IsNullOrWhiteSpace($authorName)) {
        throw "A Git commit author name is required."
    }
    git config user.name $authorName
}

$authorEmail = git config user.email
if ([string]::IsNullOrWhiteSpace([string]$authorEmail)) {
    $authorEmail = Read-Host "Enter a verified GitHub email or GitHub no-reply email"
    if ([string]::IsNullOrWhiteSpace($authorEmail)) {
        throw "A Git commit email is required."
    }
    git config user.email $authorEmail
}

# Inspect only files that Git would include.
$candidatePaths = @(
    git ls-files `
        --cached `
        --others `
        --exclude-standard
)

$largeFiles = @(
    foreach ($relativePath in $candidatePaths) {
        if ([string]::IsNullOrWhiteSpace($relativePath)) {
            continue
        }

        $fullPath = Join-Path $ProjectRoot $relativePath
        if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
            $item = Get-Item -LiteralPath $fullPath
            if ($item.Length -ge 95MB) {
                [pscustomobject]@{
                    File   = $relativePath
                    SizeMB = [math]::Round($item.Length / 1MB, 2)
                }
            }
        }
    }
)

if ($largeFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Files at or above 95 MB cannot be pushed normally:"
    $largeFiles | Format-Table -AutoSize
    throw "Exclude these files or configure Git LFS before continuing."
}

$sensitiveCandidates = @(
    $candidatePaths |
        Where-Object {
            $_ -match '(^|[\\/])\.env($|\.)' -or
            $_ -match 'secrets\.toml$' -or
            $_ -match '\.(pem|key|pfx|p12)$'
        }
)

if ($sensitiveCandidates.Count -gt 0) {
    Write-Host ""
    Write-Host "Potential sensitive files would be published:"
    $sensitiveCandidates
    throw "Remove or ignore the sensitive files before continuing."
}

git add .
if ($LASTEXITCODE -ne 0) {
    throw "git add failed."
}

$stagedFiles = @(
    git diff --cached --name-only --diff-filter=ACMR
)

if ($stagedFiles.Count -eq 0) {
    Write-Host "No new files need to be committed."
}
else {
    Write-Host ""
    Write-Host "Files staged for publication:" $stagedFiles.Count
    git diff --cached --stat

    git commit `
        -m "Initial release: hybrid AI-powered network intrusion detection system"

    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed."
    }
}

gh auth status *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Opening GitHub browser authentication..."
    gh auth login `
        --hostname github.com `
        --git-protocol https `
        --web

    if ($LASTEXITCODE -ne 0) {
        throw "GitHub authentication failed."
    }
}

$repoFullName = "$GitHubOwner/$RepositoryName"

gh repo view $repoFullName --json name *> $null
$repoExists = ($LASTEXITCODE -eq 0)

if ($repoExists) {
    Write-Host "Repository already exists: $repoFullName"

    git remote get-url origin *> $null
    if ($LASTEXITCODE -ne 0) {
        git remote add origin "https://github.com/$repoFullName.git"
    }

    git push -u origin main
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed."
    }
}
else {
    $visibilityFlag = "--$Visibility"

    gh repo create `
        $repoFullName `
        $visibilityFlag `
        --source . `
        --remote origin `
        --push `
        --description "Hybrid network intrusion detection prototype using machine learning, Suricata, MITRE ATT&CK, Streamlit, and cross-flow correlation."

    if ($LASTEXITCODE -ne 0) {
        throw "GitHub repository creation or push failed."
    }
}

$repoUrl = gh repo view $repoFullName --json url --jq ".url"

Write-Host ""
Write-Host "PASS: Project was pushed successfully."
Write-Host "Repository: $repoUrl"
Write-Host ""
Write-Host "The repository was created as '$Visibility'."
Write-Host "Review it before changing a private repository to public."
