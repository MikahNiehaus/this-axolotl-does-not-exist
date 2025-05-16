#!/usr/bin/env pwsh
# Script to verify model files are properly tracked in git

$modelDir = "backend/data"
$fullPath = Join-Path $PSScriptRoot $modelDir
$fullModelPath = Join-Path $fullPath "gan_full_model.pth"
$checkpointPath = Join-Path $fullPath "gan_checkpoint.pth"

Write-Host "===== Model File Verification ====="
Write-Host "Checking paths and git status..."

# Check if directory exists
if (Test-Path $fullPath) {
    Write-Host "✓ Data directory exists at: $fullPath" -ForegroundColor Green
} else {
    Write-Host "✗ Data directory NOT FOUND at: $fullPath" -ForegroundColor Red
    exit 1
}

# Check model files
Write-Host "`nModel Files:"
$modelFiles = @(
    @{ Path = $fullModelPath; Name = "Full Model" },
    @{ Path = $checkpointPath; Name = "Checkpoint" }
)

foreach ($file in $modelFiles) {
    if (Test-Path $file.Path) {
        $fileInfo = Get-Item $file.Path
        $size = "{0:N2} MB" -f ($fileInfo.Length / 1MB)
        $lastMod = $fileInfo.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "✓ $($file.Name) exists: $($file.Path)" -ForegroundColor Green
        Write-Host "  - Size: $size"
        Write-Host "  - Last Modified: $lastMod"
        
        # Check if file is tracked in git
        $gitStatus = git ls-files --error-unmatch $file.Path 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  - Git Status: TRACKED" -ForegroundColor Green
        } else {
            Write-Host "  - Git Status: NOT TRACKED" -ForegroundColor Red
            
            # Show gitignore status for this file
            Write-Host "`nChecking .gitignore status for $($file.Name):"
            git check-ignore -v $file.Path
            Write-Host ""
        }
    } else {
        Write-Host "✗ $($file.Name) NOT FOUND at: $($file.Path)" -ForegroundColor Red
    }
    Write-Host ""
}

# Check .gitignore configuration
Write-Host "Checking .gitignore configuration:"
$gitignoreContent = Get-Content (Join-Path $PSScriptRoot ".gitignore")
$modelRulesFound = $false

foreach ($line in $gitignoreContent) {
    if ($line -match "gan_full_model\.pth" -or 
        $line -match "gan_checkpoint\.pth" -or 
        ($line -match "\.pth" -and $line -match "!")) {
        Write-Host "  $line" -ForegroundColor Cyan
        $modelRulesFound = $true
    }
}

if (-not $modelRulesFound) {
    Write-Host "No specific model file rules found in .gitignore" -ForegroundColor Yellow
}

Write-Host "`nVerification complete."
