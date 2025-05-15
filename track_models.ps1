# PowerShell script to ensure model files are tracked by Git

# List of model files to track
$modelFiles = @(
    "backend/data/gan_checkpoint.pth",
    "backend/data/best_diffusion_model.pth"
)

# Check if Git LFS is installed
try {
    $gitLfsVersion = git lfs version 2>$null
    Write-Host "Git LFS is installed: $gitLfsVersion"
}
catch {
    Write-Host "Warning: Git LFS is not installed. It's recommended for tracking large model files." -ForegroundColor Yellow
    Write-Host "You can install it from https://git-lfs.github.com/" -ForegroundColor Yellow
}

# Check each model file
foreach ($file in $modelFiles) {
    if (Test-Path $file) {
        # If the file exists but is not tracked, track it
        try {
            $null = git ls-files --error-unmatch $file 2>$null
            $isTracked = $true
        } 
        catch {
            $isTracked = $false
        }

        if (-not $isTracked) {
            Write-Host "Tracking model file: $file" -ForegroundColor Green
            git add $file
        }
        else {
            # Check if file is modified
            $status = git status --porcelain $file
            if ($status -match "^\s*M") {
                Write-Host "Model file changed: $file (staging now)" -ForegroundColor Cyan
                git add $file
            }
            else {
                Write-Host "Model file $file is already tracked and unchanged" -ForegroundColor Gray
            }
        }
    }
    else {
        Write-Host "Warning: Model file $file not found" -ForegroundColor Yellow
    }
}

Write-Host "`nAll model files processed. They should now be tracked by Git.`n" -ForegroundColor Green
