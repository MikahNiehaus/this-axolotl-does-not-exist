# Test the Git model handler
param(
    [string]$ModelPath = "data/gan_checkpoint.pth",
    [string]$Branch = "main"
)

# Change to the backend directory
Push-Location $PSScriptRoot\backend

Write-Host "Testing Git model handler..." -ForegroundColor Cyan

# Run the test script
python models/test_git_handler.py --model $ModelPath --branch $Branch

# Return to the original directory
Pop-Location
