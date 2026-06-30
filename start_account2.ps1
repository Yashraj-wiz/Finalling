# Script to run a different model on a SECOND Modal account in parallel

param (
    [string]$TokenId = "ak-tYLmv1K8Wmeg45V2rCLI53",
    [string]$TokenSecret = "as-HiqPSjcOrfFA2XaJ4Y8pdJ",
    [string]$ModelToRun = "phi4_multimodal" # <-- Set to a model that Account 1 hasn't reached yet
)

if ($TokenId -match "REPLACE") {
    Write-Host "Please edit start_account2.ps1 to add the second person's Modal Token ID and Secret!" -ForegroundColor Red
    exit
}

# Override the Modal environment variables for this terminal session ONLY
$env:MODAL_TOKEN_ID = $TokenId
$env:MODAL_TOKEN_SECRET = $TokenSecret
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting Modal Inference for $ModelToRun on Account 2..." -ForegroundColor Green
Write-Host "A background job will sync Account 2's volume to modal_run_account2.log." -ForegroundColor Cyan

# Start the sync script as a background job, passing the tokens into the job's context
$SyncJob = Start-Job -ScriptBlock {
    param($TId, $TSec)
    $env:MODAL_TOKEN_ID = $TId
    $env:MODAL_TOKEN_SECRET = $TSec
    Set-Location "g:\IvLabs\Impact_Speech"
    .\sync_modal.ps1 -LogFile "modal_run_account2.log"
} -ArgumentList $TokenId, $TokenSecret

try {
    # Run the main modal inference for ONE SPECIFIC MODEL to avoid overlap with Account 1
    modal run aip-speech/scripts/run_modal_inference.py --model $ModelToRun --task all
}
finally {
    Write-Host "`nInference finished or stopped. Stopping Account 2 sync loop..." -ForegroundColor Yellow
    Stop-Job $SyncJob
    Remove-Job $SyncJob
    
    Write-Host "Running final sync for Account 2... Please wait." -ForegroundColor Yellow
    modal volume get aip-inference-out / aip-speech/inference/ --force | Out-Null
    Write-Host "Account 2 All done! Results saved locally." -ForegroundColor Green
}
