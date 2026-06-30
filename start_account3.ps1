# Script to run gemma3n_e4b on a THIRD Modal account in parallel

param (
    [string]$TokenId = "ak-dL0hkXjh1R9668mU4LTbtn",
    [string]$TokenSecret = "as-4EAZZRfy6jArEllMPGDz3Q",
    [string]$HfToken = "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",
    [string]$ModelToRun = "gemma3n_e4b"
)

if ($TokenId -match "REPLACE" -or $HfToken -match "REPLACE") {
    Write-Host "Please edit start_account3.ps1 to add the third person's Modal Token ID, Secret, and Hugging Face Token!" -ForegroundColor Red
    exit
}

# Override the Modal environment variables for this terminal session ONLY
$env:MODAL_TOKEN_ID = $TokenId
$env:MODAL_TOKEN_SECRET = $TokenSecret
if ($HfToken -and $HfToken -notmatch "REPLACE") {
    $env:HF_TOKEN = $HfToken
}
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting Modal Inference for $ModelToRun on Account 3..." -ForegroundColor Green
Write-Host "A background job will sync Account 3's volume to modal_run_account3.log." -ForegroundColor Cyan

# Start the sync script as a background job, passing the tokens into the job's context
$SyncJob = Start-Job -ScriptBlock {
    param($TId, $TSec)
    $env:MODAL_TOKEN_ID = $TId
    $env:MODAL_TOKEN_SECRET = $TSec
    Set-Location "g:\IvLabs\Impact_Speech"
    .\sync_modal.ps1 -LogFile "modal_run_account3.log"
} -ArgumentList $TokenId, $TokenSecret

try {
    # Run gemma3n_e4b — all tasks
    modal run aip-speech/scripts/run_modal_inference.py --model $ModelToRun --task all
}
finally {
    Write-Host "`nInference finished or stopped. Stopping Account 3 sync loop..." -ForegroundColor Yellow
    Stop-Job $SyncJob
    Remove-Job $SyncJob

    Write-Host "Running final sync for Account 3... Please wait." -ForegroundColor Yellow
    modal volume get aip-inference-out / aip-speech/inference/ --force | Out-Null
    Write-Host "Account 3 All done! Results saved locally." -ForegroundColor Green
}
