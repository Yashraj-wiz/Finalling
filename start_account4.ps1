# Script to run kimi_audio_7b on a FOURTH Modal account in parallel

param (
    [string]$TokenId = "ak-nvbslGa1ZrKRZAFqhmlXjl",
    [string]$TokenSecret = "as-ubVD3Lkaxph1woNf0pXTti",
    [string]$ModelToRun = "kimi_audio_7b"
)

if ($TokenId -match "REPLACE") {
    Write-Host "Please edit start_account4.ps1 to add the fourth person's Modal Token ID and Secret!" -ForegroundColor Red
    exit
}

# Override the Modal environment variables for this terminal session ONLY
$env:MODAL_TOKEN_ID = $TokenId
$env:MODAL_TOKEN_SECRET = $TokenSecret
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting Modal Inference for $ModelToRun on Account 4..." -ForegroundColor Green
Write-Host "A background job will sync Account 4's volume to modal_run_account4.log." -ForegroundColor Cyan

# Start the sync script as a background job, passing the tokens into the job's context
$SyncJob = Start-Job -ScriptBlock {
    param($TId, $TSec)
    $env:MODAL_TOKEN_ID = $TId
    $env:MODAL_TOKEN_SECRET = $TSec
    Set-Location "g:\IvLabs\Impact_Speech"
    .\sync_modal.ps1 -LogFile "modal_run_account4.log"
} -ArgumentList $TokenId, $TokenSecret

try {
    # Run kimi_audio_7b — all tasks
    modal run aip-speech/scripts/run_modal_inference.py --model $ModelToRun --task all
}
finally {
    Write-Host "`nInference finished or stopped. Stopping Account 4 sync loop..." -ForegroundColor Yellow
    Stop-Job $SyncJob
    Remove-Job $SyncJob

    Write-Host "Running final sync for Account 4... Please wait." -ForegroundColor Yellow
    modal volume get aip-inference-out / aip-speech/inference/ --force | Out-Null
    Write-Host "Account 4 All done! Results saved locally." -ForegroundColor Green
}
