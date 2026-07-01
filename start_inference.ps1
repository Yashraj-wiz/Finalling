# Start inference and sync loop in a single command

Write-Host "Starting Modal Inference..." -ForegroundColor Green
Write-Host "A background job will sync the volume every 5 minutes to modal_run.log." -ForegroundColor Cyan

# Start the sync script as a background job
$SyncJob = Start-Job -ScriptBlock {
    # We navigate to the right directory and run the sync script
    Set-Location "g:\IvLabs\Impact_Speech"
    .\sync_modal.ps1
}

try {
    # Run the main modal inference
    $env:PYTHONIOENCODING="utf-8"
    $QwenModels = @("qwen25_omni_3b", "qwen25_omni_7b", "qwen2_audio_7b")
    foreach ($m in $QwenModels) {
        Write-Host "Running model: $m" -ForegroundColor Cyan
        modal run aip-speech/scripts/run_modal_inference.py --model $m --task all
    }
}
finally {
    # Ensure we clean up the background job when inference stops (even if interrupted)
    Write-Host "`nInference finished or stopped. Stopping sync loop..." -ForegroundColor Yellow
    Stop-Job $SyncJob
    Remove-Job $SyncJob
    
    # Do one final guaranteed sync
    Write-Host "Running final sync to pull latest results... Please wait." -ForegroundColor Yellow
    modal volume get aip-inference-out / aip-speech/inference/ --force | Out-Null
    Write-Host "All done! Results saved locally." -ForegroundColor Green
}
