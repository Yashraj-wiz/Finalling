param (
    [int]$IntervalSeconds = 300,  # Default to 5 minutes
    [string]$LogFile = "modal_run.log"
)

$LocalInferenceDir = "g:\IvLabs\Impact_Speech\aip-speech\inference\"
$ModalVolumePath = "aip-inference-out"

Write-Output "Starting Modal sync loop..."
Write-Output "Syncing every $IntervalSeconds seconds to $LocalInferenceDir"
Write-Output "Logging to $LogFile"
Write-Output "Press Ctrl+C to stop."
Write-Output "========================================"

# Make sure the local directory exists
if (-not (Test-Path -Path $LocalInferenceDir)) {
    New-Item -ItemType Directory -Path $LocalInferenceDir | Out-Null
}

while ($true) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    Write-Output "[$Timestamp] Syncing volume..."
    
    # Run the modal volume get command. 
    # Use / to pull all contents, and output to the local inference dir
    # Note: Using --force to overwrite local files with the latest from the volume
    $syncCmd = "modal volume get $ModalVolumePath / $LocalInferenceDir --force"
    Invoke-Expression $syncCmd
    
    # Calculate progress
    $ProgressSummary = ""
    if (Test-Path $LocalInferenceDir) {
        $Models = Get-ChildItem -Path $LocalInferenceDir -Directory
        foreach ($Model in $Models) {
            $JsonlFiles = Get-ChildItem -Path $Model.FullName -Filter "*.jsonl"
            foreach ($File in $JsonlFiles) {
                $Task = $File.BaseName
                if (Test-Path $File.FullName) {
                    $LineCount = (Get-Content $File.FullName | Measure-Object -Line).Lines
                    $Total = 6893
                    if ($Task -like "*asr*") { $Total = 6100 }
                    $Pct = [math]::Round(($LineCount / $Total * 100), 2)
                    $ProgressSummary += "`r`n    * $($Model.Name) / $($Task): $LineCount / $Total ($Pct%)"
                }
            }
        }
    }
    
    $Status = "[$Timestamp] Sync completed.$ProgressSummary"
    Write-Output $Status
    Add-Content -Path $LogFile -Value $Status
    
    Start-Sleep -Seconds $IntervalSeconds
}
