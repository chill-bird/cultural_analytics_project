# Activate your virtual environment
& "C:\Users\melli\Workspace\narrative_detection\.venv\Scripts\Activate.ps1"

# Create logs folder if it doesn't exist
$logDir = "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir
}

# Loop over first 20 episodes
for ($SLURM_ARRAY_TASK_ID = 72; $SLURM_ARRAY_TASK_ID -le 237; $SLURM_ARRAY_TASK_ID++) {
    Write-Host "Processing episode $SLURM_ARRAY_TASK_ID"
    
    # Set the environment variable so Python sees it
    $env:SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID

    $startTime = Get-Date

    python detect_narratives.py --episode  $SLURM_ARRAY_TASK_ID > "logs\episode_ $SLURM_ARRAY_TASK_ID.out" 2> "logs\episode_ $SLURM_ARRAY_TASK_ID.err"

    $endTime = Get-Date
    $runtime = $endTime - $startTime
    Write-Host "Episode $SLURM_ARRAY_TASK_ID runtime: $($runtime.TotalSeconds) seconds"
}

Write-Host "All 20 episodes completed."