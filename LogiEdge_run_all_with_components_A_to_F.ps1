<#
.SYNOPSIS
    LogiEdge complete end-to-end validation pipeline (Tasks 1-32).

.DESCRIPTION
    Executes the complete LogiEdge assignment pipeline:

    Assignment terminology mapping (comments only):
      Component A - System Architecture and Deployment Justification [Module 1]
        Task A1 - Constraint Analysis
        Task A2 - System Architecture Diagram
      Component B - Hardware Selection and Justification [Module 2]
        Task B1 - Constraint Triangle Application
        Task B2 - Arithmetic Intensity and Roofline Analysis
      Component C - Sensor Pipeline and MQTT Architecture [Module 3]
        Task C1 - Sensor Simulator
        Task C2 - Preprocessing Pipeline
        Task C3 - Data Fusion Justification
      Component D - Model Training, Conversion, and Docker Deployment [Module 4]
        Task D1 - Dataset Generation and Model Training
        Task D2 - Docker Containerisation and OTA Demo
        Task D3 - The 10-Stage Pipeline Mapping
      Component E - Edge MLOps: Monitoring and Deployment Management [Module 5]
        Task E1 - PSI Drift Monitoring
        Task E2 - Ansible Deployment Playbook
        Task E3 - OTA Strategy Selection
      Component F - Model Optimisation [Module 6]
        Task F1 - Three Model Variants
        Task F2 - Five-Metric Benchmarking
        Task F3 - Deployment Recommendation

    run_all.ps1 execution-task split:
      Pipeline Task 1       - Shared prerequisite validation for Components C-F
      Pipeline Tasks 2-5    - Component D, Task D1
      Pipeline Tasks 6-7    - Component F, Task F1
      Pipeline Task 8       - Component C, Task C2 mandatory normalisation experiment
      Pipeline Tasks 9-11   - Component F, Tasks F2-F3
      Pipeline Tasks 12-14  - Component E, Task E1
      Pipeline Task 15      - Cross-component automated verification
      Pipeline Task 16      - Components A-B and report calculations/diagrams for A1, A2, B1, B2, C3, D3 and E3
      Pipeline Task 17      - Runtime preparation supporting Components C-E
      Pipeline Tasks 18-19  - Component D, Task D2 Docker containerisation and OTA layer-cache demo
      Pipeline Tasks 20-26  - Component C, Tasks C1-C2, plus offline-first runtime evidence for Component D
      Pipeline Tasks 27-30  - Component E, Task E2
      Pipeline Tasks 31-32  - Final verification across Components A-F

      Tasks 1-3   Validate prerequisites and generate datasets
      Tasks 4-7   Train M1 and generate M2/M3 optimized models
      Tasks 8-11  Run normalization and benchmark validation
      Tasks 12-14 Build and validate PSI monitoring evidence
      Tasks 15-16 Run tests, diagrams, evidence figures and calculations
      Task 17      Prepare inference context and runtime files
      Tasks 18-21 Build and start the Docker runtime
      Tasks 22-26 Monitor MQTT, simulate, store offline and replay
      Tasks 27-30 Populate registry and run Ansible deployment
      Tasks 31-32 Perform final verification and report the result

    Run from:
      C:\LogiEdge\LogiEdge_Complete_Assignment_Package

.EXAMPLE
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    .\run_all.ps1

.EXAMPLE
    .\run_all.ps1 -SkipSimulation -SkipAnsible

.NOTES
    The script uses the actual broker container names from docker-compose:
      logiedge-local-broker
      logiedge-uplink-broker

    For localhost deployment it replaces:
      registry.freightbridge.local:5000
    with:
      localhost:5000
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = "C:\LogiEdge\LogiEdge_Complete_Assignment_Package",
    [string]$RuntimeDir = "C:\LogiEdge\runtime",
    [string]$TruckId = "TRK-01",
    [int]$SimulationDuration = 180,
    [int]$OfflineSimulationDuration = 120,
    [int]$SimulationSpeed = 5,
    [switch]$SkipSimulation,
    [switch]$SkipAnsible,
    [switch]$SkipOtaBuild,
    [switch]$SkipTraining,
    [switch]$SkipTests,
    [switch]$NoRegistryReplacement,
    [switch]$SubmissionValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Reduce non-error TensorFlow/TFLite informational messages.
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

# -------------------------------------------------------------------------
# Strict final-submission validation mode
# -------------------------------------------------------------------------
if (
    $SubmissionValidation -and
    (
        $SkipSimulation -or
        $SkipAnsible -or
        $SkipOtaBuild -or
        $SkipTraining -or
        $SkipTests
    )
) {
    throw "SubmissionValidation cannot be used with any Skip switch."
}

$ComposeFile       = Join-Path $ProjectRoot "demo\docker-compose.yml"
$InferenceDir      = Join-Path $ProjectRoot "inference"
$DeploymentDir     = Join-Path $ProjectRoot "deployment"
$InventoryFile     = Join-Path $DeploymentDir "inventory.ini"
$PlaybookFile      = Join-Path $DeploymentDir "logibridge_deploy.yml"
$EvidenceDir       = Join-Path $ProjectRoot "evidence"
$PipelineLog       = Join-Path $EvidenceDir "run_all_tasks_01_32.log"

$ModelSource       = Join-Path $ProjectRoot "training\models\m3_pruned_int8.tflite"
$StatsSource       = Join-Path $ProjectRoot "data_pipeline\training_stats.npy"
$PsiSource         = Join-Path $ProjectRoot "monitoring\reference_dist.json"
$OtaModelSource    = Join-Path $ProjectRoot "training\models\m3_pruned_int8_ota.tflite"
$DockerModel       = Join-Path $InferenceDir "model.tflite"
$OtaBuildLog       = Join-Path $EvidenceDir "manual_task19_ota_cache_build.log"
$OptimisationMetadataPath = Join-Path $ProjectRoot "optimisation\results\model_optimisation_metadata.json"
$PsiMetadataPath   = Join-Path $ProjectRoot "monitoring\psi_metadata.json"

$RuntimeModel      = Join-Path $RuntimeDir "model.tflite"
$RuntimeStats      = Join-Path $RuntimeDir "training_stats.npy"
$RuntimePsi        = Join-Path $RuntimeDir "reference_dist.json"
$RuntimeDatabase   = Join-Path $RuntimeDir "alerts.db"

$ImageV1           = "logibridge/inference:v1"
$ImageOta          = "logibridge/inference:ota-model-only"
$RegistryImage     = "localhost:5000/logibridge/inference:v2"

$InferenceContainer = "logibridge-inference"
$LocalBroker         = "logiedge-local-broker"
$UplinkBroker        = "logiedge-uplink-broker"
$RegistryContainer   = "logibridge-registry"

$script:Passed = 0
$script:Warnings = 0
$script:CurrentTask = ""

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 88) -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("=" * 88) -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] $Message" -ForegroundColor Yellow
}

function Write-Pass {
    param([string]$Message)
    $script:Passed++
    Write-Host "[PASS] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    $script:Warnings++
    Write-Warning $Message
}

function Stop-Pipeline {
    param([string]$Message)
    throw "[$($script:CurrentTask)] $Message"
}

function Assert-Path {
    param(
        [Parameter(Mandatory)][string]$Path,
        [string]$Description = $Path,
        [ValidateSet("Leaf","Container","Any")][string]$PathType = "Any"
    )

    $exists = switch ($PathType) {
        "Leaf"      { Test-Path -LiteralPath $Path -PathType Leaf }
        "Container" { Test-Path -LiteralPath $Path -PathType Container }
        default     { Test-Path -LiteralPath $Path }
    }

    if (-not $exists) {
        Stop-Pipeline "Required $Description was not found: $Path"
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [string]$Description = "$FilePath $($Arguments -join ' ')",
        [switch]$AllowFailure,
        [switch]$CaptureOutput
    )

    Write-Step $Description

    if ($CaptureOutput) {
        $output = & $FilePath @Arguments 2>&1 | ForEach-Object { "$_" }
        $exitCode = $LASTEXITCODE
        $output | ForEach-Object { Write-Host $_ }

        if (($exitCode -ne 0) -and (-not $AllowFailure)) {
            Stop-Pipeline "Command failed with exit code ${exitCode}: $Description"
        }

        return [pscustomobject]@{
            ExitCode = $exitCode
            Output   = $output
        }
    }

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE

    if (($exitCode -ne 0) -and (-not $AllowFailure)) {
        Stop-Pipeline "Command failed with exit code ${exitCode}: $Description"
    }

    return $exitCode
}


function Invoke-LoggedNative {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$LogPath,
        [string]$Description = "$FilePath $($Arguments -join ' ')"
    )

    Write-Step $Description

    $parent = Split-Path -Parent $LogPath
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    # TensorFlow Lite can write harmless INFO messages to stderr. With the global
    # ErrorActionPreference set to Stop, PowerShell may otherwise treat those
    # messages as terminating errors before LASTEXITCODE can be inspected.
    $previousErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"

        $output = @(
            & $FilePath @Arguments 2>&1 |
                ForEach-Object { "$_" }
        )

        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $output | Set-Content -LiteralPath $LogPath -Encoding UTF8
    $output | ForEach-Object { Write-Host $_ }

    if ($exitCode -ne 0) {
        Stop-Pipeline "Command failed with exit code ${exitCode}: $Description. Log: $LogPath"
    }

    return $output
}

function Assert-FilePattern {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][string]$Filter,
        [Parameter(Mandatory)][string]$Description
    )

    $match = Get-ChildItem `
        -LiteralPath $Directory `
        -File `
        -Filter $Filter `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if ($null -eq $match) {
        Stop-Pipeline "$Description was not generated in $Directory using pattern $Filter"
    }

    Write-Pass "$Description found: $($match.Name)"
    return $match.FullName
}

function Get-BenchmarkM3 {
    $benchmarkPath = Join-Path $ProjectRoot "optimisation\results\benchmark_results.csv"
    Assert-Path -Path $benchmarkPath -Description "benchmark results CSV" -PathType Leaf

    $rows = Import-Csv -LiteralPath $benchmarkPath
    $m3 = $rows | Where-Object { $_.variant -like "M3*" } | Select-Object -First 1

    if ($null -eq $m3) {
        Stop-Pipeline "M3 benchmark row was not found in $benchmarkPath"
    }

    return $m3
}

function Get-ContainerRunning {
    param([Parameter(Mandatory)][string]$Name)

    $result = & docker inspect `
        --format "{{.State.Running}}" `
        $Name 2>$null

    return ($LASTEXITCODE -eq 0 -and "$result".Trim() -eq "true")
}

function Get-ContainerHealth {
    param([Parameter(Mandatory)][string]$Name)

    $health = & docker inspect `
        --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}" `
        $Name 2>$null

    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }

    return "$health".Trim()
}

function Wait-Container {
    param(
        [Parameter(Mandatory)][string]$Name,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        if (Get-ContainerRunning -Name $Name) {
            $health = Get-ContainerHealth -Name $Name

            if ($health -in @("healthy", "none")) {
                Write-Pass "Container '$Name' is running; health=$health"
                return
            }

            if ($health -eq "unhealthy") {
                & docker logs $Name --tail 100
                Stop-Pipeline "Container '$Name' became unhealthy."
            }
        }

        Start-Sleep -Seconds 2
    }

    & docker ps -a --filter "name=$Name"
    & docker logs $Name --tail 100 2>$null
    Stop-Pipeline "Container '$Name' did not become ready within $TimeoutSeconds seconds."
}

function Get-SimulatorPath {
    $preferred = @(
        (Join-Path $ProjectRoot "data_pipeline\simulator.py"),
        (Join-Path $ProjectRoot "simulation\sensor_simulator.py"),
        (Join-Path $ProjectRoot "demo\sensor_simulator.py"),
        (Join-Path $ProjectRoot "simulation\mqtt_sensor_simulator.py"),
        (Join-Path $ProjectRoot "demo\sensor_publisher.py")
    )

    foreach ($path in $preferred) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            return $path
        }
    }

    $candidate = Get-ChildItem `
        -Path $ProjectRoot `
        -Recurse `
        -File `
        -Include "*simulator*.py","*sensor*publisher*.py","*mqtt*publisher*.py" `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if ($null -ne $candidate) {
        return $candidate.FullName
    }

    return $null
}

function Get-PythonExecutable {
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        return $venvPython
    }

    $command = Get-Command python -ErrorAction SilentlyContinue

    if ($null -eq $command) {
        Stop-Pipeline "Python was not found. Activate .venv or install Python."
    }

    return $command.Source
}

function Invoke-Simulation {
    param(
        [Parameter(Mandatory)][string]$SimulatorPath,
        [Parameter(Mandatory)][int]$Duration,
        [switch]$UseSpeed
    )

    $python = Get-PythonExecutable

    $helpResult = Invoke-Native `
        -FilePath $python `
        -Arguments @($SimulatorPath, "--help") `
        -Description "Inspect simulator command-line options" `
        -AllowFailure `
        -CaptureOutput

    $helpText = $helpResult.Output -join "`n"

    $args = @($SimulatorPath)

    if ($helpText -match "--truck-id") {
        $args += @("--truck-id", $TruckId)
    }
    elseif ($helpText -match "--truck_id") {
        $args += @("--truck_id", $TruckId)
    }
    else {
        Stop-Pipeline "Simulator exists but does not expose --truck-id or --truck_id: $SimulatorPath"
    }

    if ($helpText -match "--anomaly") {
        $args += @("--anomaly", "combined")
    }

    if ($helpText -match "--duration") {
        $args += @("--duration", "$Duration")
    }
    else {
        Stop-Pipeline "Simulator does not expose --duration: $SimulatorPath"
    }

    if ($UseSpeed -and $helpText -match "--speed") {
        $args += @("--speed", "$SimulationSpeed")
    }

    Invoke-Native `
        -FilePath $python `
        -Arguments $args `
        -Description "Publish simulated sensor data for truck $TruckId"
}

function Invoke-SqlScalar {
    param([Parameter(Mandatory)][string]$Query)

    if (-not (Test-Path -LiteralPath $RuntimeDatabase -PathType Leaf)) {
        return $null
    }

    $python = Get-PythonExecutable
    $tempScript = Join-Path ([System.IO.Path]::GetTempPath()) ("logiedge_sql_scalar_{0}.py" -f ([guid]::NewGuid().ToString("N")))

    $pythonCode = @'
import os
import sqlite3
import sys

try:
    database_path = os.environ["LOGIEDGE_SQLITE_DB"]
    query = os.environ["LOGIEDGE_SQL_QUERY"]
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(query).fetchone()
    if row is not None and len(row) > 0 and row[0] is not None:
        print(row[0])
except Exception as exc:
    print(f"SQLite query failed: {exc}", file=sys.stderr)
    sys.exit(1)
'@

    try {
        Set-Content -LiteralPath $tempScript -Value $pythonCode -Encoding UTF8

        $previousDb = $env:LOGIEDGE_SQLITE_DB
        $previousQuery = $env:LOGIEDGE_SQL_QUERY
        $env:LOGIEDGE_SQLITE_DB = $RuntimeDatabase
        $env:LOGIEDGE_SQL_QUERY = $Query

        $result = & $python $tempScript 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            $result | ForEach-Object { Write-Warning "$_" }
            return $null
        }

        return (($result | ForEach-Object { "$_" }) -join "`n").Trim()
    }
    finally {
        $env:LOGIEDGE_SQLITE_DB = $previousDb
        $env:LOGIEDGE_SQL_QUERY = $previousQuery
        Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    }
}

function Show-DatabaseSummary {
    $python = Get-PythonExecutable

    if (-not (Test-Path -LiteralPath $RuntimeDatabase -PathType Leaf)) {
        Write-Warn "Database does not exist yet: $RuntimeDatabase"
        return
    }

    $tempScript = Join-Path ([System.IO.Path]::GetTempPath()) ("logiedge_db_summary_{0}.py" -f ([guid]::NewGuid().ToString("N")))

    $pythonCode = @'
import os
import sqlite3
import sys

try:
    database_path = os.environ["LOGIEDGE_SQLITE_DB"]
    with sqlite3.connect(database_path) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        ]

        print("tables:", tables)

        if "inference_records" not in tables:
            print("inference_records table not found")
            sys.exit(0)

        labels = connection.execute(
            "SELECT label, COUNT(*) FROM inference_records GROUP BY label ORDER BY label"
        ).fetchall()
        print("labels:", labels)

        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(inference_records)"
            ).fetchall()
        ]
        print("columns:", columns)

        if "inference_synced" in columns:
            unsynced = connection.execute(
                "SELECT COUNT(*) FROM inference_records WHERE inference_synced = 0"
            ).fetchone()[0]
            print("unsynced:", unsynced)
except Exception as exc:
    print(f"Unable to query SQLite runtime database: {exc}", file=sys.stderr)
    sys.exit(1)
'@

    try {
        Set-Content -LiteralPath $tempScript -Value $pythonCode -Encoding UTF8

        $previousDb = $env:LOGIEDGE_SQLITE_DB
        $env:LOGIEDGE_SQLITE_DB = $RuntimeDatabase

        $output = & $python $tempScript 2>&1
        $exitCode = $LASTEXITCODE
        $output | ForEach-Object { Write-Host "$_" }

        if ($exitCode -ne 0) {
            Stop-Pipeline "Unable to query SQLite runtime database."
        }
    }
    finally {
        $env:LOGIEDGE_SQLITE_DB = $previousDb
        Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    }
}

function Update-LocalRegistryReferences {
    if ($NoRegistryReplacement) {
        Write-Warn "Registry hostname replacement was disabled."
        return
    }

    $files = Get-ChildItem `
        -Path $DeploymentDir `
        -Recurse `
        -File `
        -Include "*.yml","*.yaml","*.ini" `
        -ErrorAction SilentlyContinue

    foreach ($file in $files) {
        $content = Get-Content -LiteralPath $file.FullName -Raw
        $updated = $content.Replace(
            "registry.freightbridge.local:5000",
            "localhost:5000"
        )

        if ($updated -ne $content) {
            Set-Content `
                -LiteralPath $file.FullName `
                -Value $updated `
                -Encoding UTF8

            Write-Host "[FIX] Updated registry reference: $($file.FullName)" -ForegroundColor Magenta
        }
    }

    $remaining = Get-ChildItem `
        -Path $DeploymentDir `
        -Recurse `
        -File `
        -ErrorAction SilentlyContinue |
        Select-String `
            -Pattern "registry\.freightbridge\.local:5000" `
            -ErrorAction SilentlyContinue

    if ($remaining) {
        Stop-Pipeline "Old registry hostname remains in deployment files."
    }

    Write-Pass "Deployment registry references use localhost:5000"
}

function Test-CommandAvailable {
    param([Parameter(Mandatory)][string]$Command)

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        Stop-Pipeline "Required command is unavailable: $Command"
    }
}

function Start-TranscriptSafe {
    New-Item -ItemType Directory -Path $EvidenceDir -Force | Out-Null

    try {
        Start-Transcript -Path $PipelineLog -Force | Out-Null
    }
    catch {
        Write-Warn "PowerShell transcript could not be started: $($_.Exception.Message)"
    }
}

function Stop-TranscriptSafe {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
        # No active transcript.
    }
}

Start-TranscriptSafe

try {
    Write-Section "LogiEdge Complete End-to-End Pipeline - Tasks 1 to 32"

    $script:CurrentTask = "Preflight"
    Write-Step "Validate environment and required source files"

    Set-Location -LiteralPath $ProjectRoot

    Test-CommandAvailable "docker"
    Test-CommandAvailable "wsl"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("version") `
        -Description "Verify Docker engine"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("compose", "version") `
        -Description "Verify Docker Compose"

    Assert-Path -Path $ComposeFile -Description "Docker Compose file" -PathType Leaf
    Assert-Path -Path $InferenceDir -Description "inference build directory" -PathType Container
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
    New-Item -ItemType Directory -Path $EvidenceDir -Force | Out-Null

    Write-Pass "Preflight validation complete"


    # ---------------------------------------------------------------------
    # Task 1
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: ASSIGNMENT SUPPORT - SHARED PREREQUISITES
    # Supports executable validation across Components C, D, E and F.
    # No assignment task is replaced or skipped.
    # =====================================================================
    $script:CurrentTask = "Task 1"
    Write-Section "Task 1 - Check Prerequisites"

    $python = Get-PythonExecutable
    Write-Host "[INFO] Python executable: $python" -ForegroundColor Cyan

    Invoke-Native -FilePath $python -Arguments @("--version") -Description "Check Python version"
    Invoke-Native -FilePath $python -Arguments @("-c", "import numpy; print('NumPy:', numpy.__version__)") -Description "Check NumPy"
    Invoke-Native -FilePath $python -Arguments @("-c", "import tensorflow as tf; print('TensorFlow:', tf.__version__)") -Description "Check TensorFlow"
    Invoke-Native -FilePath $python -Arguments @("-c", "import paho.mqtt.client; print('Paho MQTT: PASS')") -Description "Check Paho MQTT"

    Invoke-Native -FilePath "docker" -Arguments @("version") -Description "Check Docker"
    Invoke-Native -FilePath "docker" -Arguments @("compose", "version") -Description "Check Docker Compose"
    Invoke-Native -FilePath "wsl" -Arguments @("--status") -Description "Check WSL"

    Assert-Path -Path $ComposeFile -Description "demo Docker Compose file" -PathType Leaf
    Assert-Path -Path (Join-Path $InferenceDir "Dockerfile") -Description "inference Dockerfile" -PathType Leaf
    Write-Pass "Task 1 prerequisites passed"

    if ($SkipTraining) {
        Write-Section "Tasks 2 to 7 - Training and Model Generation Skipped"
        Write-Warn "Dataset generation, M1 training, PTQ and pruning were skipped by -SkipTraining. Existing artifacts will be validated."
    }
    else {
        # -----------------------------------------------------------------
        # Task 2
        # -----------------------------------------------------------------
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT D - MODEL TRAINING, CONVERSION, AND DOCKER DEPLOYMENT [MODULE 4]
        # Assignment Task D1 - Dataset Generation and Model Training
        # Pipeline split: Task 2 generates the assignment-duration labelled dataset.
        # =====================================================================
        $script:CurrentTask = "Task 2"
        Write-Section "Task 2 - Generate Assignment-Duration Dataset"

        $assignmentGenerator = Join-Path $ProjectRoot "training\generate_assignment_dataset.py"
        Assert-Path -Path $assignmentGenerator -Description "assignment dataset generator" -PathType Leaf

        Invoke-LoggedNative `
            -FilePath $python `
            -Arguments @($assignmentGenerator) `
            -LogPath (Join-Path $EvidenceDir "manual_task02_assignment_dataset.log") `
            -Description "Generate assignment-duration dataset"

        Assert-Path `
            -Path (Join-Path $ProjectRoot "training\assignment_dataset.npz") `
            -Description "assignment dataset" `
            -PathType Leaf

        Write-Pass "Task 2 assignment dataset generated"

        # -----------------------------------------------------------------
        # Task 3
        # -----------------------------------------------------------------
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT D - TASK D1 CONTINUED
        # Pipeline split: Task 3 generates the grouped, leakage-safe training dataset and frozen statistics.
        # =====================================================================
        $script:CurrentTask = "Task 3"
        Write-Section "Task 3 - Generate Grouped Training Dataset"

        $groupedGenerator = Join-Path $ProjectRoot "training\generate_dataset.py"
        Assert-Path -Path $groupedGenerator -Description "grouped dataset generator" -PathType Leaf

        Invoke-LoggedNative `
            -FilePath $python `
            -Arguments @($groupedGenerator) `
            -LogPath (Join-Path $EvidenceDir "manual_task03_grouped_dataset.log") `
            -Description "Generate grouped training dataset"

        Assert-Path -Path $StatsSource -Description "training_stats.npy" -PathType Leaf

        $datasetArtifacts = Get-ChildItem `
            -Path (Join-Path $ProjectRoot "training"), (Join-Path $ProjectRoot "data_pipeline") `
            -Recurse `
            -File `
            -Include "*.npz","training_stats.npy" `
            -ErrorAction SilentlyContinue

        $datasetArtifacts | Select-Object FullName | Format-Table -AutoSize
        Write-Pass "Task 3 grouped dataset and statistics generated"

        # -----------------------------------------------------------------
        # Task 4 and Task 5
        # -----------------------------------------------------------------
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT D - TASK D1 CONTINUED
        # Pipeline split: Task 4 trains the M1 FP32 baseline classification model.
        # =====================================================================
        $script:CurrentTask = "Task 4"
        Write-Section "Task 4 - Train M1 FP32 Model"

        $trainingScript = Join-Path $ProjectRoot "training\train_model.py"
        $trainingLog = Join-Path $EvidenceDir "manual_task04_training.log"
        Assert-Path -Path $trainingScript -Description "M1 training script" -PathType Leaf

        $trainingOutput = Invoke-LoggedNative `
            -FilePath $python `
            -Arguments @($trainingScript) `
            -LogPath $trainingLog `
            -Description "Train M1 FP32 model"

        Write-Pass "Task 4 M1 training completed"

        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT D - TASK D1 ACCEPTANCE GATE
        # Pipeline split: Task 5 validates that held-out accuracy exceeds the mandatory 88 percent threshold.
        # =====================================================================
        $script:CurrentTask = "Task 5"
        Write-Section "Task 5 - Validate Accuracy Above 88 Percent"

        $trainingText = $trainingOutput -join "`n"
        $accuracyMatches = [regex]::Matches(
            $trainingText,
            '(?im)(?:grouped\s+validation\s+accuracy|validation\s+accuracy|val_accuracy)[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%?'
        )

        if ($accuracyMatches.Count -eq 0) {
            Stop-Pipeline "Could not find validation accuracy in $trainingLog"
        }

        $reportedAccuracy = [double]$accuracyMatches[$accuracyMatches.Count - 1].Groups[1].Value
        if ($reportedAccuracy -le 1.0) {
            $reportedAccuracy *= 100.0
        }

        Write-Host ("Grouped validation accuracy: {0:N2}%" -f $reportedAccuracy)

        if ($reportedAccuracy -le 88.0) {
            Stop-Pipeline ("M1 validation accuracy {0:N2}% does not exceed 88%." -f $reportedAccuracy)
        }

        Write-Pass ("Task 5 validation accuracy passed: {0:N2}%" -f $reportedAccuracy)

        # -----------------------------------------------------------------
        # Task 6
        # -----------------------------------------------------------------
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT F - MODEL OPTIMISATION [MODULE 6]
        # Assignment Task F1 - Three Model Variants
        # Pipeline split: Task 6 generates M2 using Full INT8 post-training quantisation.
        # =====================================================================
        $script:CurrentTask = "Task 6"
        Write-Section "Task 6 - Generate M2 PTQ INT8 Model"

        $ptqScript = Join-Path $ProjectRoot "training\convert_ptq.py"
        Assert-Path -Path $ptqScript -Description "PTQ conversion script" -PathType Leaf

        Invoke-LoggedNative `
            -FilePath $python `
            -Arguments @($ptqScript) `
            -LogPath (Join-Path $EvidenceDir "manual_task06_ptq.log") `
            -Description "Generate M2 PTQ INT8 model"

        Assert-FilePattern `
            -Directory (Join-Path $ProjectRoot "training\models") `
            -Filter "*m2*.tflite" `
            -Description "M2 PTQ INT8 model" | Out-Null

        # -----------------------------------------------------------------
        # Task 7
        # -----------------------------------------------------------------
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT F - TASK F1 CONTINUED
        # Pipeline split: Task 7 generates M3 using 35 percent structured pruning followed by Full INT8 PTQ.
        # =====================================================================
        $script:CurrentTask = "Task 7"
        Write-Section "Task 7 - Generate M3 Pruned INT8 Model"

        $pruneScript = Join-Path $ProjectRoot "training\prune_quantise.py"
        Assert-Path -Path $pruneScript -Description "M3 pruning and quantization script" -PathType Leaf

        Invoke-LoggedNative `
            -FilePath $python `
            -Arguments @($pruneScript) `
            -LogPath (Join-Path $EvidenceDir "manual_task07_pruning.log") `
            -Description "Generate M3 pruned INT8 model"

        Assert-FilePattern `
            -Directory (Join-Path $ProjectRoot "training\models") `
            -Filter "*m3*.tflite" `
            -Description "M3 pruned INT8 model" | Out-Null
    }

    # Validate model artifacts even when training was skipped.
    Assert-Path -Path $ModelSource -Description "M3 TFLite model" -PathType Leaf
    Assert-Path -Path $StatsSource -Description "training statistics" -PathType Leaf

    # ---------------------------------------------------------------------
    # Task 8
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT C - SENSOR PIPELINE AND MQTT ARCHITECTURE [MODULE 3]
    # Assignment Task C2 - Preprocessing Pipeline
    # Pipeline split: Task 8 executes the mandatory correct-statistics versus 3-sigma-shift experiment.
    # =====================================================================
    $script:CurrentTask = "Task 8"
    Write-Section "Task 8 - Run and Validate Normalization Experiment"

    $normalisationScript = Join-Path `
        $ProjectRoot `
        "experiments\normalisation_experiment.py"

    $normalisationCsv = Join-Path `
        $ProjectRoot `
        "experiments\normalisation_experiment.csv"

    $normalisationConfusion = Join-Path `
        $ProjectRoot `
        "experiments\normalisation_confusion_matrices.npz"

    $normalisationLog = Join-Path `
        $EvidenceDir `
        "manual_task08_normalisation.log"

    Assert-Path `
        -Path $normalisationScript `
        -Description "normalization experiment script" `
        -PathType Leaf

    Invoke-LoggedNative `
        -FilePath $python `
        -Arguments @($normalisationScript) `
        -LogPath $normalisationLog `
        -Description "Run normalization sensitivity experiment"

    Assert-Path `
        -Path $normalisationCsv `
        -Description "normalization experiment CSV" `
        -PathType Leaf

    Assert-Path `
        -Path $normalisationConfusion `
        -Description "normalization confusion matrices" `
        -PathType Leaf

    $normalisationRows = @(
        Import-Csv -LiteralPath $normalisationCsv
    )

    if ($normalisationRows.Count -eq 0) {
        Stop-Pipeline "Normalization experiment CSV is empty."
    }

    # These are the essential columns. The script also accepts an extended CSV
    # containing mean_shift_sigma, validation_samples and reported change fields.
    $requiredNormalisationColumns = @(
        "condition",
        "accuracy_pct",
        "recall_critical_pct"
    )

    $actualNormalisationColumns = `
        $normalisationRows[0].PSObject.Properties.Name

    $missingNormalisationColumns = @(
        $requiredNormalisationColumns |
        Where-Object {
            $_ -notin $actualNormalisationColumns
        }
    )

    if ($missingNormalisationColumns.Count -gt 0) {
        Stop-Pipeline `
            "Normalization CSV missing columns: $($missingNormalisationColumns -join ', ')"
    }

    $correctRows = @(
        $normalisationRows |
        Where-Object { $_.condition -eq "correct_stats" }
    )

    $shiftedRows = @(
        $normalisationRows |
        Where-Object { $_.condition -eq "shifted_3sigma" }
    )

    if ($correctRows.Count -ne 1) {
        Stop-Pipeline `
            "Expected exactly one correct_stats row; found $($correctRows.Count)."
    }

    if ($shiftedRows.Count -ne 1) {
        Stop-Pipeline `
            "Expected exactly one shifted_3sigma row; found $($shiftedRows.Count)."
    }

    $correctRow = $correctRows[0]
    $shiftedRow = $shiftedRows[0]

    try {
        $correctAccuracy = [double]$correctRow.accuracy_pct
        $shiftedAccuracy = [double]$shiftedRow.accuracy_pct
        $correctRecall = [double]$correctRow.recall_critical_pct
        $shiftedRecall = [double]$shiftedRow.recall_critical_pct
    }
    catch {
        Stop-Pipeline `
            "Normalization CSV contains an invalid numeric value: $($_.Exception.Message)"
    }

    $metricChecks = @(
        [pscustomobject]@{ Name = "correct accuracy"; Value = $correctAccuracy },
        [pscustomobject]@{ Name = "shifted accuracy"; Value = $shiftedAccuracy },
        [pscustomobject]@{ Name = "correct Critical recall"; Value = $correctRecall },
        [pscustomobject]@{ Name = "shifted Critical recall"; Value = $shiftedRecall }
    )

    foreach ($metric in $metricChecks) {
        if (
            [double]::IsNaN($metric.Value) -or
            [double]::IsInfinity($metric.Value)
        ) {
            Stop-Pipeline "$($metric.Name) is NaN or infinite."
        }

        if ($metric.Value -lt 0.0 -or $metric.Value -gt 100.0) {
            Stop-Pipeline "$($metric.Name) is outside the valid 0-100 range."
        }
    }

    $calculatedAccuracyChange = $shiftedAccuracy - $correctAccuracy
    $calculatedRecallChange = $shiftedRecall - $correctRecall

    # Validate optional extended-schema fields when present.
    if ("mean_shift_sigma" -in $actualNormalisationColumns) {
        try {
            $correctShift = [double]$correctRow.mean_shift_sigma
            $shiftedShift = [double]$shiftedRow.mean_shift_sigma
        }
        catch {
            Stop-Pipeline "mean_shift_sigma contains an invalid numeric value."
        }

        if ([math]::Abs($correctShift - 0.0) -gt 0.000001) {
            Stop-Pipeline "correct_stats mean_shift_sigma must be 0.0."
        }

        if ([math]::Abs($shiftedShift - 3.0) -gt 0.000001) {
            Stop-Pipeline "shifted_3sigma mean_shift_sigma must be 3.0."
        }
    }

    if ("validation_samples" -in $actualNormalisationColumns) {
        try {
            $correctSamples = [int64]$correctRow.validation_samples
            $shiftedSamples = [int64]$shiftedRow.validation_samples
        }
        catch {
            Stop-Pipeline "validation_samples contains an invalid integer value."
        }

        if ($correctSamples -le 0) {
            Stop-Pipeline "Normalization experiment has no validation samples."
        }

        if ($correctSamples -ne $shiftedSamples) {
            Stop-Pipeline `
                "Correct and shifted experiments used different validation samples."
        }
    }

    if ("accuracy_change_points" -in $actualNormalisationColumns) {
        $reportedAccuracyChange = [double]$shiftedRow.accuracy_change_points
        if ([math]::Abs($calculatedAccuracyChange - $reportedAccuracyChange) -gt 0.01) {
            Stop-Pipeline `
                "Reported accuracy change does not match the calculated change."
        }
    }

    if ("critical_recall_change_points" -in $actualNormalisationColumns) {
        $reportedRecallChange = [double]$shiftedRow.critical_recall_change_points
        if ([math]::Abs($calculatedRecallChange - $reportedRecallChange) -gt 0.01) {
            Stop-Pipeline `
                "Reported Critical recall change does not match the calculated change."
        }
    }

    Write-Host ""
    Write-Host "Normalization experiment results:" -ForegroundColor Cyan

    $normalisationRows |
        Format-Table condition, accuracy_pct, recall_critical_pct -AutoSize

    Write-Host ("Correct-stats accuracy  : {0:N2}%" -f $correctAccuracy)
    Write-Host ("Shifted-3sigma accuracy : {0:N2}%" -f $shiftedAccuracy)
    Write-Host (
        "Accuracy change         : {0:+0.00;-0.00;0.00} points" `
            -f $calculatedAccuracyChange
    )
    Write-Host ("Correct Critical recall : {0:N2}%" -f $correctRecall)
    Write-Host ("Shifted Critical recall : {0:N2}%" -f $shiftedRecall)
    Write-Host (
        "Critical recall change  : {0:+0.00;-0.00;0.00} points" `
            -f $calculatedRecallChange
    )

    Write-Pass `
        "Task 8 correct and shifted normalization results validated"
    # ---------------------------------------------------------------------
    # Task 9
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT F - MODEL OPTIMISATION [MODULE 6]
    # Assignment Task F2 - Five-Metric Benchmarking
    # Pipeline split: Task 9 benchmarks M1, M2 and M3.
    # =====================================================================
    $script:CurrentTask = "Task 9"
    Write-Section "Task 9 - Run Model Benchmark"

    $benchmarkScript = Join-Path $ProjectRoot "optimisation\benchmark.py"
    Assert-Path -Path $benchmarkScript -Description "benchmark script" -PathType Leaf

    Invoke-LoggedNative `
        -FilePath $python `
        -Arguments @($benchmarkScript) `
        -LogPath (Join-Path $EvidenceDir "manual_task09_benchmark.log") `
        -Description "Benchmark M1, M2 and M3"

    $benchmarkCsv = Join-Path $ProjectRoot "optimisation\results\benchmark_results.csv"
    $paretoChart = Join-Path $ProjectRoot "optimisation\results\pareto_chart.png"

    Assert-Path -Path $benchmarkCsv -Description "benchmark results CSV" -PathType Leaf
    Assert-Path -Path $paretoChart -Description "Pareto chart" -PathType Leaf
    Write-Pass "Task 9 benchmark outputs generated"

    # ---------------------------------------------------------------------
    # Task 10
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT F - TASK F2 CONTINUED
    # Pipeline split: Task 10 validates the benchmark methodology, five metrics and Pareto analysis.
    # =====================================================================
    $script:CurrentTask = "Task 10"
    Write-Section "Task 10 - Validate Benchmark Methodology and Results"

    $benchmarkRows = @(
        Import-Csv -LiteralPath $benchmarkCsv
    )

    $requiredColumns = @(
        "variant",
        "size_kb",
        "accuracy_pct",
        "recall_critical_pct",
        "mean_latency_ms",
        "p95_latency_ms",
        "energy_mj_per_inference",
        "warmup_runs",
        "measured_runs",
        "estimated_power_w",
        "laptop_tdp_w",
        "cpu_percent"
    )

    foreach ($row in $benchmarkRows) {
        if ([int]$row.warmup_runs -ne 10) {
            Stop-Pipeline `
                "$($row.variant) did not use exactly 10 warm-up runs."
        }

        if ([int]$row.measured_runs -ne 200) {
            Stop-Pipeline `
                "$($row.variant) did not use exactly 200 measured runs."
        }

        foreach ($field in @(
            "size_kb",
            "accuracy_pct",
            "recall_critical_pct",
            "mean_latency_ms",
            "p95_latency_ms",
            "energy_mj_per_inference"
        )) {
            $value = [double]$row.$field

            if ($value -lt 0) {
                Stop-Pipeline `
                    "$($row.variant) has an invalid negative value in $field."
            }
        }

        if (
            [double]$row.accuracy_pct -gt 100 -or
            [double]$row.recall_critical_pct -gt 100
        ) {
            Stop-Pipeline `
                "$($row.variant) has an accuracy or recall above 100%."
        }

        # Mean latency may legitimately exceed p95 when a small number
        # of extreme outliers raise the arithmetic mean. Validate each
        # latency metric independently instead of comparing them.
        $meanLatencyCheck = [double]$row.mean_latency_ms
        $p95LatencyCheck = [double]$row.p95_latency_ms

        if (
            [double]::IsNaN($meanLatencyCheck) -or
            [double]::IsInfinity($meanLatencyCheck) -or
            $meanLatencyCheck -le 0
        ) {
            Stop-Pipeline `
                "$($row.variant) has invalid mean latency."
        }

        if (
            [double]::IsNaN($p95LatencyCheck) -or
            [double]::IsInfinity($p95LatencyCheck) -or
            $p95LatencyCheck -le 0
        ) {
            Stop-Pipeline `
                "$($row.variant) has invalid p95 latency."
        }
    }

    Write-Pass `
        "Benchmark used 10 warm-ups and 200 measured runs for all variants"

    if ($benchmarkRows.Count -eq 0) {
        Stop-Pipeline "Benchmark CSV is empty."
    }

    $actualColumns = `
        $benchmarkRows[0].PSObject.Properties.Name

    $missingColumns = `
        $requiredColumns |
        Where-Object {
            $_ -notin $actualColumns
        }

    if ($missingColumns) {
        Stop-Pipeline `
            "Benchmark CSV is missing columns: $($missingColumns -join ', ')"
    }

    $expectedVariants = @(
        "M1_FP32",
        "M2_PTQ_INT8",
        "M3_PRUNE35_INT8"
    )

    $actualVariants = @(
        $benchmarkRows.variant
    )

    foreach ($expectedVariant in $expectedVariants) {
        $matchingRows = @(
            $benchmarkRows |
            Where-Object {
                $_.variant -eq $expectedVariant
            }
        )

        if ($matchingRows.Count -ne 1) {
            Stop-Pipeline `
                "Expected exactly one benchmark row for $expectedVariant; found $($matchingRows.Count)."
        }
    }

    if ($benchmarkRows.Count -ne 3) {
        Stop-Pipeline `
            "Benchmark CSV must contain exactly three rows; found $($benchmarkRows.Count)."
    }

    foreach ($row in $benchmarkRows) {
        try {
            $sizeKb = [double]$row.size_kb
            $accuracyPct = [double]$row.accuracy_pct
            $criticalRecallPct = `
                [double]$row.recall_critical_pct

            $meanLatencyMs = `
                [double]$row.mean_latency_ms

            $p95LatencyMs = `
                [double]$row.p95_latency_ms

            $energyMj = `
                [double]$row.energy_mj_per_inference

            $warmupRuns = `
                [int]$row.warmup_runs

            $measuredRuns = `
                [int]$row.measured_runs

            $estimatedPowerW = `
                [double]$row.estimated_power_w

            $laptopTdpW = `
                [double]$row.laptop_tdp_w

            $cpuPercent = `
                [double]$row.cpu_percent
        }
        catch {
            Stop-Pipeline `
                "$($row.variant) has a non-numeric benchmark value: $($_.Exception.Message)"
        }

        if ($warmupRuns -ne 10) {
            Stop-Pipeline `
                "$($row.variant) did not use exactly 10 warm-up runs; found $warmupRuns."
        }

        if ($measuredRuns -ne 200) {
            Stop-Pipeline `
                "$($row.variant) did not use exactly 200 measured runs; found $measuredRuns."
        }

        if ($sizeKb -le 0) {
            Stop-Pipeline `
                "$($row.variant) model size must be greater than zero."
        }

        if (
            $accuracyPct -lt 0 -or
            $accuracyPct -gt 100
        ) {
            Stop-Pipeline `
                "$($row.variant) accuracy is outside 0-100%."
        }

        if (
            $criticalRecallPct -lt 0 -or
            $criticalRecallPct -gt 100
        ) {
            Stop-Pipeline `
                "$($row.variant) Critical recall is outside 0-100%."
        }

        if ($meanLatencyMs -le 0) {
            Stop-Pipeline `
                "$($row.variant) mean latency must be greater than zero."
        }

        if ($p95LatencyMs -le 0) {
            Stop-Pipeline `
                "$($row.variant) p95 latency must be greater than zero."
        }

        # Do not require p95 >= mean. Both values have already been
        # validated independently as finite and positive.
        if ($energyMj -lt 0) {
            Stop-Pipeline `
                "$($row.variant) energy estimate cannot be negative."
        }

        if ($estimatedPowerW -lt 0) {
            Stop-Pipeline `
                "$($row.variant) estimated power cannot be negative."
        }

        if ($laptopTdpW -le 0) {
            Stop-Pipeline `
                "$($row.variant) laptop TDP must be greater than zero."
        }

        if (
            $cpuPercent -lt 0 -or
            $cpuPercent -gt 100
        ) {
            Stop-Pipeline `
                "$($row.variant) CPU percentage is outside 0-100."
        }

        $expectedEnergyMj = `
            $estimatedPowerW * $meanLatencyMs

        $energyTolerance = [math]::Max(
            0.001,
            [math]::Abs($expectedEnergyMj) * 0.02
        )

        if (
            [math]::Abs(
                $energyMj - $expectedEnergyMj
            ) -gt $energyTolerance
        ) {
            Stop-Pipeline `
                "$($row.variant) energy does not match power x latency."
        }
    }

    $benchmarkRows |
        Format-Table `
            variant,
            size_kb,
            accuracy_pct,
            recall_critical_pct,
            mean_latency_ms,
            p95_latency_ms,
            energy_mj_per_inference,
            warmup_runs,
            measured_runs,
            estimated_power_w,
            laptop_tdp_w `
            -AutoSize

    $paretoAnalysisCsv = Join-Path `
        $ProjectRoot `
        "optimisation\results\pareto_analysis.csv"

    Assert-Path `
        -Path $paretoAnalysisCsv `
        -Description "Pareto analysis CSV" `
        -PathType Leaf

    $paretoRows = @(
        Import-Csv -LiteralPath $paretoAnalysisCsv
    )

    if ($paretoRows.Count -ne 3) {
        Stop-Pipeline `
            "Pareto analysis must contain exactly three rows; found $($paretoRows.Count)."
    }

    $requiredParetoColumns = @(
        "variant",
        "accuracy_pct",
        "mean_latency_ms",
        "size_kb",
        "pareto_optimal",
        "recommended"
    )

    $actualParetoColumns = @(
        $paretoRows[0].PSObject.Properties.Name
    )

    $missingParetoColumns = @(
        $requiredParetoColumns |
        Where-Object {
            $_ -notin $actualParetoColumns
        }
    )

    if ($missingParetoColumns.Count -gt 0) {
        Stop-Pipeline `
            "Pareto CSV is missing columns: $($missingParetoColumns -join ', ')"
    }

    foreach ($expectedVariant in $expectedVariants) {
        $matchingParetoRows = @(
            $paretoRows |
            Where-Object {
                $_.variant -eq $expectedVariant
            }
        )

        if ($matchingParetoRows.Count -ne 1) {
            Stop-Pipeline `
                "Expected exactly one Pareto row for $expectedVariant; found $($matchingParetoRows.Count)."
        }
    }

    foreach ($row in $paretoRows) {
        try {
            $paretoAccuracy = [double]$row.accuracy_pct
            $paretoLatency = [double]$row.mean_latency_ms
            $paretoSize = [double]$row.size_kb
        }
        catch {
            Stop-Pipeline `
                "$($row.variant) contains a non-numeric Pareto value."
        }

        foreach ($metric in @(
            [pscustomobject]@{
                Name = "accuracy_pct"
                Value = $paretoAccuracy
            },
            [pscustomobject]@{
                Name = "mean_latency_ms"
                Value = $paretoLatency
            },
            [pscustomobject]@{
                Name = "size_kb"
                Value = $paretoSize
            }
        )) {
            if (
                [double]::IsNaN($metric.Value) -or
                [double]::IsInfinity($metric.Value)
            ) {
                Stop-Pipeline `
                    "$($row.variant) has a non-finite Pareto value in $($metric.Name)."
            }
        }

        if ($paretoAccuracy -lt 0 -or $paretoAccuracy -gt 100) {
            Stop-Pipeline `
                "$($row.variant) Pareto accuracy is outside 0-100%."
        }

        if ($paretoLatency -le 0) {
            Stop-Pipeline `
                "$($row.variant) Pareto latency must be greater than zero."
        }

        if ($paretoSize -le 0) {
            Stop-Pipeline `
                "$($row.variant) Pareto size must be greater than zero."
        }

        $paretoFlag = "$($row.pareto_optimal)".Trim().ToLower()
        $recommendedFlag = "$($row.recommended)".Trim().ToLower()

        if ($paretoFlag -notin @("true", "false")) {
            Stop-Pipeline `
                "$($row.variant) has an invalid pareto_optimal flag."
        }

        if ($recommendedFlag -notin @("true", "false")) {
            Stop-Pipeline `
                "$($row.variant) has an invalid recommended flag."
        }
    }

    $paretoOptimalRows = @(
        $paretoRows |
        Where-Object {
            "$($_.pareto_optimal)".Trim().ToLower() -eq "true"
        }
    )

    if ($paretoOptimalRows.Count -eq 0) {
        Stop-Pipeline `
            "Pareto analysis identifies no Pareto-optimal model."
    }

    $recommendedRows = @(
        $paretoRows |
        Where-Object {
            "$($_.recommended)".Trim().ToLower() -eq "true"
        }
    )

    if ($recommendedRows.Count -ne 1) {
        Stop-Pipeline `
            "Pareto analysis must identify exactly one recommended model; found $($recommendedRows.Count)."
    }

    if (
        "$($recommendedRows[0].pareto_optimal)".Trim().ToLower() `
            -ne "true"
    ) {
        Stop-Pipeline `
            "The recommended model is not Pareto optimal."
    }

    $paretoRows |
        Format-Table `
            variant,
            accuracy_pct,
            mean_latency_ms,
            size_kb,
            pareto_optimal,
            recommended `
            -AutoSize

    Write-Pass `
        "Task 10 validated benchmark methodology and Pareto frontier"

    Assert-Path `
        -Path $OptimisationMetadataPath `
        -Description "model optimisation metadata" `
        -PathType Leaf

    $optimisationMetadata = `
        Get-Content -LiteralPath $OptimisationMetadataPath -Raw |
        ConvertFrom-Json

    if ([int]$optimisationMetadata.m2_calibration_samples -lt 200) {
        Stop-Pipeline "M2 used fewer than 200 calibration samples."
    }

    if ([int]$optimisationMetadata.m3_calibration_samples -lt 200) {
        Stop-Pipeline "M3 used fewer than 200 calibration samples."
    }

    if ([double]$optimisationMetadata.m3_target_sparsity -ne 0.35) {
        Stop-Pipeline "M3 target sparsity is not 35%."
    }

    if ("$($optimisationMetadata.pruning_schedule)" -ne "PolynomialDecay") {
        Stop-Pipeline "M3 did not use PolynomialDecay pruning."
    }

    Write-Pass "Model optimisation metadata validated"

    # ---------------------------------------------------------------------
    # Task 11
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT F - TASK F3 DEPLOYMENT RECOMMENDATION
    # Pipeline split: Task 11 verifies Critical-class recall above 95 percent for the recommended M3 variant.
    # =====================================================================
    $script:CurrentTask = "Task 11"
    Write-Section "Task 11 - Validate M3 Critical Recall"

    $m3 = Get-BenchmarkM3
    $m3 | Format-List

    $m3Recall = [double]$m3.recall_critical_pct
    Write-Host ("M3 critical recall: {0:N2}%" -f $m3Recall)

    if ($m3Recall -le 95.0) {
        Stop-Pipeline ("M3 critical recall {0:N2}% does not exceed 95%." -f $m3Recall)
    }

    Write-Pass ("Task 11 M3 critical recall passed: {0:N2}%" -f $m3Recall)

    # ---------------------------------------------------------------------
    # Task 12
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT E - EDGE MLOPS: MONITORING AND DEPLOYMENT MANAGEMENT [MODULE 5]
    # Assignment Task E1 - PSI Drift Monitoring
    # Pipeline split: Task 12 creates the reference distribution from clean Normal-class windows.
    # =====================================================================
    $script:CurrentTask = "Task 12"
    Write-Section "Task 12 - Generate PSI Reference Distribution"

    $driftScript = Join-Path $ProjectRoot "monitoring\drift_monitor.py"
    Assert-Path -Path $driftScript -Description "drift monitor script" -PathType Leaf

    Invoke-LoggedNative `
        -FilePath $python `
        -Arguments @($driftScript, "--mode", "reference", "--score", "normal_prob") `
        -LogPath (Join-Path $EvidenceDir "manual_task12_psi_reference.log") `
        -Description "Generate PSI reference distribution"

    Assert-Path -Path $PsiSource -Description "PSI reference distribution" -PathType Leaf
    Write-Pass "Task 12 PSI reference generated"

    # ---------------------------------------------------------------------
    # Task 13
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT E - TASK E1 CONTINUED
    # Pipeline split: Task 13 injects drift and demonstrates recovery.
    # =====================================================================
    $script:CurrentTask = "Task 13"
    Write-Section "Task 13 - Run PSI Drift and Recovery Simulation"

    $psiSimulationLog = Join-Path $EvidenceDir "manual_task13_psi_simulation.log"

    $psiOutput = Invoke-LoggedNative `
        -FilePath $python `
        -Arguments @($driftScript, "--mode", "simulate", "--score", "normal_prob") `
        -LogPath $psiSimulationLog `
        -Description "Run PSI drift and recovery simulation"

    Write-Pass "Task 13 PSI simulation completed"

    # ---------------------------------------------------------------------
    # Task 14
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT E - TASK E1 ACCEPTANCE GATES
    # Pipeline split: Task 14 validates PSI greater than 0.25 within five minutes and recovery below 0.10.
    # =====================================================================
    $script:CurrentTask = "Task 14"
    Write-Section "Task 14 - Validate PSI Thresholds"

    $psiText = $psiOutput -join "`n"

    $injectedMatch = [regex]::Match(
        $psiText,
        '(?im)injected(?:\s+maximum|\s+max)?\s+PSI\s*:\s*([0-9]+(?:\.[0-9]+)?)'
    )

    $recoveredMatch = [regex]::Match(
        $psiText,
        '(?im)recovered(?:\s+final)?\s+PSI\s*:\s*([0-9]+(?:\.[0-9]+)?)'
    )

    if (-not $injectedMatch.Success) {
        Stop-Pipeline "Injected maximum PSI was not found in $psiSimulationLog"
    }

    if (-not $recoveredMatch.Success) {
        Stop-Pipeline "Recovered final PSI was not found in $psiSimulationLog"
    }

    $injectedPsi = [double]$injectedMatch.Groups[1].Value
    $recoveredPsi = [double]$recoveredMatch.Groups[1].Value

    Write-Host ("Injected maximum PSI: {0:N3}" -f $injectedPsi)
    Write-Host ("Recovered final PSI: {0:N3}" -f $recoveredPsi)

    if ($injectedPsi -le 0.25) {
        Stop-Pipeline ("Injected PSI {0:N3} did not exceed 0.25." -f $injectedPsi)
    }

    if ($recoveredPsi -ge 0.10) {
        Stop-Pipeline ("Recovered PSI {0:N3} was not below 0.10." -f $recoveredPsi)
    }

    Assert-Path `
        -Path (Join-Path $ProjectRoot "monitoring\psi_trace.json") `
        -Description "PSI trace JSON" `
        -PathType Leaf

    Assert-Path `
        -Path $PsiMetadataPath `
        -Description "PSI metadata" `
        -PathType Leaf

    $psiMetadata = `
        Get-Content -LiteralPath $PsiMetadataPath -Raw |
        ConvertFrom-Json

    if ([int]$psiMetadata.reference_windows -ne 300) {
        Stop-Pipeline "PSI reference must use exactly 300 clean windows."
    }

    if ([int]$psiMetadata.rolling_window -ne 100) {
        Stop-Pipeline "PSI rolling window must be 100 inferences."
    }

    if ([int]$psiMetadata.evaluation_interval_seconds -ne 60) {
        Stop-Pipeline "PSI evaluation interval must be 60 seconds."
    }

    if ([double]$psiMetadata.detection_lag_minutes -gt 5.0) {
        Stop-Pipeline "PSI drift detection exceeded five minutes."
    }

    $psiBins = @($psiMetadata.bins)
    if ($psiBins.Count -lt 2) {
        Stop-Pipeline "PSI metadata must define at least two bin boundaries."
    }

    Write-Pass "Task 14 PSI thresholds, timing and configuration validated"

    # ---------------------------------------------------------------------
    # Task 15
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: CROSS-COMPONENT AUTOMATED VERIFICATION
    # Pipeline split: Task 15 runs regression tests supporting Components C-F.
    # =====================================================================
    $script:CurrentTask = "Task 15"
    Write-Section "Task 15 - Run Automated Tests"

    if ($SkipTests) {
        Write-Warn "Task 15 skipped by -SkipTests."
    }
    else {
        $preprocessingContractTest = Join-Path `
            $ProjectRoot `
            "tests\test_preprocessing_contract.py"

        Assert-Path `
            -Path $preprocessingContractTest `
            -Description "preprocessing contract test" `
            -PathType Leaf

        Invoke-LoggedNative `
            -FilePath $python `
            -Arguments @(
                "-m",
                "pytest",
                $preprocessingContractTest,
                "-q"
            ) `
            -LogPath (
                Join-Path `
                    $EvidenceDir `
                    "manual_task15_preprocessing_contract.log"
            ) `
            -Description "Run preprocessing contract tests"

        Invoke-LoggedNative `
            -FilePath $python `
            -Arguments @("-m", "pytest", ".\tests", "-q") `
            -LogPath (Join-Path $EvidenceDir "manual_task15_pytest.log") `
            -Description "Run complete automated pytest suite"

        Write-Pass `
            "Task 15 preprocessing contract and complete pytest suite passed"
    }

    # ---------------------------------------------------------------------
    # Task 16
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENTS A AND B - REPORT-BASED ASSIGNMENT EVIDENCE
    # Component A - System Architecture and Deployment Justification [Module 1]
    #   Task A1 - Constraint Analysis
    #   Task A2 - System Architecture Diagram
    # Component B - Hardware Selection and Justification [Module 2]
    #   Task B1 - Constraint Triangle Application
    #   Task B2 - Arithmetic Intensity and Roofline Analysis
    # Also supports Task C3 Data Fusion Justification, Task D3 10-Stage Pipeline Mapping,
    # and Task E3 OTA Strategy Selection through generated diagrams and calculations.
    # =====================================================================
    $script:CurrentTask = "Task 16"
    Write-Section "Task 16 - Generate Diagrams and Calculations"

    $diagramScript = Join-Path $ProjectRoot "scenario_architecture\make_diagrams.py"
    $figureScript = Join-Path $ProjectRoot "reports\build_evidence_figures.py"
    $constraintScript = Join-Path $ProjectRoot "experiments\constraint_numbers.py"

    Assert-Path -Path $diagramScript -Description "architecture diagram script" -PathType Leaf
    Assert-Path -Path $figureScript -Description "evidence figure script" -PathType Leaf
    Assert-Path -Path $constraintScript -Description "constraint calculation script" -PathType Leaf

    Invoke-LoggedNative `
        -FilePath $python `
        -Arguments @($diagramScript) `
        -LogPath (Join-Path $EvidenceDir "manual_task16_diagrams.log") `
        -Description "Generate architecture diagrams"

    Invoke-LoggedNative `
        -FilePath $python `
        -Arguments @($figureScript) `
        -LogPath (Join-Path $EvidenceDir "manual_task16_evidence_figures.log") `
        -Description "Generate report evidence figures"

    Invoke-LoggedNative `
        -FilePath $python `
        -Arguments @($constraintScript) `
        -LogPath (Join-Path $EvidenceDir "manual_task16_calculations.txt") `
        -Description "Generate constraint calculations"

    Assert-Path `
        -Path (Join-Path $ProjectRoot "scenario_architecture\system_architecture.png") `
        -Description "system architecture diagram" `
        -PathType Leaf

    Assert-Path `
        -Path (Join-Path $ProjectRoot "data_pipeline\mqtt_topic_tree.png") `
        -Description "MQTT topic tree diagram" `
        -PathType Leaf

    Assert-Path `
        -Path (Join-Path $ProjectRoot "reports\figures") `
        -Description "report figures directory" `
        -PathType Container

    Write-Pass "Task 16 diagrams, figures and calculations generated"


    # ---------------------------------------------------------------------
    # Task 17
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: RUNTIME PREPARATION SUPPORTING COMPONENTS C, D AND E
    # Pipeline split: Task 17 stages the model, frozen training statistics, PSI reference and clean runtime database.
    # =====================================================================
    $script:CurrentTask = "Task 17"
    Write-Section "Task 17 - Prepare Inference Context and Runtime Environment"


    $preprocessingSource = Join-Path $ProjectRoot "data_pipeline\preprocessing.py"
    $preprocessingDestination = Join-Path $InferenceDir "preprocessing.py"
    $tfliteEvalSource = Join-Path $ProjectRoot "optimisation\tflite_eval.py"
    $tfliteEvalDestination = Join-Path $InferenceDir "tflite_eval.py"

    Assert-Path `
        -Path $preprocessingSource `
        -Description "data-pipeline preprocessing module" `
        -PathType Leaf

    Copy-Item `
        -LiteralPath $preprocessingSource `
        -Destination $preprocessingDestination `
        -Force

    Assert-Path `
        -Path $preprocessingDestination `
        -Description "inference preprocessing module" `
        -PathType Leaf

    Write-Pass "Step 17.1 copied preprocessing.py into the inference context"

    if (Test-Path -LiteralPath $tfliteEvalSource -PathType Leaf) {
        Copy-Item `
            -LiteralPath $tfliteEvalSource `
            -Destination $tfliteEvalDestination `
            -Force

        Assert-Path `
            -Path $tfliteEvalDestination `
            -Description "inference TFLite evaluation module" `
            -PathType Leaf

        Write-Pass "Step 17.2 copied tflite_eval.py into the inference context"
    }
    else {
        Write-Warn "optimisation\tflite_eval.py was not found; no copy was performed."
    }

    Copy-Item -LiteralPath $ModelSource -Destination $RuntimeModel -Force
    Assert-Path -Path $RuntimeModel -Description "runtime model" -PathType Leaf
    Write-Pass "Step 17.3 copied M3 model to $RuntimeModel"

    Copy-Item -LiteralPath $StatsSource -Destination $RuntimeStats -Force
    Assert-Path -Path $RuntimeStats -Description "runtime training statistics" -PathType Leaf
    Write-Pass "Step 17.4 copied training statistics"

    Copy-Item -LiteralPath $PsiSource -Destination $RuntimePsi -Force
    Assert-Path -Path $RuntimePsi -Description "runtime PSI reference" -PathType Leaf
    Write-Pass "Step 17.5 copied PSI reference distribution"

    if (Test-Path -LiteralPath $RuntimeDatabase) {
        Remove-Item -LiteralPath $RuntimeDatabase -Force
    }

    if (Test-Path -LiteralPath $RuntimeDatabase) {
        Stop-Pipeline "Old database could not be removed."
    }

    Write-Pass "Step 17.6 removed old runtime database"

    # Validate and automatically correct the Paho MQTT v2 ReasonCode callback.
    $inferenceServiceFile = Join-Path $InferenceDir "inference_service.py"
    Assert-Path `
        -Path $inferenceServiceFile `
        -Description "inference service source" `
        -PathType Leaf

    $inferenceServiceContent = Get-Content `
        -LiteralPath $inferenceServiceFile `
        -Raw

    if ($inferenceServiceContent -match 'if\s+int\(reason_code\)\s*!=\s*0\s*:') {
        $inferenceServiceContent = $inferenceServiceContent -replace `
            'if\s+int\(reason_code\)\s*!=\s*0\s*:', `
            'if reason_code.is_failure:'

        Set-Content `
            -LiteralPath $inferenceServiceFile `
            -Value $inferenceServiceContent `
            -Encoding UTF8

        Write-Host `
            "[FIX] Updated Paho MQTT ReasonCode handling in inference_service.py" `
            -ForegroundColor Magenta
    }

    $remainingReasonCodeIssue = Select-String `
        -LiteralPath $inferenceServiceFile `
        -Pattern 'int\(reason_code\)' `
        -Quiet

    if ($remainingReasonCodeIssue) {
        Stop-Pipeline "Old int(reason_code) handling remains in inference_service.py."
    }

    $validReasonCodeHandling = Select-String `
        -LiteralPath $inferenceServiceFile `
        -Pattern 'reason_code\.is_failure' `
        -Quiet

    if (-not $validReasonCodeHandling) {
        Stop-Pipeline "Expected reason_code.is_failure handling was not found in inference_service.py."
    }

    Write-Pass "Step 17.7 validated Paho MQTT v2 callback compatibility"

    # ---------------------------------------------------------------------
    # Dockerfile contract validation before Task 18
    # ---------------------------------------------------------------------
    $script:CurrentTask = "Dockerfile contract"
    Write-Section "Validate Dockerfile Contract"

    $dockerfilePath = Join-Path $InferenceDir "Dockerfile"
    Assert-Path -Path $dockerfilePath -Description "inference Dockerfile" -PathType Leaf

    $dockerfileText = Get-Content -LiteralPath $dockerfilePath -Raw

    if ($dockerfileText -notmatch '(?im)^FROM\s+python:3\.11-slim\s*$') {
        Stop-Pipeline "Dockerfile must use python:3.11-slim."
    }

    if ($dockerfileText -notmatch '(?im)\bMODEL_PATH\b') {
        Stop-Pipeline "Dockerfile does not define MODEL_PATH."
    }

    $pipIndex = $dockerfileText.IndexOf(
        "pip install",
        [System.StringComparison]::OrdinalIgnoreCase
    )

    $modelCopyIndex = $dockerfileText.IndexOf(
        "COPY model.tflite",
        [System.StringComparison]::OrdinalIgnoreCase
    )

    if ($pipIndex -lt 0) {
        Stop-Pipeline "Dockerfile has no pip install layer."
    }

    if ($modelCopyIndex -lt 0) {
        Stop-Pipeline "Dockerfile has no COPY model.tflite instruction."
    }

    if ($pipIndex -gt $modelCopyIndex) {
        Stop-Pipeline "pip install must appear before COPY model.tflite."
    }

    $requiredDockerCopies = @(
        "COPY preprocessing.py /app/preprocessing.py",
        "COPY tflite_eval.py /app/tflite_eval.py",
        "COPY psi.py /app/psi.py",
        "COPY inference_service.py /app/inference_service.py",
        "COPY training_stats.npy /data/training_stats.npy",
        "COPY reference_dist.json /data/reference_dist.json",
        "COPY model.tflite /data/model.tflite"
    )

    foreach ($copyInstruction in $requiredDockerCopies) {
        if ($dockerfileText.IndexOf($copyInstruction, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
            Stop-Pipeline "Dockerfile is missing required instruction: $copyInstruction"
        }
    }

    Write-Pass "Dockerfile base image, MODEL_PATH and layer order validated"

    # ---------------------------------------------------------------------
    # Task 18
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT D - MODEL TRAINING, CONVERSION, AND DOCKER DEPLOYMENT [MODULE 4]
    # Assignment Task D2 - Docker Containerisation and OTA Demo
    # Pipeline split: Task 18 builds the cached baseline Docker image.
    # =====================================================================
    $script:CurrentTask = "Task 18"
    Write-Section "Task 18 - Build Cached Baseline Docker Image v1"

    Assert-Path -Path $DockerModel -Description "Docker-context baseline model" -PathType Leaf

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @(
            "build",
            "--progress=plain",
            "-t", $ImageV1,
            $InferenceDir
        ) `
        -Description "Build cached baseline image $ImageV1"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("image", "inspect", $ImageV1) `
        -Description "Verify $ImageV1"

    Write-Pass "Docker baseline image $ImageV1 is available"

    # ---------------------------------------------------------------------
    # Task 19
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT D - TASK D2 OTA DEMONSTRATION
    # Pipeline split: Task 19 changes only the model layer and validates Docker layer-cache reuse.
    # =====================================================================
    $script:CurrentTask = "Task 19"
    Write-Section "Task 19 - OTA Model-Only Layer Cache Test"

    if ($SkipOtaBuild) {
        Write-Warn "Task 19 skipped by -SkipOtaBuild."
    }
    else {
        Assert-Path `
            -Path $OtaModelSource `
            -Description "OTA replacement TFLite model" `
            -PathType Leaf

        Assert-Path `
            -Path $DockerModel `
            -Description "Docker-context model" `
            -PathType Leaf

        $baselineBackup = Join-Path $EvidenceDir "task19_baseline_model_backup.tflite"

        # Always initialise the Docker build context from the authoritative
        # baseline M3 model. This also repairs the context after an interrupted
        # or previously failed OTA test that left the OTA model in place.
        Copy-Item `
            -LiteralPath $ModelSource `
            -Destination $DockerModel `
            -Force

        $modelSourceHash = (
            Get-FileHash `
                -LiteralPath $ModelSource `
                -Algorithm SHA256
        ).Hash

        $dockerBaselineHash = (
            Get-FileHash `
                -LiteralPath $DockerModel `
                -Algorithm SHA256
        ).Hash

        if ($dockerBaselineHash -ne $modelSourceHash) {
            Stop-Pipeline "Unable to initialise Docker-context model from the baseline M3 model."
        }

        Copy-Item `
            -LiteralPath $DockerModel `
            -Destination $baselineBackup `
            -Force

        try {
            $beforeHash = $dockerBaselineHash
            $sourceHash = (Get-FileHash -LiteralPath $OtaModelSource -Algorithm SHA256).Hash

            if ($beforeHash -eq $sourceHash) {
                Write-Warn "OTA source is identical to the baseline. Generating a hash-distinct, loadable OTA artifact."

                $otaGenerator = Join-Path `
                    ([System.IO.Path]::GetTempPath()) `
                    ("logiedge_generate_ota_{0}.py" -f ([guid]::NewGuid().ToString("N")))

                $otaGeneratorCode = @'
import os
from pathlib import Path

import tensorflow as tf

source = Path(os.environ["LOGIEDGE_OTA_BASELINE"])
target = Path(os.environ["LOGIEDGE_OTA_TARGET"])

payload = source.read_bytes()
marker = b"\nLOGIEDGE_OTA_MODEL_ONLY_V2\n"

target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(payload + marker)

interpreter = tf.lite.Interpreter(model_path=str(target))
interpreter.allocate_tensors()

if target.read_bytes() == payload:
    raise RuntimeError("Generated OTA artifact is still identical to baseline")

print(f"Generated validated OTA artifact: {target}")
'@

                $previousOtaBaseline = $env:LOGIEDGE_OTA_BASELINE
                $previousOtaTarget = $env:LOGIEDGE_OTA_TARGET

                try {
                    Set-Content `
                        -LiteralPath $otaGenerator `
                        -Value $otaGeneratorCode `
                        -Encoding UTF8

                    $env:LOGIEDGE_OTA_BASELINE = $ModelSource
                    $env:LOGIEDGE_OTA_TARGET = $OtaModelSource

                    Invoke-Native `
                        -FilePath $python `
                        -Arguments @($otaGenerator) `
                        -Description "Generate and validate hash-distinct OTA model artifact"
                }
                finally {
                    $env:LOGIEDGE_OTA_BASELINE = $previousOtaBaseline
                    $env:LOGIEDGE_OTA_TARGET = $previousOtaTarget

                    Remove-Item `
                        -LiteralPath $otaGenerator `
                        -Force `
                        -ErrorAction SilentlyContinue
                }

                $sourceHash = (
                    Get-FileHash `
                        -LiteralPath $OtaModelSource `
                        -Algorithm SHA256
                ).Hash

                if ($beforeHash -eq $sourceHash) {
                    Stop-Pipeline "Generated OTA source still has the same SHA-256 hash as the baseline model."
                }
            }

            Copy-Item -LiteralPath $OtaModelSource -Destination $DockerModel -Force

            $afterHash = (Get-FileHash -LiteralPath $DockerModel -Algorithm SHA256).Hash

            if ($beforeHash -eq $afterHash) {
                Stop-Pipeline "OTA model replacement did not change model hash."
            }

            Write-Host "Baseline model hash: $beforeHash"
            Write-Host "OTA model hash     : $afterHash"

            $otaBuildOutput = Invoke-LoggedNative `
                -FilePath "docker" `
                -Arguments @(
                    "build",
                    "--progress=plain",
                    "-t", $ImageOta,
                    $InferenceDir
                ) `
                -LogPath $OtaBuildLog `
                -Description "Build OTA image using Docker layer cache"

            $otaBuildText = $otaBuildOutput -join "`n"

            if ($otaBuildText -notmatch '(?i)CACHED') {
                Stop-Pipeline "OTA build did not demonstrate cached layers."
            }

            if ($otaBuildText -notmatch '(?i)COPY\s+model\.tflite|model\.tflite') {
                Stop-Pipeline "OTA build output does not show the model layer."
            }

            Invoke-Native `
                -FilePath "docker" `
                -Arguments @("image", "inspect", $ImageOta) `
                -Description "Verify OTA image"

            $imageSizeText = & docker image inspect $ImageV1 --format "{{.Size}}"
            if ($LASTEXITCODE -ne 0) {
                Stop-Pipeline "Unable to obtain baseline Docker image size."
            }

            $imageSizeBytes = [int64]("$imageSizeText".Trim())
            $imageSizeMb = $imageSizeBytes / 1MB
            $modelSizeMb = (Get-Item -LiteralPath $DockerModel).Length / 1MB

            if ($imageSizeMb -le 0) {
                Stop-Pipeline "Baseline image size must be greater than zero."
            }

            $fullFleetMb = $imageSizeMb * 85
            $modelFleetMb = $modelSizeMb * 85
            $fullFleetCost = $fullFleetMb * 0.10
            $modelFleetCost = $modelFleetMb * 0.10
            $bandwidthSavingPct = (1.0 - ($modelFleetMb / $fullFleetMb)) * 100.0

            $otaEvidence = @(
                "Baseline model hash: $beforeHash",
                "OTA model hash     : $afterHash",
                ("Full image size       : {0:N2} MB" -f $imageSizeMb),
                ("Model size            : {0:N4} MB" -f $modelSizeMb),
                ("85-truck full cost    : INR {0:N2}" -f $fullFleetCost),
                ("85-truck model cost   : INR {0:N2}" -f $modelFleetCost),
                ("Bandwidth saving      : {0:N3}%" -f $bandwidthSavingPct)
            )

            $otaEvidence | ForEach-Object { Write-Host $_ }
            $otaEvidence | Add-Content -LiteralPath $OtaBuildLog -Encoding UTF8

            Write-Pass "OTA model hash changed and Docker cache was reused"
        }
        finally {
            # Restore directly from the authoritative source first. The backup
            # remains a secondary fallback only if the source unexpectedly
            # becomes unavailable during the test.
            if (Test-Path -LiteralPath $ModelSource -PathType Leaf) {
                Copy-Item `
                    -LiteralPath $ModelSource `
                    -Destination $DockerModel `
                    -Force
            }
            elseif (Test-Path -LiteralPath $baselineBackup -PathType Leaf) {
                Copy-Item `
                    -LiteralPath $baselineBackup `
                    -Destination $DockerModel `
                    -Force
            }

            Remove-Item `
                -LiteralPath $baselineBackup `
                -Force `
                -ErrorAction SilentlyContinue
        }

        Assert-Path `
            -Path $DockerModel `
            -Description "restored Docker-context baseline model" `
            -PathType Leaf

        $restoredHash = (
            Get-FileHash `
                -LiteralPath $DockerModel `
                -Algorithm SHA256
        ).Hash

        $modelSourceHash = (
            Get-FileHash `
                -LiteralPath $ModelSource `
                -Algorithm SHA256
        ).Hash

        Write-Host "Restored model hash : $restoredHash"
        Write-Host "Expected model hash : $modelSourceHash"

        if ($restoredHash -ne $modelSourceHash) {
            Stop-Pipeline `
                "Baseline Docker-context model was not restored after OTA test. Expected $modelSourceHash but found $restoredHash."
        }

        Write-Pass "Baseline model restored after OTA cache test"
    }

    # ---------------------------------------------------------------------
    # Task 20
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT C - SENSOR PIPELINE AND MQTT ARCHITECTURE [MODULE 3]
    # Assignment Task C1 - Sensor Simulator and local MQTT architecture
    # Pipeline split: Task 20 starts the local and uplink Mosquitto brokers.
    # =====================================================================
    $script:CurrentTask = "Task 20"
    Write-Section "Task 20 - Start Docker Compose MQTT Brokers"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("compose", "-f", $ComposeFile, "down", "--remove-orphans") `
        -Description "Stop previous Compose stack"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("compose", "-f", $ComposeFile, "up", "-d") `
        -Description "Start MQTT brokers"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("compose", "-f", $ComposeFile, "ps") `
        -Description "Display Compose services"

    Wait-Container -Name $LocalBroker -TimeoutSeconds 60
    Wait-Container -Name $UplinkBroker -TimeoutSeconds 60

    Write-Pass "Local and uplink MQTT brokers are running"

    # ---------------------------------------------------------------------
    # Task 21
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT D - TASK D2 RUNTIME CONTAINER
    # Pipeline split: Task 21 starts the containerised preprocessing, inference and MQTT publishing service.
    # =====================================================================
    $script:CurrentTask = "Task 21"
    Write-Section "Task 21 - Start Inference Container"

    $existingInferenceContainer = docker ps -a `
    --filter "name=^/$InferenceContainer$" `
    --format "{{.Names}}"

    if ($existingInferenceContainer -eq $InferenceContainer) {
        Write-Host "[INFO] Removing existing inference container: $InferenceContainer"
        docker rm -f $InferenceContainer | Out-Null

        if ($LASTEXITCODE -ne 0) {
            Stop-Pipeline "Unable to remove existing inference container."
        }
    }
    else {
        Write-Host "[INFO] No previous inference container found. Continuing."
    }

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @(
            "run", "-d",
            "--name", $InferenceContainer,
            "--restart", "unless-stopped",
            "-e", "TRUCK_ID=$TruckId",
            "-e", "MODEL_PATH=/data/model.tflite",
            "-e", "LOCAL_MQTT_HOST=host.docker.internal",
            "-e", "LOCAL_MQTT_PORT=1883",
            "-e", "UPLINK_MQTT_HOST=host.docker.internal",
            "-e", "UPLINK_MQTT_PORT=1884",
            "-v", "${RuntimeDir}:/data",
            $ImageV1
        ) `
        -Description "Run inference service container"

    Wait-Container -Name $InferenceContainer -TimeoutSeconds 120

    $containerEnvironment = @(
        & docker inspect `
            --format "{{range .Config.Env}}{{println .}}{{end}}" `
            $InferenceContainer 2>&1 |
        ForEach-Object {
            "$_".Trim()
        }
    )

    if ($LASTEXITCODE -ne 0) {
        Stop-Pipeline `
            "Unable to inspect inference container environment variables."
    }

    if (
        "MODEL_PATH=/data/model.tflite" `
            -notin $containerEnvironment
    ) {
        Stop-Pipeline `
            "Inference container does not define MODEL_PATH=/data/model.tflite."
    }

    $modelPathCheck = @(
        & docker exec `
            $InferenceContainer `
            sh `
            -c `
            'test -f "$MODEL_PATH" && echo MODEL_OK' `
            2>&1
    )

    if (
        $LASTEXITCODE -ne 0 -or
        (($modelPathCheck -join "`n") -notmatch '(?im)^MODEL_OK$')
    ) {
        Stop-Pipeline `
            "MODEL_PATH does not point to an existing model inside the container."
    }

    Write-Pass `
        "Inference container MODEL_PATH validated"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("logs", $InferenceContainer, "--tail", "100") `
        -Description "Display inference service startup logs"

    # Inspect mounts as JSON rather than using a Docker Go template.
    # This is reliable in Windows PowerShell and avoids template quote parsing errors.
    $inspectJson = & docker inspect $InferenceContainer 2>&1
    $inspectExitCode = $LASTEXITCODE

    if ($inspectExitCode -ne 0) {
        Stop-Pipeline "Unable to inspect container mounts. Docker exit code: $inspectExitCode"
    }

    try {
        $inspectData = $inspectJson | ConvertFrom-Json
    }
    catch {
        Stop-Pipeline "Docker inspect returned invalid JSON: $($_.Exception.Message)"
    }

    $dataMount = @(
        $inspectData[0].Mounts |
        Where-Object { $_.Destination -eq "/data" }
    )

    $inspectData[0].Mounts | ForEach-Object {
        Write-Host ("{0} -> {1}" -f $_.Source, $_.Destination)
    }

    if ($dataMount.Count -eq 0) {
        Stop-Pipeline "Runtime directory is not mounted to /data."
    }

    Write-Pass ("Inference container is running with /data mounted from {0}" -f $dataMount[0].Source)

    # ---------------------------------------------------------------------
    # Task 22
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT C - TASK C1 MQTT EVIDENCE
    # Pipeline split: Task 22 provides the MQTT monitoring command for inference-topic verification.
    # =====================================================================
    $script:CurrentTask = "Task 22"
    Write-Section "Task 22 - MQTT Monitor Command"

    Write-Host "Open a second PowerShell window and run:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "docker exec -it $LocalBroker mosquitto_sub -h localhost -p 1883 -t `"logibridge/#`" -v" -ForegroundColor White
    Write-Host ""
    Write-Host "The automated pipeline continues without opening an interactive subscriber." -ForegroundColor DarkGray
    Write-Pass "Correct MQTT monitor command prepared"

    # ---------------------------------------------------------------------
    # Tasks 23-26
    # ---------------------------------------------------------------------
    $simulatorPath = Get-SimulatorPath

    if ($SkipSimulation) {
        Write-Section "Tasks 23 to 26 - Simulation Tests Skipped"
        Write-Warn "Simulation, database, offline storage and replay tests were skipped by -SkipSimulation."
    }
    elseif ($null -eq $simulatorPath) {
        Write-Section "Tasks 23 to 26 - Simulator Missing"
        Write-Warn @"
No MQTT sensor simulator was found.

Expected one of:
  data_pipeline\simulator.py
  simulation\sensor_simulator.py
  demo\sensor_simulator.py
  simulation\mqtt_sensor_simulator.py
  demo\sensor_publisher.py

Tasks 23-26 require a simulator that exposes --duration and either --truck-id
or --truck_id, and publishes the MQTT topics and payload schema expected by the
inference service.

Continue with registry and Ansible stages. Add the correct simulator and rerun
without -SkipSimulation to validate Tasks 23-26.
"@
    }
    else {
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT C - TASK C1 SENSOR SIMULATION
        # Pipeline split: Task 23 publishes realistic cold-chain sensor streams and anomaly modes.
        # =====================================================================
        $script:CurrentTask = "Task 23"
        Write-Section "Task 23 - Publish Simulation"
        Write-Host "[INFO] Simulator: $simulatorPath" -ForegroundColor Cyan

        $inferenceEvidence = Join-Path `
            $EvidenceDir `
            "manual_task23_inference_topic.log"

        Remove-Item -LiteralPath $inferenceEvidence -Force -ErrorAction SilentlyContinue

        $inferenceSubscriber = Start-Process `
            -FilePath "docker" `
            -ArgumentList @(
                "exec",
                $LocalBroker,
                "mosquitto_sub",
                "-h", "localhost",
                "-p", "1883",
                "-t", "logibridge/trucks/$TruckId/inference",
                "-C", "1",
                "-W", "60",
                "-v"
            ) `
            -RedirectStandardOutput $inferenceEvidence `
            -RedirectStandardError "$inferenceEvidence.stderr" `
            -PassThru `
            -NoNewWindow

        Invoke-Simulation `
            -SimulatorPath $simulatorPath `
            -Duration $SimulationDuration `
            -UseSpeed

        $inferenceSubscriber.WaitForExit(65000)
        if (-not $inferenceSubscriber.HasExited) {
            $inferenceSubscriber.Kill()
            $inferenceSubscriber.WaitForExit()
        }

        Assert-Path `
            -Path $inferenceEvidence `
            -Description "inference MQTT topic evidence" `
            -PathType Leaf

        $inferenceText = Get-Content -LiteralPath $inferenceEvidence -Raw
        $expectedTopic = "logibridge/trucks/$TruckId/inference"

        if ($inferenceText -notmatch [regex]::Escape($expectedTopic)) {
            Stop-Pipeline "Expected inference topic was not observed: $expectedTopic"
        }

        Write-Pass "Required inference MQTT topic validated"

        Start-Sleep -Seconds 5

        Invoke-Native `
            -FilePath "docker" `
            -Arguments @("logs", $InferenceContainer, "--tail", "150") `
            -Description "Inspect inference logs after simulation"

        Write-Pass "Sensor simulation command completed"

        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT C - TASK C2 AND OFFLINE-FIRST AUDIT EVIDENCE
        # Pipeline split: Task 24 verifies local inference records in SQLite.
        # =====================================================================
        $script:CurrentTask = "Task 24"
        Write-Section "Task 24 - Verify Database"

        Show-DatabaseSummary

        $recordCount = Invoke-SqlScalar `
            -Query "select count(*) from inference_records"

        if ($null -eq $recordCount) {
            Stop-Pipeline "Unable to query inference_records."
        }

        if ([int64]$recordCount -le 0) {
            Stop-Pipeline "Simulation completed but no inference records were created."
        }

        Write-Pass "Database contains $recordCount inference record(s)"

        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: OFFLINE-FIRST DEPLOYMENT REQUIREMENT
        # Supports Components C, D and E by proving inference storage continues when the uplink is unavailable.
        # =====================================================================
        $script:CurrentTask = "Task 25"
        Write-Section "Task 25 - Test Offline Storage"

        # Capture the current database count so this task proves that the
        # offline simulation creates NEW inference records.
        $beforeTotal = Invoke-SqlScalar `
            -Query "select count(*) from inference_records"

        if ($null -eq $beforeTotal) {
            $beforeTotal = 0
        }

        Write-Host "[INFO] Records before offline test: $beforeTotal" `
            -ForegroundColor Cyan

        Invoke-Native `
            -FilePath "docker" `
            -Arguments @("compose", "-f", $ComposeFile, "stop", "uplink-broker") `
            -Description "Stop uplink broker"

        try {
            # Restarting only the inference container clears the previous
            # simulator window/buffer state. The local broker remains online,
            # while the uplink broker remains offline.
            Write-Host "[WAIT] Restart inference service with uplink unavailable..." `
                -ForegroundColor Yellow

            Invoke-Native `
                -FilePath "docker" `
                -Arguments @("restart", $InferenceContainer) `
                -Description "Restart inference service and clear sensor buffers"

            Wait-Container `
                -Name $InferenceContainer `
                -TimeoutSeconds 60

            # Allow the service to connect to the local broker and confirm that
            # the uplink path is unavailable before publishing test data.
            Start-Sleep -Seconds 15

            Invoke-Native `
                -FilePath "docker" `
                -Arguments @("logs", $InferenceContainer, "--tail", "100") `
                -Description "Confirm offline inference service startup"

            Invoke-Simulation `
                -SimulatorPath $simulatorPath `
                -Duration $OfflineSimulationDuration

            # Allow the final 30-second feature window to be processed and
            # committed to SQLite.
            Start-Sleep -Seconds 10

            $afterTotal = Invoke-SqlScalar `
                -Query "select count(*) from inference_records"

            $unsynced = Invoke-SqlScalar `
                -Query "select count(*) from inference_records where inference_synced=0"

            if ($null -eq $afterTotal) {
                Stop-Pipeline "Unable to count inference records after offline simulation."
            }

            if ($null -eq $unsynced) {
                Stop-Pipeline "Unable to query inference_synced column."
            }

            $newRecords = [int64]$afterTotal - [int64]$beforeTotal

            Write-Host "[INFO] Records before test : $beforeTotal"
            Write-Host "[INFO] Records after test  : $afterTotal"
            Write-Host "[INFO] New offline records : $newRecords"
            Write-Host "[INFO] Unsynced records    : $unsynced"

            Invoke-Native `
                -FilePath "docker" `
                -Arguments @("logs", $InferenceContainer, "--tail", "150") `
                -Description "Inspect offline inference logs"

            if ($newRecords -le 0) {
                Stop-Pipeline "Sensor messages were published, but no new inference records were generated."
            }

            if ([int64]$unsynced -le 0) {
                Stop-Pipeline "New inference records were generated, but none were marked unsynced. Check inference_service.py SQLite sync logic."
            }

            Write-Pass "Offline storage contains $unsynced unsynced record(s); new offline records=$newRecords"
        }
        finally {
            # Always restore the uplink broker, even when Task 25 fails.
            & docker compose -f $ComposeFile start uplink-broker | Out-Host
        }

        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: OFFLINE-FIRST SYNCHRONISATION REQUIREMENT
        # Supports Components C, D and E by proving queued records replay when connectivity returns.
        # =====================================================================
        $script:CurrentTask = "Task 26"
        Write-Section "Task 26 - Replay"

        Wait-Container -Name $UplinkBroker -TimeoutSeconds 60

        Write-Step "Restart inference service to reconnect uplink and replay backlog"

        Invoke-Native `
            -FilePath "docker" `
            -Arguments @("restart", $InferenceContainer) `
            -Description "Restart inference service after uplink recovery"

        Wait-Container `
            -Name $InferenceContainer `
            -TimeoutSeconds 60

        Start-Sleep -Seconds 20
        Invoke-Native `
            -FilePath "docker" `
            -Arguments @("logs", $InferenceContainer, "--tail", "250") `
            -Description "Inspect replay logs"

        $remainingUnsynced = Invoke-SqlScalar `
            -Query "select count(*) from inference_records where inference_synced=0"

        if ($null -eq $remainingUnsynced) {
            Stop-Pipeline "Unable to verify replay state."
        }

        if ([int64]$remainingUnsynced -ne 0) {
            Stop-Pipeline "Replay did not clear the backlog; unsynced=$remainingUnsynced."
        }

        Write-Pass "Uplink replay completed and backlog returned to zero"
    }

    # ---------------------------------------------------------------------
    # Task 27
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: COMPONENT E - EDGE MLOPS: MONITORING AND DEPLOYMENT MANAGEMENT [MODULE 5]
    # Assignment Task E2 - Ansible Deployment Playbook
    # Pipeline split: Task 27 starts and populates the local Docker registry.
    # =====================================================================
    $script:CurrentTask = "Task 27"
    Write-Section "Task 27 - Local Registry"

    & docker rm -f $RegistryContainer 2>$null | Out-Null

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @(
            "run", "-d",
            "-p", "5000:5000",
            "--restart", "unless-stopped",
            "--name", $RegistryContainer,
            "registry:2"
        ) `
        -Description "Start local Docker registry"

    Wait-Container -Name $RegistryContainer -TimeoutSeconds 60

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("tag", $ImageV1, $RegistryImage) `
        -Description "Tag v1 image as registry v2"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("push", $RegistryImage) `
        -Description "Push v2 image to local registry"

    Write-Pass "Image pushed to $RegistryImage"

        # ---------------------------------------------------------------------
        # Task 28
        # ---------------------------------------------------------------------
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT E - TASK E2 DEPLOYMENT PREREQUISITE
        # Pipeline split: Task 28 verifies the required image and v2 tag in the local registry.
        # =====================================================================
        $script:CurrentTask = "Task 28"
        Write-Section "Task 28 - Verify Registry"

        $catalog = Invoke-RestMethod `
            -Uri "http://localhost:5000/v2/_catalog" `
            -Method Get

        $tags = Invoke-RestMethod `
            -Uri "http://localhost:5000/v2/logibridge/inference/tags/list" `
            -Method Get

        Write-Host "Repositories: $($catalog.repositories -join ', ')"
        Write-Host "Tags: $($tags.tags -join ', ')"

        if ($catalog.repositories -notcontains "logibridge/inference") {
            Stop-Pipeline "Registry catalog does not contain logibridge/inference."
        }

        if ($tags.tags -notcontains "v2") {
            Stop-Pipeline "Registry does not contain the v2 tag."
        }

        Write-Pass "Registry contains logibridge/inference:v2"

        # ---------------------------------------------------------------------
        # Fix local registry references before Ansible
        # ---------------------------------------------------------------------
        $script:CurrentTask = "Registry hostname correction"
        Write-Section "Correct Local Registry References"
        Update-LocalRegistryReferences

        # ---------------------------------------------------------------------
        # Prepare WSL project path for Tasks 29 and 30
        # ---------------------------------------------------------------------
        if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
            Stop-Pipeline "ProjectRoot is empty and cannot be converted to a WSL path."
        }

        if ($ProjectRoot -notmatch '^[A-Za-z]:\\') {
            Stop-Pipeline "ProjectRoot is not a valid Windows drive path: $ProjectRoot"
        }

        $wslDrive = $ProjectRoot.Substring(0, 1).ToLowerInvariant()
        $wslRelativePath = $ProjectRoot.Substring(2).Replace("\", "/")
        $wslProjectRoot = "/mnt/$wslDrive$wslRelativePath"

        Write-Host "Windows project root : $ProjectRoot" -ForegroundColor Cyan
        Write-Host "WSL project root     : $wslProjectRoot" -ForegroundColor Cyan

        # Remove unsupported Bash options that may have been inherited from
        # earlier shell experiments.
        Remove-Item Env:SHELLOPTS -ErrorAction SilentlyContinue
        Remove-Item Env:BASH_ENV -ErrorAction SilentlyContinue
        Remove-Item Env:WSLENV -ErrorAction SilentlyContinue

        # ---------------------------------------------------------------------
        # Task 29
        # ---------------------------------------------------------------------
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT E - TASK E2 ANSIBLE CONTRACT
        # Pipeline split: Task 29 verifies the exact seven-task playbook and runs the Ansible syntax check.
        # =====================================================================
        $script:CurrentTask = "Task 29"
        Write-Section "Task 29 - Verify Exact Ansible Contract and Syntax"

        $playbookValidator = Join-Path `
            $DeploymentDir `
            "validate_playbook.py"

        Assert-Path `
            -Path $playbookValidator `
            -Description "Ansible playbook validator" `
            -PathType Leaf

        Invoke-Native `
            -FilePath $python `
            -Arguments @($playbookValidator) `
            -Description "Validate exact seven-task Ansible contract" |
            Out-Null

        # Use a one-line Bash command to prevent Windows CRLF characters
        # from being interpreted by Bash as part of the command.
        $syntaxCommand = (
            "set -e; " +
            "cd '$wslProjectRoot'; " +
            "ansible-playbook --syntax-check " +
            "-i deployment/inventory.ini " +
            "deployment/logibridge_deploy.yml"
        )

        Invoke-Native `
            -FilePath "wsl" `
            -Arguments @(
                "bash",
                "-lc",
                $syntaxCommand
            ) `
            -Description "Run Ansible syntax check" |
            Out-Null

        Write-Pass "Ansible syntax check passed"

        # ---------------------------------------------------------------------
        # Task 30
        # ---------------------------------------------------------------------
        # =====================================================================
        # ASSIGNMENT TERMINOLOGY: COMPONENT E - TASK E2 IDEMPOTENCY DEMONSTRATION
        # Pipeline split: Task 30 runs the playbook twice and requires changed=0 on the second run.
        # =====================================================================
        $script:CurrentTask = "Task 30"
        Write-Section "Task 30 - Run Deployment Twice"

        # Use the local demonstration target. The physical truck IP addresses
        # in the inventory are not expected to be reachable from this laptop.
        $deployCommand = (
            "set -e; " +
            "cd '$wslProjectRoot'; " +
            "ansible-playbook " +
            "-i deployment/inventory.ini " +
            "deployment/logibridge_deploy.yml " +
            "--limit localhost_demo"
        )

        $firstRun = Invoke-Native `
            -FilePath "wsl" `
            -Arguments @(
                "bash",
                "-lc",
                $deployCommand
            ) `
            -Description "Run first Ansible deployment" `
            -CaptureOutput

        $firstText = $firstRun.Output -join "`n"

        if ($firstText -notmatch "failed=0") {
            Stop-Pipeline `
                "First Ansible deployment did not report failed=0."
        }

        Write-Pass "First Ansible deployment completed with failed=0"

        $secondRun = Invoke-Native `
            -FilePath "wsl" `
            -Arguments @(
                "bash",
                "-lc",
                $deployCommand
            ) `
            -Description "Run second Ansible deployment for idempotency" `
            -CaptureOutput

        $secondText = $secondRun.Output -join "`n"

        if ($secondText -notmatch "failed=0") {
            Stop-Pipeline `
                "Second Ansible deployment did not report failed=0."
        }

        if ($secondText -notmatch "changed=0") {
            Stop-Pipeline `
                "Second Ansible run must report changed=0 and failed=0."
        }

        Write-Pass `
            "Second Ansible run is idempotent: changed=0, failed=0"

    # ---------------------------------------------------------------------
    # Task 31
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: FINAL ASSIGNMENT VERIFICATION - COMPONENTS A-F
    # Pipeline split: Task 31 verifies the complete set of required runtime and evidence artifacts.
    # =====================================================================
    $script:CurrentTask = "Task 31"
    Write-Section "Task 31 - Final Verification"

    # Ansible may recreate the inference container. Wait for it to become
    # fully ready before checking logs, mounts, and the runtime database.
    Wait-Container -Name $LocalBroker -TimeoutSeconds 60
    Wait-Container -Name $UplinkBroker -TimeoutSeconds 60
    Wait-Container -Name $InferenceContainer -TimeoutSeconds 120
    Wait-Container -Name $RegistryContainer -TimeoutSeconds 60

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("ps") `
        -Description "List running containers"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @(
            "images",
            "--format",
            "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}"
        ) `
        -Description "List Docker images"

    Show-DatabaseSummary

    $finalTags = Invoke-RestMethod `
        -Uri "http://localhost:5000/v2/logibridge/inference/tags/list" `
        -Method Get

    Write-Host "Registry verification: $($finalTags | ConvertTo-Json -Compress)"

    foreach ($container in @(
        $LocalBroker,
        $UplinkBroker,
        $InferenceContainer,
        $RegistryContainer
    )) {
        if (-not (Get-ContainerRunning -Name $container)) {
            Stop-Pipeline "Required container is not running: $container"
        }
    }

    Write-Pass "All required containers are running"

    # ---------------------------------------------------------------------
    # Task 32
    # ---------------------------------------------------------------------
    # =====================================================================
    # ASSIGNMENT TERMINOLOGY: FINAL ASSIGNMENT RESULT - COMPONENTS A-F
    # Pipeline split: Task 32 reports the end-to-end validation result without changing any component logic.
    # =====================================================================
    $script:CurrentTask = "Task 32"
    Write-Section "Task 32 - Complete Pipeline Result"

    Write-Host "Passed checks : $script:Passed" -ForegroundColor Green
    Write-Host "Warnings      : $script:Warnings" -ForegroundColor Yellow
    Write-Host "Evidence log  : $PipelineLog" -ForegroundColor Cyan

    if ($SubmissionValidation -and $script:Warnings -gt 0) {
        Stop-Pipeline `
            "Submission validation completed with $script:Warnings warning(s)."
    }

    if ($script:Warnings -gt 0) {
        Write-Host ""
        Write-Host `
            "[COMPLETE WITH WARNINGS] Review the warning messages above." `
            -ForegroundColor Yellow
    }
    else {
        Write-Host ""
        Write-Host `
            "[SUCCESS] Tasks 1-32 completed successfully." `
            -ForegroundColor Green
    }

    exit 0
}
catch {
    Write-Host ""
    Write-Host "[FAILED] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Current stage: $script:CurrentTask" -ForegroundColor Red

    if (Get-Command docker -ErrorAction SilentlyContinue) {
        Write-Host ""
        Write-Host "Recent inference logs:" -ForegroundColor Yellow
        try {
            $recentLogs = & docker logs $InferenceContainer --tail 100 2>&1
            $recentLogs | ForEach-Object { Write-Host "$_" }
        }
        catch {
            Write-Warning "Could not retrieve inference container logs."
        }

        Write-Host ""
        Write-Host "Container status:" -ForegroundColor Yellow
        try {
            $containerStatus = & docker ps -a 2>&1
            $containerStatus | ForEach-Object { Write-Host "$_" }
        }
        catch {
            Write-Warning "Could not retrieve Docker container status."
        }
    }

    Write-Host ""
    Write-Host "Evidence log: $PipelineLog" -ForegroundColor Cyan
    exit 1
}
finally {
    Stop-TranscriptSafe
}