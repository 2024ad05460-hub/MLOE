<#
.SYNOPSIS
    LogiEdge complete end-to-end validation pipeline (Tasks 1-32).

.DESCRIPTION
    Executes the complete LogiEdge assignment pipeline:
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
    [switch]$NoRegistryReplacement
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

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

    # Capture both standard output and standard error as plain text. This avoids
    # PowerShell presenting harmless TensorFlow stderr warnings as command failures.
    $output = & $FilePath @Arguments 2>&1 | ForEach-Object { "$_" }
    $exitCode = $LASTEXITCODE

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
        -Include "*sensor*simulator*.py","*sensor*publisher*.py","*mqtt*simulator*.py" `
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
    $escapedDb = $RuntimeDatabase.Replace("\", "\\").Replace("'", "\'")
    $escapedQuery = $Query.Replace("\", "\\").Replace("'", "\'")

    $code = "import sqlite3; c=sqlite3.connect(r'$escapedDb'); print(c.execute('$escapedQuery').fetchone()[0])"
    $result = & $python -c $code

    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    return "$result".Trim()
}

function Show-DatabaseSummary {
    $python = Get-PythonExecutable

    if (-not (Test-Path -LiteralPath $RuntimeDatabase -PathType Leaf)) {
        Write-Warn "Database does not exist yet: $RuntimeDatabase"
        return
    }

    $code = @"
import sqlite3
p = r'$RuntimeDatabase'
c = sqlite3.connect(p)
tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'").fetchall()]
print("tables:", tables)
if "inference_records" in tables:
    print("labels:", c.execute("select label,count(*) from inference_records group by label").fetchall())
    cols = [r[1] for r in c.execute("pragma table_info(inference_records)").fetchall()]
    if "inference_synced" in cols:
        print("unsynced:", c.execute("select count(*) from inference_records where inference_synced=0").fetchone())
else:
    print("inference_records table not found")
"@

    & $python -c $code
    if ($LASTEXITCODE -ne 0) {
        Stop-Pipeline "Unable to query SQLite runtime database."
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
    $script:CurrentTask = "Task 8"
    Write-Section "Task 8 - Run Normalization Experiment"

    $normalisationScript = Join-Path $ProjectRoot "experiments\normalisation_experiment.py"
    Assert-Path -Path $normalisationScript -Description "normalization experiment script" -PathType Leaf

    Invoke-LoggedNative `
        -FilePath $python `
        -Arguments @($normalisationScript) `
        -LogPath (Join-Path $EvidenceDir "manual_task08_normalisation.log") `
        -Description "Run normalization experiment"

    Assert-Path `
        -Path (Join-Path $ProjectRoot "experiments\normalisation_experiment.csv") `
        -Description "normalization experiment CSV" `
        -PathType Leaf

    Write-Pass "Task 8 normalization experiment completed"

    # ---------------------------------------------------------------------
    # Task 9
    # ---------------------------------------------------------------------
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
    $script:CurrentTask = "Task 10"
    Write-Section "Task 10 - Validate Benchmark Columns and Variants"

    $benchmarkRows = Import-Csv -LiteralPath $benchmarkCsv
    $requiredColumns = @(
        "variant",
        "size_kb",
        "accuracy_pct",
        "recall_critical_pct",
        "mean_latency_ms",
        "p95_latency_ms",
        "energy_mj_per_inference"
    )

    if ($benchmarkRows.Count -eq 0) {
        Stop-Pipeline "Benchmark CSV is empty."
    }

    $actualColumns = $benchmarkRows[0].PSObject.Properties.Name
    $missingColumns = $requiredColumns | Where-Object { $_ -notin $actualColumns }

    if ($missingColumns) {
        Stop-Pipeline "Benchmark CSV is missing columns: $($missingColumns -join ', ')"
    }

    $benchmarkRows |
        Format-Table `
            variant,
            size_kb,
            accuracy_pct,
            recall_critical_pct,
            mean_latency_ms,
            p95_latency_ms,
            energy_mj_per_inference `
            -AutoSize

    $variants = @($benchmarkRows.variant)
    foreach ($expectedVariant in @("M1_FP32", "M2_PTQ_INT8", "M3_PRUNE35_INT8")) {
        if ($expectedVariant -notin $variants) {
            Stop-Pipeline "Benchmark variant missing: $expectedVariant"
        }
    }

    Write-Pass "Task 10 benchmark columns and all three variants validated"

    # ---------------------------------------------------------------------
    # Task 11
    # ---------------------------------------------------------------------
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

    Write-Pass "Task 14 PSI alert and recovery thresholds passed"

    # ---------------------------------------------------------------------
    # Task 15
    # ---------------------------------------------------------------------
    $script:CurrentTask = "Task 15"
    Write-Section "Task 15 - Run Automated Tests"

    if ($SkipTests) {
        Write-Warn "Task 15 skipped by -SkipTests."
    }
    else {
        Invoke-LoggedNative `
            -FilePath $python `
            -Arguments @("-m", "pytest", "-q") `
            -LogPath (Join-Path $EvidenceDir "manual_task15_pytest.log") `
            -Description "Run automated pytest suite"

        Write-Pass "Task 15 automated tests passed"
    }

    # ---------------------------------------------------------------------
    # Task 16
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Task 18
    # ---------------------------------------------------------------------
    $script:CurrentTask = "Task 18"
    Write-Section "Task 18 - Build Docker Image v1"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @(
            "build",
            "--progress=plain",
            "-t", $ImageV1,
            $InferenceDir
        ) `
        -Description "Build $ImageV1"

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @("image", "inspect", $ImageV1) `
        -Description "Verify $ImageV1"

    Write-Pass "Docker image $ImageV1 is available"

    # ---------------------------------------------------------------------
    # Task 19
    # ---------------------------------------------------------------------
    $script:CurrentTask = "Task 19"
    Write-Section "Task 19 - OTA Image Test"

    if ($SkipOtaBuild) {
        Write-Warn "Task 19 skipped by -SkipOtaBuild."
    }
    else {
        Invoke-Native `
            -FilePath "docker" `
            -Arguments @(
                "build",
                "--progress=plain",
                "-t", $ImageOta,
                $InferenceDir
            ) `
            -Description "Build $ImageOta"

        Invoke-Native `
            -FilePath "docker" `
            -Arguments @("image", "inspect", $ImageOta) `
            -Description "Verify $ImageOta"

        Write-Pass "OTA test image $ImageOta is available"
    }

    # ---------------------------------------------------------------------
    # Task 20
    # ---------------------------------------------------------------------
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
    $script:CurrentTask = "Task 21"
    Write-Section "Task 21 - Start Inference Container"

    # Remove an existing inference container only when it is present.
    # Calling `docker rm` for a missing container can terminate the pipeline
    # when native-command errors are promoted to PowerShell exceptions.
    $existingInferenceContainer = & docker ps -a `
        --filter "name=^/${InferenceContainer}$" `
        --format "{{.Names}}" 2>$null

    if ($LASTEXITCODE -ne 0) {
        Stop-Pipeline "Unable to check whether inference container exists."
    }

    if (("$existingInferenceContainer").Trim() -eq $InferenceContainer) {
        Invoke-Native `
            -FilePath "docker" `
            -Arguments @("rm", "-f", $InferenceContainer) `
            -Description "Remove previous inference container"
    }
    else {
        Write-Host "[INFO] No previous inference container found; creating a new one." `
            -ForegroundColor Cyan
    }

    Invoke-Native `
        -FilePath "docker" `
        -Arguments @(
            "run", "-d",
            "--name", $InferenceContainer,
            "--restart", "unless-stopped",
            "-e", "TRUCK_ID=$TruckId",
            "-e", "LOCAL_MQTT_HOST=host.docker.internal",
            "-e", "LOCAL_MQTT_PORT=1883",
            "-e", "UPLINK_MQTT_HOST=host.docker.internal",
            "-e", "UPLINK_MQTT_PORT=1884",
            "-v", "${RuntimeDir}:/data",
            $ImageV1
        ) `
        -Description "Run inference service container"

    Wait-Container -Name $InferenceContainer -TimeoutSeconds 120

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
  simulation\sensor_simulator.py
  demo\sensor_simulator.py
  simulation\mqtt_sensor_simulator.py
  demo\sensor_publisher.py

The file data_pipeline\simulator.py is a Python module unless it exposes an MQTT
command-line publisher. Tasks 23-26 cannot be honestly validated without a
publisher that uses the inference service's expected MQTT topics and payload schema.

Continue with registry and Ansible stages. Add the correct simulator and rerun
without -SkipSimulation to validate Tasks 23-26.
"@
    }
    else {
        $script:CurrentTask = "Task 23"
        Write-Section "Task 23 - Publish Simulation"
        Write-Host "[INFO] Simulator: $simulatorPath" -ForegroundColor Cyan

        Invoke-Simulation `
            -SimulatorPath $simulatorPath `
            -Duration $SimulationDuration `
            -UseSpeed

        Start-Sleep -Seconds 5

        Invoke-Native `
            -FilePath "docker" `
            -Arguments @("logs", $InferenceContainer, "--tail", "150") `
            -Description "Inspect inference logs after simulation"

        Write-Pass "Sensor simulation command completed"

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

        $script:CurrentTask = "Task 26"
        Write-Section "Task 26 - Replay"

        Wait-Container -Name $UplinkBroker -TimeoutSeconds 60
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
    # Tasks 29-30
    # ---------------------------------------------------------------------
    if ($SkipAnsible) {
        Write-Section "Tasks 29 and 30 - Ansible Skipped"
        Write-Warn "Ansible syntax and deployment tests were skipped by -SkipAnsible."
    }
    else {
        Assert-Path -Path $InventoryFile -Description "Ansible inventory" -PathType Leaf
        Assert-Path -Path $PlaybookFile -Description "Ansible playbook" -PathType Leaf

        $wslProjectRoot = "/mnt/" + `
            $ProjectRoot.Substring(0, 1).ToLower() + `
            $ProjectRoot.Substring(2).Replace("\", "/")

        $script:CurrentTask = "Task 29"
        Write-Section "Task 29 - Verify Ansible Syntax"

        $syntaxCommand = @"
set -e
cd '$wslProjectRoot'
ansible-playbook --syntax-check \
  -i deployment/inventory.ini \
  deployment/logibridge_deploy.yml
"@

        Invoke-Native `
            -FilePath "wsl" `
            -Arguments @("bash", "-lc", $syntaxCommand) `
            -Description "Run Ansible syntax check"

        Write-Pass "Ansible syntax check passed"

        $script:CurrentTask = "Task 30"
        Write-Section "Task 30 - Run Deployment Twice"

        $deployCommand = @"
set -e
cd '$wslProjectRoot'
ansible-playbook \
  -i deployment/inventory.ini \
  deployment/logibridge_deploy.yml \
  --limit localhost_demo
"@

        $firstRun = Invoke-Native `
            -FilePath "wsl" `
            -Arguments @("bash", "-lc", $deployCommand) `
            -Description "Run first Ansible deployment" `
            -CaptureOutput

        if (($firstRun.Output -join "`n") -notmatch "failed=0") {
            Stop-Pipeline "First Ansible deployment did not report failed=0."
        }

        $secondRun = Invoke-Native `
            -FilePath "wsl" `
            -Arguments @("bash", "-lc", $deployCommand) `
            -Description "Run second Ansible deployment for idempotency" `
            -CaptureOutput

        $secondText = $secondRun.Output -join "`n"

        if ($secondText -notmatch "failed=0") {
            Stop-Pipeline "Second Ansible deployment did not report failed=0."
        }

        if ($secondText -match "changed=0") {
            Write-Pass "Second Ansible run is idempotent: changed=0, failed=0"
        }
        else {
            Write-Warn "Second Ansible run passed but did not report changed=0. Review changed tasks."
        }
    }

    # ---------------------------------------------------------------------
    # Task 31
    # ---------------------------------------------------------------------
    $script:CurrentTask = "Task 31"
    Write-Section "Task 31 - Final Verification"

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
    $script:CurrentTask = "Task 32"
    Write-Section "Task 32 - Complete Pipeline Result"

    Write-Host "Passed checks : $script:Passed" -ForegroundColor Green
    Write-Host "Warnings      : $script:Warnings" -ForegroundColor Yellow
    Write-Host "Evidence log  : $PipelineLog" -ForegroundColor Cyan

    if ($script:Warnings -gt 0) {
        Write-Host ""
        Write-Host "[COMPLETE WITH WARNINGS] Review the warning messages above." -ForegroundColor Yellow
        Write-Host "The most likely warning is the absent MQTT sensor publisher." -ForegroundColor Yellow
    }
    else {
        Write-Host ""
        Write-Host "[SUCCESS] Tasks 1-32 completed successfully." -ForegroundColor Green
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