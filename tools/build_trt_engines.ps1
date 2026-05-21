# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [ValidateSet("all", "onnx", "engine")]
    [string]$Stage = "all",

    [ValidateSet("all", "onnx", "sim", "engine")]
    [string]$PatchPriorStage = "all",

    [int]$PatchPriorWorkspaceGb = 16,
    [int]$FullPriorWorkspaceGb = 24,

    [int[]]$PreviewBatches = @(16, 5),

    [switch]$SkipPatchPrior,
    [switch]$SkipFullPrior512,
    [switch]$SkipBackbone,
    [switch]$IncludeFullResolutionBackbone,
    [switch]$UseSystemPython,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TrtDir = Join-Path $RepoRoot "checkpoints\trt"

function Get-PythonDisplay {
    param([string[]]$Arguments)

    if ($UseSystemPython) {
        return "python " + ($Arguments -join " ")
    }
    return "uv run python " + ($Arguments -join " ")
}

function Invoke-Export {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    Write-Host (Get-PythonDisplay -Arguments $Arguments)

    if ($DryRun) {
        return
    }

    if ($UseSystemPython) {
        & python @Arguments
    } else {
        & uv run python @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Invoke-BackboneDecoderExport {
    param(
        [string]$Component,
        [int]$Height,
        [int]$Width,
        [int]$Batch
    )

    Invoke-Export `
        -Name "backbone $Component decoder, batch $Batch, input ${Height}x${Width}" `
        -Arguments @(
            "tools\export_backbone_trt.py",
            "--component", $Component,
            "--height", $Height.ToString(),
            "--width", $Width.ToString(),
            "--batch", $Batch.ToString(),
            "--precision", "fp32",
            "--stage", $Stage
        )
}

$PreviewDecoderShapes = @(
    @{ Component = "dec6"; Height = 11; Width = 16 },
    @{ Component = "dec5"; Height = 22; Width = 32 },
    @{ Component = "dec4"; Height = 44; Width = 64 },
    @{ Component = "dec3"; Height = 88; Width = 128 }
)

$FullResolutionDecoderShapes = @(
    @{ Component = "dec6"; Height = 51; Width = 77 },
    @{ Component = "dec5"; Height = 103; Width = 155 },
    @{ Component = "dec4"; Height = 206; Width = 310 },
    @{ Component = "dec3"; Height = 412; Width = 620 },
    @{ Component = "dec2"; Height = 824; Width = 1240 }
)

Push-Location $RepoRoot
try {
    Write-Host "OMNI-DC-MA TensorRT engine build"
    Write-Host "Repo: $RepoRoot"
    Write-Host "Output: $TrtDir"

    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $TrtDir | Out-Null
    }

    if (-not $SkipPatchPrior) {
        Invoke-Export `
            -Name "MA-depthmap patch prior" `
            -Arguments @(
                "tools\export_prior_trt.py",
                "--stage", $PatchPriorStage,
                "--workspace-gb", $PatchPriorWorkspaceGb.ToString()
            )
    }

    if (-not $SkipFullPrior512) {
        Invoke-Export `
            -Name "MA-depthmap fixed 352x512 prior" `
            -Arguments @(
                "tools\export_full_prior_512_trt.py",
                "--stage", $Stage,
                "--workspace-gb", $FullPriorWorkspaceGb.ToString()
            )
    }

    if (-not $SkipBackbone) {
        foreach ($Batch in $PreviewBatches) {
            foreach ($Shape in $PreviewDecoderShapes) {
                Invoke-BackboneDecoderExport `
                    -Component ([string]$Shape["Component"]) `
                    -Height ([int]$Shape["Height"]) `
                    -Width ([int]$Shape["Width"]) `
                    -Batch $Batch
            }
        }

        if ($IncludeFullResolutionBackbone) {
            foreach ($Shape in $FullResolutionDecoderShapes) {
                Invoke-BackboneDecoderExport `
                    -Component ([string]$Shape["Component"]) `
                    -Height ([int]$Shape["Height"]) `
                    -Width ([int]$Shape["Width"]) `
                    -Batch 1
            }
        }
    }

    Write-Host ""
    if ($DryRun) {
        Write-Host "Dry run complete. No files were written."
    } else {
        Write-Host "TensorRT build complete. Engines are under checkpoints\trt."
    }
}
finally {
    Pop-Location
}
