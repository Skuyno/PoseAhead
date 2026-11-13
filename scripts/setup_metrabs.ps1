$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'native_command.ps1')

$repoRoot = Split-Path -Parent $PSScriptRoot
$constraints = Join-Path $repoRoot 'requirements\metrabs-constraints.txt'
$vendorRoot = Join-Path $repoRoot '.vendor'
$sourceRoot = Join-Path $vendorRoot 'metrabs-source'
$modelName = 'metrabs_eff2s_256px_800k_28ds_pytorch'
$modelRoot = Join-Path $vendorRoot $modelName
$archive = Join-Path $vendorRoot ($modelName + '.tar.gz')
$yoloPath = Join-Path $vendorRoot 'yolov8n.pt'
$expectedCommit = '8b2b116dd27372e7dbd8207809f868df4e3f852e'
$archiveHash = 'D6861899D0C05A93A4BAE078DCA5382186F04D568255EB688ACE1C7DEA7A7634'
$yoloHash = '31E20DDE3DEF09E2CF938C7BE6FE23D9150BBBE503982AF13345706515F2EF95'
$modelUrl = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2s_256px_800k_28ds_pytorch.tar.gz'
$yoloUrl = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'

function Assert-FileHash {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Expected
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -ne $Expected) {
        throw "SHA-256 mismatch for $Path`: $actual (expected $Expected)"
    }
}

Push-Location -LiteralPath $repoRoot
try {
    $pythonExe = (Invoke-NativeCommand poetry @('env', 'info', '--executable')).Trim()
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
        throw "Poetry environment is missing. Run 'poetry install' first."
    }
    Invoke-NativeCommand $pythonExe @('-c', 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)')
    Invoke-NativeCommand $pythonExe @(
        '-m', 'pip', 'install', 'torch==2.1.0+cu121', 'torchvision==0.16.0+cu121',
        '--index-url', 'https://download.pytorch.org/whl/cu121', '--constraint', $constraints
    )
    Invoke-NativeCommand $pythonExe @(
        '-m', 'pip', 'install', 'opencv-python==4.11.0.86', 'ultralytics==8.0.145',
        'einops==0.8.1', '--constraint', $constraints
    )

    New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot '.git') -PathType Container)) {
        Invoke-NativeCommand git @(
            'clone', '--filter=blob:none', '--no-checkout',
            'https://github.com/isarandi/metrabs.git', $sourceRoot
        )
        Invoke-NativeCommand git @('-C', $sourceRoot, 'checkout', $expectedCommit)
    }
    $actualCommit = (Invoke-NativeCommand git @('-C', $sourceRoot, 'rev-parse', 'HEAD')).Trim()
    if ($actualCommit -ne $expectedCommit) {
        throw "Unexpected MeTRAbs revision: $actualCommit (expected $expectedCommit)"
    }

    if (-not (Test-Path -LiteralPath $modelRoot -PathType Container)) {
        Invoke-NativeCommand curl.exe @('-L', '--fail', '--retry', '3', $modelUrl, '-o', $archive)
        Assert-FileHash -Path $archive -Expected $archiveHash
        try {
            Invoke-NativeCommand tar @('-xf', $archive, '-C', $vendorRoot)
        }
        finally {
            if (Test-Path -LiteralPath $archive -PathType Leaf) {
                Remove-Item -LiteralPath $archive
            }
        }
    }
    elseif (Test-Path -LiteralPath $archive -PathType Leaf) {
        # A successful extraction makes the large archive redundant.
        Remove-Item -LiteralPath $archive
    }
    if (-not (Test-Path -LiteralPath $yoloPath -PathType Leaf)) {
        Invoke-NativeCommand curl.exe @('-L', '--fail', '--retry', '3', $yoloUrl, '-o', $yoloPath)
    }
    Assert-FileHash -Path $yoloPath -Expected $yoloHash

    Invoke-NativeCommand $pythonExe @('-m', 'pip', 'check')
    Invoke-NativeCommand $pythonExe @(
        '-c', 'from poseahead.perception.metrabs import MetrabsBackend; MetrabsBackend()'
    )
    Write-Host 'MeTRAbs model load: PASS'
    Write-Host "MeTRAbs environment checks passed at commit $actualCommit"
}
finally { Pop-Location }
