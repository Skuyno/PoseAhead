param(
    [Parameter(Mandatory = $true)]
    [string]$UnityEditor,
    [string]$Recording
)

$ErrorActionPreference = 'Stop'
$poseAheadRepository = Split-Path -Parent $PSScriptRoot
$poseAheadProject = Join-Path $poseAheadRepository 'unity/PoseAheadAvatar'
$poseAheadEditorPath = (Resolve-Path -LiteralPath $UnityEditor).Path
$poseAheadPinnedVersion = (Get-Content -LiteralPath (Join-Path $poseAheadProject 'ProjectSettings/ProjectVersion.txt') |
    Where-Object { $_ -match '^m_EditorVersion: ' }) -replace '^m_EditorVersion: ', ''
$poseAheadInstalledVersion = (Get-Item -LiteralPath $poseAheadEditorPath).VersionInfo.ProductVersion
if ($poseAheadInstalledVersion -ne $poseAheadPinnedVersion -and
    -not $poseAheadInstalledVersion.StartsWith($poseAheadPinnedVersion + '_')) {
    throw "Use pinned Unity $poseAheadPinnedVersion; supplied Editor is $poseAheadInstalledVersion."
}

$poseAheadRecordingPath = $null
if ($Recording) { $poseAheadRecordingPath = (Resolve-Path -LiteralPath $Recording).Path }
$poseAheadRunName = 'unity_native_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$poseAheadOutput = Join-Path (Join-Path $poseAheadRepository 'outputs') $poseAheadRunName
New-Item -ItemType Directory -Path $poseAheadOutput -ErrorAction Stop | Out-Null
$poseAheadXmlPath = Join-Path $poseAheadOutput 'editmode.xml'
$poseAheadLogPath = Join-Path $poseAheadOutput 'editor.log'
$poseAheadPreviousRecording = $env:POSEAHEAD_UNITY_REPLAY_RECORDING
$poseAheadProcess = $null
try {
    # Scoped to this process and the Editor child, never a persistent environment setting.
    $env:POSEAHEAD_UNITY_REPLAY_RECORDING = $poseAheadRecordingPath
    $poseAheadArguments = @(
        '-batchmode', '-nographics', '-projectPath', ('"' + $poseAheadProject + '"'),
        '-runTests', '-testPlatform', 'EditMode', '-testResults', ('"' + $poseAheadXmlPath + '"'),
        '-logFile', ('"' + $poseAheadLogPath + '"')
    )
    $poseAheadProcess = Start-Process -FilePath $poseAheadEditorPath -ArgumentList $poseAheadArguments -WindowStyle Hidden -PassThru
    # Wait for this Editor, not for the whole Hub/licensing descendant process tree.
    if (-not $poseAheadProcess.WaitForExit(300000)) {
        $poseAheadProcess.Kill() # Only the exact Editor instance started above.
        throw "Unity test timeout; logs retained in $poseAheadOutput"
    }
    $poseAheadProcess.Refresh()
    if ($poseAheadProcess.ExitCode -ne 0) {
        throw "Unity exited $($poseAheadProcess.ExitCode); inspect $poseAheadLogPath"
    }
    if (-not (Test-Path -LiteralPath $poseAheadXmlPath -PathType Leaf)) {
        throw "Unity did not produce test results; inspect $poseAheadLogPath"
    }
    [xml]$poseAheadXml = Get-Content -LiteralPath $poseAheadXmlPath -Raw
    $poseAheadResult = $poseAheadXml.DocumentElement
    $poseAheadExpectedPassed = if ($poseAheadRecordingPath) { 4 } else { 3 }
    $poseAheadExpectedSkipped = if ($poseAheadRecordingPath) { 0 } else { 1 }
    # This pinned NUnit runner marks the root Skipped:Ignored when one optional test is ignored.
    $poseAheadExpectedResult = if ($poseAheadRecordingPath) { 'Passed' } else { 'Skipped:Ignored' }
    if ($poseAheadResult.GetAttribute('result') -ne $poseAheadExpectedResult -or
        [int]$poseAheadResult.GetAttribute('total') -ne 4 -or
        [int]$poseAheadResult.GetAttribute('passed') -ne $poseAheadExpectedPassed -or
        [int]$poseAheadResult.GetAttribute('skipped') -ne $poseAheadExpectedSkipped -or
        [int]$poseAheadResult.GetAttribute('failed') -ne 0) {
        throw "Unexpected test counts/results (including an empty discovery); inspect $poseAheadXmlPath"
    }
    $poseAheadExpectedNames = @(
        'BuiltHumanoidCalibratesAndNumericRotationsAgreeWithUnity',
        'DemoFrameRateLimitRestoresSettingsAndHandlesOverlap',
        'ActualUdpDrivesEightPhasesAndHoldsExpiredPose',
        'PythonRecordingReplayReachesNativeAvatar'
    )
    foreach ($poseAheadTest in $poseAheadXml.SelectNodes('//test-case')) {
        $poseAheadTestName = $poseAheadTest.GetAttribute('name')
        $poseAheadTestResult = if ($poseAheadTestName -eq 'PythonRecordingReplayReachesNativeAvatar' -and
            -not $poseAheadRecordingPath) { 'Skipped' } else { 'Passed' }
        if ($poseAheadTestName -notin $poseAheadExpectedNames -or
            $poseAheadTest.GetAttribute('result') -ne $poseAheadTestResult) {
            throw "Unexpected native test: $poseAheadTestName; inspect $poseAheadXmlPath"
        }
    }
    Write-Output "Native Unity: $poseAheadExpectedPassed passed, $poseAheadExpectedSkipped optional replay skipped."
    Write-Output "Results: $poseAheadXmlPath"
} finally {
    $env:POSEAHEAD_UNITY_REPLAY_RECORDING = $poseAheadPreviousRecording
    if ($poseAheadProcess) { $poseAheadProcess.Dispose() }
}
