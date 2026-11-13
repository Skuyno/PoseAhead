param(
    [Parameter(Mandatory = $true)]
    [string]$UnityEditor
)
$ErrorActionPreference = 'Stop'
$poseAheadEditor = Get-Item -LiteralPath $UnityEditor
if ($poseAheadEditor.Name -ne 'Unity.exe' -or $poseAheadEditor.PSIsContainer) {
    throw 'Select the actual Unity Editor Unity.exe, not Hub or a directory.'
}
Write-Output $poseAheadEditor.VersionInfo.ProductVersion
