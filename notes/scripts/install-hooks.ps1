# install-hooks.ps1 — copy hooks from notes/scripts/hooks/ into .git/hooks/
#
# Usage (from repo root):
#   .\notes\scripts\install-hooks.ps1

$ErrorActionPreference = 'Stop'

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$HooksSource = Join-Path $ScriptDir 'hooks'
$GitRoot     = git rev-parse --show-toplevel
$HooksDest   = Join-Path $GitRoot '.git' 'hooks'

if (-not (Test-Path $HooksSource)) {
    Write-Error "Hooks directory not found: $HooksSource"
    exit 1
}

if (-not (Test-Path $HooksDest)) {
    New-Item -ItemType Directory -Path $HooksDest | Out-Null
}

$installed = 0
Get-ChildItem -Path $HooksSource -File | ForEach-Object {
    $dest = Join-Path $HooksDest $_.Name
    Copy-Item -Path $_.FullName -Destination $dest -Force
    Write-Host "  Installed: $($_.Name)"
    $installed++
}

Write-Host ""
Write-Host "$installed hook(s) installed into .git\hooks\"
Write-Host "Test with: git add . ; git commit -m 'test' (on a branch)"
