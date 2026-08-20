# Creates "D&D Bot" shortcuts on the Desktop and in the Start Menu.
# Run once:  powershell -ExecutionPolicy Bypass -File install_shortcut.ps1

$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$batch      = Join-Path $projectDir 'start_bot.bat'
$iconSource = Join-Path $projectDir '.venv\Scripts\python.exe'

if (-not (Test-Path $batch)) {
    Write-Host "ERROR: start_bot.bat not found next to this script." -ForegroundColor Red
    exit 1
}

# The shortcut targets cmd.exe rather than the .bat directly. Windows will
# only pin a shortcut to the taskbar when its target is an executable, so
# pointing straight at a .bat gives you a shortcut that can be pinned to
# Start but not to the taskbar.
$targetExe  = "$env:SystemRoot\System32\cmd.exe"
$targetArgs = "/c `"$batch`""

$locations = @(
    [PSCustomObject]@{ Name = 'Desktop';    Path = [Environment]::GetFolderPath('Desktop') }
    [PSCustomObject]@{ Name = 'Start Menu'; Path = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs' }
)

$shell = New-Object -ComObject WScript.Shell
foreach ($loc in $locations) {
    if (-not (Test-Path $loc.Path)) { continue }
    $linkPath = Join-Path $loc.Path 'D&D Bot.lnk'

    $sc = $shell.CreateShortcut($linkPath)
    $sc.TargetPath       = $targetExe
    $sc.Arguments        = $targetArgs
    $sc.WorkingDirectory = $projectDir
    $sc.Description      = 'Start the D&D Discord bot (transcription, dice, recaps)'
    if (Test-Path $iconSource) { $sc.IconLocation = "$iconSource,0" }
    $sc.Save()

    Write-Host "Created: $linkPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. To pin it:" -ForegroundColor Cyan
Write-Host "  Taskbar - right-click the Desktop shortcut, Show more options, Pin to taskbar"
Write-Host "  Start   - press Start, find 'D&D Bot' in the app list, right-click, Pin to Start"
Write-Host ""
Write-Host "Close the console window to stop the bot."
