# Thin PowerShell wrapper around setup.py for Windows convenience.
# All real logic lives in setup.py (cross-platform).
#
# Note: Windows requires Developer Mode enabled for symlink creation.
# If Developer Mode is off, the script falls back to file copies (use
# `--copy` to force this explicitly).
#
# Usage:
#     pwsh crt-royale-msl/integration/setup.ps1            # try symlinks
#     pwsh crt-royale-msl/integration/setup.ps1 --copy     # force copies
#     pwsh crt-royale-msl/integration/setup.ps1 --undo     # reverse setup

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Prefer `py` launcher on Windows, fall back to `python3` / `python`.
$pythonCmd = $null
foreach ($candidate in @("py", "python3", "python")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $pythonCmd = $candidate
        break
    }
}
if (-not $pythonCmd) {
    Write-Error "Python 3 not found in PATH. Install Python from python.org or the Microsoft Store."
    exit 1
}

& $pythonCmd "$here\setup.py" @args
exit $LASTEXITCODE
