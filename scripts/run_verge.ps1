# Fly tribes in the Verge - one click.
#   Starts the Verge game server if it is not running, (re)starts the fly swarm,
#   and opens the spectator page. Run from anywhere:
#     powershell -ExecutionPolicy Bypass -File scripts\run_verge.ps1            # hive mind (one memory per tribe)
#     powershell -ExecutionPolicy Bypass -File scripts\run_verge.ps1 -Solo      # individual brains, children inherit
#     powershell -ExecutionPolicy Bypass -File scripts\run_verge.ps1 -Fresh     # start the experiment over
#   or right-click -> Run with PowerShell. Stop everything with stop_verge.ps1.
#   The two modes keep separate memories (results\verge_hive, results\verge).
param([switch]$Solo, [switch]$Fresh)
$here = $PSScriptRoot
$game = Join-Path $here "..\..\xaya\prototypes\stage-b"
$hive = -not $Solo
$out  = if ($hive) { Join-Path $here "..\results\verge_hive" } else { Join-Path $here "..\results\verge" }
New-Item -ItemType Directory -Force $out | Out-Null

# 1. the game (real time; SLOW=1). Leave it alone if it is already up.
$server = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "server\.mjs" }
if (-not $server) {
  Write-Host "starting the Verge server ..."
  Start-Process -FilePath "node" -ArgumentList "server.mjs" -WorkingDirectory $game -WindowStyle Minimized
  Start-Sleep 3
} else { Write-Host "Verge server already running (pid $($server.ProcessId))" }

# 2. any old swarm goes
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "verge_swarm\.py" } | ForEach-Object {
  Write-Host "stopping old swarm (pid $($_.ProcessId))"; Stop-Process -Id $_.ProcessId -Force
}
Start-Sleep 1

# 3. the flies. Edit the numbers here: tribes, flies per tribe.
$env:PYTHONIOENCODING = "utf-8"
$outArg = if ($hive) { "../results/verge_hive --hive tribe" } else { "../results/verge" }
$args = "verge_swarm.py --tribes 2 --per-tribe 4 --inherit-tribes all --capacity 16 --max-ticks 0 " +
        "--out-dir $outArg --state-copy ../../xaya/prototypes/stage-b/flystate.json" + $(if ($Fresh) { " --fresh" } else { "" })
Write-Host ("mode: " + $(if ($hive) { "HIVE MIND - one shared memory per tribe" } else { "individual brains, children inherit" }))
Start-Process -FilePath "python3" -ArgumentList $args -WorkingDirectory $here `
  -RedirectStandardOutput (Join-Path $out "run.log") -RedirectStandardError (Join-Path $out "run.err") -WindowStyle Minimized
Write-Host "swarm starting (brain load takes ~30 s); log: $out\run.log, errors: run.err, events: events.jsonl"

# 3b. Telegram updates, if <out>\telegram.json exists (see verge_telegram.py for the setup)
if (Test-Path (Join-Path $out "telegram.json")) {
  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "verge_telegram\.py" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  Start-Process -FilePath "python3" -ArgumentList "verge_telegram.py --every 30 --out $out" -WorkingDirectory $here -WindowStyle Minimized
  Write-Host "telegram updates on (every 30 min + instant alerts)"
}

# 4. the page
Start-Sleep 2
Start-Process "http://localhost:8000/flies.html"
