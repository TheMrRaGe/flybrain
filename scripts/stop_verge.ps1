# Stops the fly swarm (and the Verge server if -All).
param([switch]$All)
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "verge_swarm\.py" } | ForEach-Object { Write-Host "stopping swarm $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }
if ($All) { Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "server\.mjs" } | ForEach-Object { Write-Host "stopping Verge server $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force } }
