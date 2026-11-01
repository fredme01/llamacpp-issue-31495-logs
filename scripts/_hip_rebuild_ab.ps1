# _vulkan_ab.ps1 - same A/B as the HIP run, but on the Vulkan backend (b11960).
$ErrorActionPreference = 'SilentlyContinue'
$bin   = 'D:\llamabuild-b11960\build\bin'
$model = 'D:\strata\model\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
$port  = 8096
$env:PATH = "D:\strata\dist\strata-windows-x64-hip\rocm\bin;$bin;" + $env:PATH

$para = "Retrieval augmented generation combines a document store with a language model. The retriever returns the passages most relevant to a question, and the model answers from them. "
$prompt = ($para * 380) + "`n`nIn one sentence: what are the key steps of retrieval augmented generation?"
$body = @{ messages=@(@{role='user'; content=$prompt}); max_tokens=8; temperature=0.7; chat_template_kwargs=@{enable_thinking=$false} } | ConvertTo-Json -Depth 6

$configs = @(
    @{ name='HIP b11960 rebuilt no-MTP'; spec=$null },
    @{ name='HIP b11960 rebuilt MTP';    spec='draft-mtp' }
)

foreach ($cfg in $configs) {
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 4
    $a = @('-m', $model, '--host', '127.0.0.1', '--port', "$port",
           '-c', '49152', '-np', '1', '-ngl', '99', '-ctk', 'q8_0', '-ctv', 'q8_0',
           '-fa', 'on', '-t', '8', '-b', '4096', '-ub', '512')
    if ($cfg.spec) { $a += @('--spec-type', $cfg.spec) }
    $tag = $cfg.name -replace '[^A-Za-z0-9]', '_'
    Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $a `
        -RedirectStandardOutput "$env:TEMP\vk-$tag.out" -RedirectStandardError "$env:TEMP\vk-$tag.err" | Out-Null

    $up = $false
    for ($i = 0; $i -lt 50; $i++) {
        Start-Sleep -Seconds 3
        try { if ((Invoke-WebRequest "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $up = $true; break } } catch { }
    }
    if (-not $up) {
        "{0}: server did not start" -f $cfg.name
        Get-Content "$env:TEMP\vk-$tag.err" -Tail 12 -ErrorAction SilentlyContinue
        continue
    }

    $job = Start-Job -ScriptBlock {
        param($b, $p)
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod "http://127.0.0.1:$p/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 1800
        [pscustomobject]@{ wall=$sw.Elapsed.TotalSeconds; pp=$r.timings.prompt_per_second; pn=$r.timings.prompt_n; tg=$r.timings.predicted_per_second; drafts=$r.timings.draft_n; acc=$r.timings.draft_n_accepted }
    } -ArgumentList $body, $port

    Start-Sleep -Seconds 2
    $d=@(); $s=@(); $g=@()
    while ((Get-Job -Id $job.Id).State -eq 'Running') {
        $d += ((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | Where-Object { $_.CookedValue -gt 100MB } | Measure-Object CookedValue -Maximum).Maximum / 1GB
        $s += ((Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1GB
        $g += (((Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples | Where-Object { $_.InstanceName -match 'engtype_compute' }) | Measure-Object CookedValue -Sum).Sum
        Start-Sleep -Milliseconds 500
    }
    $r = Receive-Job $job -Wait -AutoRemoveJob
    if (-not $r) { "{0}: request failed" -f $cfg.name; continue }

    "{0}" -f $cfg.name
    "   prefill {0,7:N1} tok/s over {1} tokens, decode {2,6:N2}, wall {3:N1}s" -f $r.pp, $r.pn, $r.tg, $r.wall
    if ($cfg.spec) { "   drafts {0} accepted {1}" -f $r.drafts, $r.acc }
    "   VRAM peak {0,5:N2} GB   shared peak {1,5:N2} GB   gpu compute mean {2:N0}% max {3:N0}%" -f `
        (($d | Measure-Object -Maximum).Maximum), (($s | Measure-Object -Maximum).Maximum), `
        (($g | Measure-Object -Average).Average), (($g | Measure-Object -Maximum).Maximum)
    ""
}
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
'hip b11960 rebuilt ab done'



