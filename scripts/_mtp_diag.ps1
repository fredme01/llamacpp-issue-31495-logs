# _mtp_diag.ps1 - why does draft-mtp cost ~4x prefill on this HIP build?
#
# Three configs, same model / prompt / -ngl / -fa / -b / -ub; only the MTP switch (and the HIP
# graph switch on AMD, which the CUDA-named env var still drives) changes.
# While each prefill runs we sample BOTH GPU memory counters:
#   Dedicated = VRAM,  Shared = system memory the GPU maps (the Windows analogue of GTT).
# A silent GTT/VRAM overflow shows up as Shared spiking during the request.
$ErrorActionPreference = 'SilentlyContinue'
$rocm  = 'D:\strata\dist\strata-windows-x64-hip\rocm\bin'
$bin   = 'D:\llamabuild\build\bin'
$model = 'D:\strata\model\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
$port  = 8092
$env:PATH = "$rocm;$bin;" + $env:PATH

$para = "Retrieval augmented generation combines a document store with a language model. The retriever returns the passages most relevant to a question, and the model answers from them. "
$prompt = ($para * 380) + "`n`nIn one sentence: what are the key steps of retrieval augmented generation?"
$body = @{ messages=@(@{role='user'; content=$prompt}); max_tokens=8; temperature=0.7; chat_template_kwargs=@{enable_thinking=$false} } | ConvertTo-Json -Depth 6

$configs = @(
    @{ name='A no-MTP';            spec=$null;            nograph=$false },
    @{ name='B MTP';               spec='draft-mtp';      nograph=$false },
    @{ name='C MTP + no HIP graph'; spec='draft-mtp';     nograph=$true  }
)

foreach ($cfg in $configs) {
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 4

    $env:GGML_CUDA_DISABLE_GRAPHS = if ($cfg.nograph) { '1' } else { '' }
    $a = @('-m', $model, '--host', '127.0.0.1', '--port', "$port",
           '-c', '49152', '-np', '1', '-ngl', '99', '-ctk', 'q8_0', '-ctv', 'q8_0',
           '-fa', 'on', '-t', '8', '-b', '4096', '-ub', '512', '--verbose')
    if ($cfg.spec) { $a += @('--spec-type', $cfg.spec) }
    $tag = $cfg.name -replace '[^A-Za-z0-9]', '_'
    Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $a `
        -RedirectStandardOutput "$env:TEMP\diag-$tag.out" -RedirectStandardError "$env:TEMP\diag-$tag.err" | Out-Null

    $up = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 3
        try { if ((Invoke-WebRequest "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $up = $true; break } } catch { }
    }
    if (-not $up) { "{0}: server did not start" -f $cfg.name; continue }

    $base_d = ((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | Where-Object { $_.CookedValue -gt 100MB } | Measure-Object CookedValue -Maximum).Maximum / 1GB
    $base_s = ((Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1GB

    $job = Start-Job -ScriptBlock {
        param($b, $p)
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod "http://127.0.0.1:$p/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 1800
        [pscustomobject]@{ wall=$sw.Elapsed.TotalSeconds; pp=$r.timings.prompt_per_second; pn=$r.timings.prompt_n; tg=$r.timings.predicted_per_second }
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

    "{0}" -f $cfg.name
    "   prefill {0:N1} tok/s over {1} tokens, decode {2:N2}, wall {3:N1}s" -f $r.pp, $r.pn, $r.tg, $r.wall
    "   VRAM dedicated  idle {0:N2} -> max {1:N2} GB   (peak +{2:N2})" -f $base_d, ($d | Measure-Object -Maximum).Maximum, (($d | Measure-Object -Maximum).Maximum - $base_d)
    "   GPU shared mem  idle {0:N2} -> max {1:N2} GB   (peak +{2:N2})" -f $base_s, ($s | Measure-Object -Maximum).Maximum, (($s | Measure-Object -Maximum).Maximum - $base_s)
    "   gpu compute     mean {0:N0}%  max {1:N0}%" -f ($g | Measure-Object -Average).Average, ($g | Measure-Object -Maximum).Maximum
    ""
}
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
'diag done'
