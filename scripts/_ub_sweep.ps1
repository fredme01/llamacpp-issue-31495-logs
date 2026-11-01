# _ub_sweep.ps1 - prefill vs ubatch on the real llama-server config
$ErrorActionPreference = 'SilentlyContinue'
$rocm = 'D:\strata\dist\strata-windows-x64-hip\rocm\bin'
$bin  = 'D:\llamabuild\build\bin'
$model = 'D:\strata\model\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
$env:PATH = "$rocm;$bin;" + $env:PATH

$para = "Retrieval augmented generation combines a document store with a language model. The retriever returns the passages most relevant to a question, and the model answers from them. "
$prompt = ($para * 380) + "`n`nIn one sentence: what are the key steps of retrieval augmented generation?"
$body = @{ messages=@(@{role='user'; content=$prompt}); max_tokens=8; temperature=0.7; chat_template_kwargs=@{enable_thinking=$false} } | ConvertTo-Json -Depth 6

foreach ($ub in @(128, 2048)) {
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 4
    $port = 8090
    Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList @(
        '-m', $model, '--host', '127.0.0.1', '--port', "$port",
        '-c', '49152', '-np', '1', '-ngl', '99', '-ctk', 'q8_0', '-ctv', 'q8_0',
        '-fa', 'on', '-t', '8', '-b', '4096', '-ub', "$ub"
    ) -RedirectStandardOutput "$env:TEMP\ub-$ub.out" -RedirectStandardError "$env:TEMP\ub-$ub.err" | Out-Null
    $ok = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 3
        try { if ((Invoke-WebRequest "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $ok = $true; break } } catch { }
    }
    if (-not $ok) { "ub=$ub : server did not come up"; continue }
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
        "ub={0,-5} prompt {1} -> pp {2:N1} tok/s, decode {3:N2}" -f $ub, $r.timings.prompt_n, $r.timings.prompt_per_second, $r.timings.predicted_per_second
    } catch { "ub=$ub : request failed - $($_.Exception.Message)" }
}
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
'done'
