# _mtp_kernels.ps1 - is the MTP prefill penalty a HIP kernel-path choice?
# The Vulkan comparison would tell us backend-level; without that binary, this asks the nearest
# question locally: with MTP on, does forcing a different HIP kernel path restore prefill?
$ErrorActionPreference = 'SilentlyContinue'
$rocm  = 'D:\strata\dist\strata-windows-x64-hip\rocm\bin'
$bin   = 'D:\llamabuild\build\bin'
$model = 'D:\strata\model\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
$port  = 8093
$env:PATH = "$rocm;$bin;" + $env:PATH

$para = "Retrieval augmented generation combines a document store with a language model. The retriever returns the passages most relevant to a question, and the model answers from them. "
$prompt = ($para * 380) + "`n`nIn one sentence: what are the key steps of retrieval augmented generation?"
$body = @{ messages=@(@{role='user'; content=$prompt}); max_tokens=8; temperature=0.7; chat_template_kwargs=@{enable_thinking=$false} } | ConvertTo-Json -Depth 6

$runs = @(
    @{ name='MTP + FORCE_MMQ';        env=@{ 'GGML_CUDA_FORCE_MMQ'='1' } },
    @{ name='MTP + no rocWMMA FATTN'; env=@{ 'GGML_HIP_ROCWMMA_FATTN'='0' } }
)

foreach ($run in $runs) {
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 4
    $env:GGML_CUDA_FORCE_MMQ = ''; $env:GGML_HIP_ROCWMMA_FATTN = ''
    foreach ($k in $run.env.Keys) { Set-Item -Path "env:$k" -Value $run.env[$k] }

    Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList @(
        '-m', $model, '--host', '127.0.0.1', '--port', "$port",
        '-c', '49152', '-np', '1', '-ngl', '99', '-ctk', 'q8_0', '-ctv', 'q8_0',
        '-fa', 'on', '-t', '8', '-b', '4096', '-ub', '512', '--spec-type', 'draft-mtp'
    ) -RedirectStandardOutput "$env:TEMP\k.out" -RedirectStandardError "$env:TEMP\k.err" | Out-Null

    $up = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 3
        try { if ((Invoke-WebRequest "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $up = $true; break } } catch { }
    }
    if (-not $up) { "{0}: did not start" -f $run.name; continue }
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 1800
        "{0,-24} prefill {1,7:N1} tok/s   decode {2,6:N2}   ({3} prompt tokens)" -f $run.name, $r.timings.prompt_per_second, $r.timings.predicted_per_second, $r.timings.prompt_n
    } catch { "{0}: request failed - {1}" -f $run.name, $_.Exception.Message }
}
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item Env:GGML_CUDA_FORCE_MMQ, Env:GGML_HIP_ROCWMMA_FATTN -ErrorAction SilentlyContinue
'kernels done'
