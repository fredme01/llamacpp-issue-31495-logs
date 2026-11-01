# Measurements

Same prompt everywhere: 12,188 API tokens / 12,184 prompt tokens as counted by the server.
Same flags: `-c 49152 -np 1 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on -t 8` (`-b 4096 -ub 512` for the
diag/rebuild/Vulkan runs).

## Headline: the MTP switch, per backend and build

| # | Backend | Build | MTP | prompt time | server tok/s | API pp | API decode |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | HIP | b843e7dc (2026-10-15) | off | 23.25 s | **523.93** | 498.8 | 16.16 |
| 2 | HIP | b843e7dc | **on** | 89.59 s | **135.99** | 130.2 | 21.02 |
| 3 | HIP | b843e7dc | on, graphs off | 93.17 s | 130.78 | 125.0 | 19.98 |
| 4 | Vulkan | b11960 (official) | off | 49.60 s | 245.63 | 223.4 | 3.81 |
| 5 | Vulkan | b11960 (official) | **on** | 51.37 s | **237.19** | 225.8 | 2.25 |
| 6 | HIP | b11960 rebuilt here | off | 22.80 s | **534.46** | 508.7 | 18.28 |
| 7 | HIP | b11960 rebuilt here | **on** | 115.67 s | **105.34** | 100.7 | 14.53 |

Verbatim `slot print_timing` lines, one per row:

```
raw/hip-b843e7dc-no-mtp.err
  0.35.050.637 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =  12184, progress = 1.00, t =  23.25 s / 523.93 tokens per second
raw/hip-b843e7dc-mtp.err
  1.44.357.145 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =  12184, progress = 1.00, t =  89.59 s / 135.99 tokens per second
raw/hip-b843e7dc-mtp-no-hip-graph.err
  1.48.265.792 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =  12184, progress = 1.00, t =  93.17 s / 130.78 tokens per second
raw/vulkan-b11960-no-mtp.err
  1.01.878.316 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =  12184, progress = 1.00, t =  49.60 s / 245.63 tokens per second
raw/vulkan-b11960-mtp.err
  1.02.752.617 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =  12184, progress = 1.00, t =  51.37 s / 237.19 tokens per second
raw/hip-b11960-rebuilt-no-mtp.err
  0.32.094.960 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =  12184, progress = 1.00, t =  22.80 s / 534.46 tokens per second
raw/hip-b11960-rebuilt-mtp.err
  2.09.691.909 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =  12184, progress = 1.00, t = 115.67 s / 105.34 tokens per second
```

## GPU and memory during prompt processing

Sampled every 500 ms for the whole request (Windows GPU counters):
`\GPU Adapter Memory(*)\Dedicated Usage` (VRAM) and `\GPU Adapter Memory(*)\Shared Usage`
(the Windows analogue of GTT — the system memory the GPU maps), plus
`\GPU Engine(*)\Utilization Percentage` for the `compute` engines.

| Run | VRAM during prompt | GPU Shared during prompt | GPU compute |
|---|---|---|---|
| HIP b843e7dc, MTP off | 10.66 → 10.96 GB (+0.30) | 1.93 → 1.94 GB (+0.00) | mean 86%, peak 113% |
| HIP b843e7dc, MTP on | 11.47 → 11.72 GB (+0.25) | 2.33 → 3.07 GB (+0.74) | mean 90%, peak 155% |
| HIP b843e7dc, MTP on, graphs off | 11.47 → 11.72 GB (+0.25) | 2.33 → 2.99 GB (+0.66) | mean 93%, peak 112% |

So there is no VRAM overflow and no GTT overflow, and the GPU is busier during the slow prompt
processing, not idle: the engine does roughly 4x more work for the same tokens.

## Variables ruled out on the HIP build

| Variable | prompt throughput | log |
|---|---:|---|
| `-ub 128` (MTP off) | 443.5 tok/s (API) | `raw/hip-b843e7dc-ub128.err` |
| `-ub 2048` (MTP off) | 467.6 tok/s (API) | `raw/hip-b843e7dc-ub2048.err` |
| `--spec-draft-n-max 1` (MTP on) | 114.8 tok/s (API) | (same prompt, 3 runs: 115 / 118 / 114) |
| `-c 16384` instead of 49152 (MTP on) | 114.0 tok/s (API) | (context size is not the driver) |
| `GGML_CUDA_FORCE_MMQ=1` (MTP on) | 124.5 tok/s (API) | (not the GEMM path) |
| `GGML_HIP_ROCWMMA_FATTN=0` (MTP on) | 126.2 tok/s (API) | (not rocWMMA flash attention) |
| `GGML_CUDA_DISABLE_GRAPHS=1` (MTP on) | 130.78 server / 125.0 API | `raw/hip-b843e7dc-mtp-no-hip-graph.err` |

## Speculation itself works

From the MTP-on runs (`timings` in the API response):

- HIP b843e7dc: `draft_n 117, draft_n_accepted 56` (~48%), decode 16.16 → 21.02 tok/s
- so the head is being used correctly; only prompt processing collapses

Caveat on the decode column: the headline runs generated only 8 tokens (`max_tokens 8`), so those
decode figures are indicative. A separate pair of runs on the same prompt with `max_tokens 128`
gave decode 16.09 (MTP off) → 20.85 (MTP on), i.e. the same ~30% gain, and `draft_n 36 / accepted 18`.

## Build provenance

- b11960 was rebuilt on this machine from tag `b11960` (commit `8642541`) with the same toolchain as
  the working build: clang from ROCm (`D:\hipbuild\rocm`), `-DGGML_HIP=ON -DGPU_TARGETS=gfx1101
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON`,
  producing a 94 MB `ggml-hip.dll` that contains only `gfx1101` — identical in shape to the build
  that was already in use (92 MB, `gfx1101` only).
- The official `llama-b11960-bin-win-rocm-10.0-x64.zip` could not be used for the comparison on this
  machine: its bundled ROCm 10.0 runtime reports `Available devices: (none)`, and with the local
  ROCm 10.2 runtime it loads the weights into VRAM but computes on the CPU
  (`raw/hip-b11960-official-cpu-fallback.err`, GPU compute mean 0%, 490 s for the same prompt).
