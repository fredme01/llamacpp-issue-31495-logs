# Logs for: `--spec-type draft-mtp` costs ~80% of prompt processing on the HIP backend

Bundle of the raw evidence behind the issue report. Everything here was produced on the same
machine, same model file, same server flags, and the **same 12,184-token prompt**; only the
backend/build/MTP switch varies.

## Layout

```
README.md                     this file
measurements.md               every number, with the verbatim log line it came from
commands.txt                  the exact command line of each run
raw/                          llama-server stderr, unmodified (verbose logging was on)
  hip-b843e7dc-no-mtp.err                     HIP, b843e7dc, MTP off
  hip-b843e7dc-mtp.err                        HIP, b843e7dc, MTP on          <-- the regression
  hip-b843e7dc-mtp-no-hip-graph.err           HIP, b843e7dc, MTP on, GGML_CUDA_DISABLE_GRAPHS=1
  hip-b843e7dc-ub128.err / -ub2048.err        ubatch sweep (MTP off) on the same build
  hip-b11960-rebuilt-no-mtp.err               HIP, b11960 rebuilt locally, MTP off
  hip-b11960-rebuilt-mtp.err                  HIP, b11960 rebuilt locally, MTP on   <-- still broken
  vulkan-b11960-no-mtp.err                    Vulkan, b11960 (official release), MTP off
  vulkan-b11960-mtp.err                       Vulkan, b11960 (official release), MTP on  <-- unaffected
  hip-b11960-official-cpu-fallback.err        official HIP release package: no GPU device -> CPU only
  hip-b843e7dc-model-load-nextn-tensors-ignored.log
                                              the MTP block's tensors being loaded and ignored when
                                              speculation is off
scripts/                      the harness that produced each row (PowerShell)
```

## Reading the numbers

Two throughput figures appear for every run and they intentionally differ by ~4–5%:

- **`server` column** — llama.cpp's own line in `raw/*.err`:
  `slot print_timing: ... prompt processing, n_tokens = 12184, ... t = 23.25 s / 523.93 tokens per second`
  This is prompt processing only.
- **`API` column** — `timings.prompt_per_second` from `POST /v1/chat/completions`. It includes the
  chat-template/prefix work and the first generated token, so it reads slightly lower. Both move
  together; the conclusion does not depend on which one you take.

## Machine

| | |
|---|---|
| GPU | AMD Radeon RX 7700 XT, 12,272 MiB, gfx1101, driver 32.0.32015.7029 |
| CPU / RAM | Intel i5-13490F (10C/16T) / 32 GB |
| OS | Windows 11 build 26100 |
| Backends | HIP (ROCm) and Vulkan |
| Server flags | `-c 49152 -np 1 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on -t 8` (`-b 4096 -ub 512` where noted) |

## Model

Dense hybrid GGUF with one MTP/NextN block, 27.3 B parameters:

- `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, 10,442,827,776 bytes
- GGUF v3, 866 tensors, architecture `qwen35`, 65 blocks (64 + 1 MTP), 5120 embd, 24/4 heads,
  context 262144, `qwen35.nextn_predict_layers = 1`, `general.name = "Swift15-GSQfix-V1MIX-ISTAalloc-IQ3_XXS MTP"`
- the same model without the MTP block (851 tensors, 64 blocks, 10,094,357,792 bytes) was also tested:
  identical throughput within noise, so the MTP tensors themselves cost nothing when speculation is off

Note: the MTP file was removed from this machine after the runs; the non-MTP variant is the one now
installed. The measurements above were taken while it was present, and the log paths below refer to
that file.

## What the logs show

1. `raw/hip-b843e7dc-model-load-nextn-tensors-ignored.log` — with speculation off, the MTP block is
   loaded and skipped: `model has unused tensor blk.64.nextn.eh_proj.weight ... -- ignoring`.
2. `raw/hip-b843e7dc-*.err` — switching `--spec-type draft-mtp` on the same build takes prompt
   processing from 23.25 s to 89.59 s (523.93 → 135.99 tok/s) for the same 12,184 tokens.
3. `raw/hip-b11960-rebuilt-*.err` — rebuilding b11960 from tag `8642541` on this machine with the
   same toolchain and gfx1101-only target reproduces it: 22.80 s → 115.67 s (534.46 → 105.34 tok/s).
4. `raw/vulkan-b11960-*.err` — the same b11960 source on Vulkan: 49.60 s → 51.37 s
   (245.63 → 237.19 tok/s). No regression, so it is not the MTP feature itself.
5. `raw/hip-b11960-official-cpu-fallback.err` — the official `bin-win-rocm-10.0-x64` package reports
   no AMD device on this machine and runs on the CPU; that is why the version comparison had to be
   done with a locally rebuilt binary instead.
