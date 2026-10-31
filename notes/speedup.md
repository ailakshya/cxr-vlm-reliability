# Speed-up of the Qwen2.5-VL and Lingshu runs (31 Oct – 1 Nov 2026)

Qwen2.5-VL-7B and Lingshu-7B now run about **15 times faster**: 2.4 s per image instead of 35.5 s, with the full
protocol (greedy report, K = 5 sampled reports, 12 yes/no questions, K' = 3 box requests per finding). All
timings below were measured on the AI PC (RTX 5070 Ti, 16 GB) on the same 20 ChestX-Det images.

| Setup | s / image | Speed-up |
|---|---|---|
| 4-bit NF4 with Hugging Face Transformers (original) | 35.5 | 1x |
| vLLM 0.32, FP8 weights, one image at a time | 7.0 | 5.1x |
| vLLM, 8 images per batch, CheXbert labelling on 6 CPU threads | 2.4 | 14.8x |

CheXagent-2 is unchanged (about 17 s per image): it needs Transformers 4.40 and its own remote code, so it cannot
be served by vLLM. It is now the slowest part of the study.

## Effect on the schedule

Remaining work dropped from about 74 GPU hours to about 16 (CheXagent-2 10.8 h, Qwen + Lingshu 5.5 h), including
re-running every Qwen and Lingshu job with the new setup.

## What changed and why

The old backend generated text slowly (bitsandbytes 4-bit kernels) and handled one image at a time, so the GPU sat
mostly idle. vLLM's prebuilt 0.32 wheel already supports the RTX 50 series (sm_120), so no source build was needed.
It lives in its own environment, `~/cxr_vlm/venv_vllm`.

Per image, CheXbert labelling of the six reports took 2.7 s of the 7.0 s, on the CPU while the GPU waited. The
pipeline now sends 8 images per vLLM call: their reports, samples and yes/no questions in one batch and their box
requests in a second. Meanwhile all reports are labelled on 6 CPU threads (3 torch threads each, leaving cores
for the vLLM engine).

## Checks

| Check | Result |
|---|---|
| Labeller, Transformers 5 (vLLM venv) vs 4.57 | identical labels on 40 reports (a small compatibility patch in `claims.py`) |
| Batch invariance (`VLLM_BATCH_INVARIANT=1`), 1 vs 8 images per batch | greedy reports 20/20, yes/no probabilities and box answers 51/51 identical; sampled reports 17/20 identical (fixed seed per image); no speed cost |
| FP8 vs 4-bit outputs | different quantisation, so outputs differ: same claim set on 11/20 images, P(yes) differs by 0.046 on average. All Qwen and Lingshu runs are therefore redone with FP8; the 4-bit runs are archived in `/mnt/hdd/cxr_vlm/runs_nf4/` |
| Chinese tokens in reports | about 4–5% of sampled reports in both backends (a known Qwen habit), so not caused by FP8 |

## Bug found on the way

Lingshu's checkpoint ships `top_k=1` in its generation config, and Transformers applied it. Lingshu's "random"
samples were therefore nearly greedy (2.3 distinct out of 5 per image, against 5 of 5 for Qwen), which made its
sample-agreement and self-grounding scores unreliable. vLLM now runs with `generation_config="vllm"`, so only the
sampling settings in our code apply, and the 4-bit wrapper sets `top_k=0`.

## Tried and dropped

| Attempt | Result |
|---|---|
| Batching the 4-bit yes/no questions | about 4% faster; P(yes) changed by at most 0.031 (kept for the old runs only) |
| Batching the 4-bit box requests | 1% faster and changed 30 of 50 greedy box answers through padding; reverted |
| gpu-unlock (tensor-core unlock) | does not support RTX 50 (Blackwell), and our runs are memory-bound 4/8-bit decoding anyway |
| CheXagent-2 load cast on the CPU | 17.7 to 17.0 s per image |
