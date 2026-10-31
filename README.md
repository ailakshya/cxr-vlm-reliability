# Evidence-Grounded Reliability Evaluation of CXR Vision-Language Models under Distribution Shift

```
paper/        Springer Nature (sn-jnl) LaTeX manuscript. Build: cd paper && latexmk -pdf main.tex
              Red "TBD" marks every number not measured yet.
literature/   literature_review.md and the raw research notes behind it.
experiments/  the pipeline (runs on the AI PC, ssh gpu-direct, code in ~/cxr_vlm, data in /mnt/hdd/cxr_vlm).
RUN_LOG.md    one row per queue job: start, duration, images, seconds per image, status (inference only, no training).
logs/ai_pc/   every log from the AI PC: queue, one generate log per job, downloads, evaluation, job lists.
results/      per-run scores from evaluate.py (one JSON per model x site x condition) and summary.csv.
runs/         raw model outputs, one JSON line per image: report, samples, claims, P(yes), boxes, timing.
```

`./sync_results.sh` refreshes all of the above from the AI PC, rebuilds the paper, and commits and pushes.
Images and datasets are not in the repo (licences); `experiments/download_open.py` fetches them.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `download_open.py` | Downloads the open datasets: ChestX-Det, Shenzhen + lesion outlines, Montgomery, IU-Xray, Mendeley India, NITRD New Delhi |
| 2 | `prepare.py` | One manifest per site (`prepared/<site>.jsonl`): resized image, labels, expert boxes mapped to the shared finding space, verifiable findings |
| 3 | `generate.py` | One model × one site × one image condition: greedy report (token log-probs), K sampled reports, CheXbert claims, yes/no probes, boxes for claimed and evidenced findings |
| 4 | `evaluate.py` | ESP / unsupported-claim rate / evidence recall, grounding, calibration of five confidence signals (incl. the self-grounding score), conformal transfer and site-adaptive recalibration, evidence-ablation index, label-site probes, artefact awareness, CheXbert F1 vs. reference |
| 5 | `make_tables.py` | Writes the paper's result tables (`tab_*.tex`) and figures (`fig_*.pdf`) from the scores |
| – | `../sync_results.sh` | One command on the Mac: score on the AI PC, copy tables/figures into `paper/generated/`, rebuild the PDF |
| – | `models.py` | Wrappers: CheXagent-2-3B (bf16, own venv with transformers 4.40), Qwen2.5-VL-7B and Lingshu-7B (4-bit) |
| – | `claims.py`, `common.py` | CheXbert sentence labelling; label mappings, IoU, corruptions, evidence ablation (`evmask` / matched `randmask` control) — each file has a self-check |
| – | `run_queue.sh`, `jobs.txt` | Runs every line of `jobs.txt` in order; lines can be appended while it runs |
| – | `gpu_pause.sh` | Pauses the CSMS staging voice services (they hold ~13 GB of GPU memory) for the length of a batch and restarts them after, even on error. Camera ffmpeg processes are never touched |

## On the AI PC

```bash
cd ~/cxr_vlm
nohup ./gpu_pause.sh run ./run_queue.sh > /mnt/hdd/cxr_vlm/logs/queue.log 2>&1 &   # start (already running)
tail -f /mnt/hdd/cxr_vlm/logs/queue.log                                                # progress
cat jobs_done.txt jobs_failed.txt                                                        # finished / failed jobs
venv/bin/python evaluate.py                                                              # results/summary.csv
```

To stop: kill `run_queue.sh` and the `generate.py` child; the voice services come back automatically.
Re-starting continues where it stopped (each run file is appended one image at a time).

Gated models (MedGemma 1.5, MAIRA-2) need the terms accepted on Hugging Face and `venv/bin/hf auth login` on the AI PC;
then add lines such as `medgemma15 chestxdet clean 0 5 3` to `jobs.txt` (wrapper still to be written).
