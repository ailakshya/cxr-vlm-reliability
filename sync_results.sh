#!/usr/bin/env bash
# Score the runs on the AI PC, rebuild the generated tables/figures, copy them into paper/generated, rebuild the PDF,
# pull every log, score file and model output into this repo, and commit + push.   ./sync_results.sh [--no-push]
set -e
cd "$(dirname "$0")"
ssh -n gpu-direct 'cd ~/cxr_vlm && CUDA_VISIBLE_DEVICES= HF_HOME=/mnt/hdd/cxr_vlm/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice venv/bin/python evaluate.py >/mnt/hdd/cxr_vlm/logs/evaluate.log 2>&1 && nice venv/bin/python make_tables.py >/dev/null && python3 make_runlog.py >/dev/null && cp jobs.txt jobs_done.txt jobs_failed.txt /mnt/hdd/cxr_vlm/logs/'
scp -q gpu-direct:/mnt/hdd/cxr_vlm/paper_out/* paper/generated/
scp -q gpu-direct:/mnt/hdd/cxr_vlm/results/summary.csv experiments/results_summary.csv

# logs, per-run scores and raw model outputs (runs/ = one JSON line per image: reports, claims, P(yes), boxes, timing)
rsync -a --delete --exclude '*.py' --exclude 'gpu_pause.state' gpu-direct:/mnt/hdd/cxr_vlm/logs/ logs/ai_pc/
rsync -a --delete gpu-direct:/mnt/hdd/cxr_vlm/results/ results/
rsync -a --delete gpu-direct:/mnt/hdd/cxr_vlm/runs/ runs/
mv logs/ai_pc/RUN_LOG.md RUN_LOG.md
if grep -rqE 'hf_[A-Za-z0-9]{30,}' logs results runs RUN_LOG.md; then echo "token found in synced files, not committing" >&2; exit 1; fi

cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null && rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out && cd ..
echo "paper/main.pdf rebuilt with current results"

[ "$1" = "--no-push" ] && exit 0
git add -A
git diff --cached --quiet || { git commit -qm "Sync results and logs from the AI PC ($(date '+%Y-%m-%d %H:%M'))" && git push -q && echo "pushed"; }
