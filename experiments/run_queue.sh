#!/usr/bin/env bash
# Work through jobs.txt one line at a time (lines can be appended while it runs).
# Optional first argument: a regex of models this queue handles, so two queues can share the GPU, e.g.
#   ./gpu_pause.sh run bash -c './run_queue.sh chexagent2 & ./run_queue.sh "qwen25vl7b|lingshu7b" & wait' 
# Line format: <model> <site> <cond> <n> <k> <kloc>     (n=0 means all images; '#' starts a comment)
# Finished lines are recorded in jobs_done.txt; failed ones in jobs_failed.txt. Start it inside gpu_pause.sh:
#   nohup ./gpu_pause.sh run ./run_queue.sh > /mnt/hdd/cxr_vlm/logs/queue.log 2>&1 &
cd "$(dirname "$0")"
export HF_HOME=/mnt/hdd/cxr_vlm/hf TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ONLY="${1:-[a-z0-9_]+}"
touch jobs_done.txt jobs_failed.txt
while true; do
  job=$(grep -v '^\s*#' jobs.txt | grep -v '^\s*$' | grep -vxFf jobs_done.txt | grep -vxFf jobs_failed.txt | grep -E "^($ONLY) " | head -1)
  [ -z "$job" ] && { echo "$(date -Is) queue empty"; break; }
  read -r model site cond n k kloc <<< "$job"
  py=venv/bin/python; [ "$model" = chexagent2 ] && py=venv_chexagent/bin/python
  [ -f "/mnt/hdd/cxr_vlm/prepared/$site.jsonl" ] || venv/bin/python prepare.py "$site" >> /mnt/hdd/cxr_vlm/logs/prepare.log 2>&1
  if [ ! -f "/mnt/hdd/cxr_vlm/prepared/$site.jsonl" ]; then echo "skip (no manifest): $job"; echo "$job" >> jobs_failed.txt; continue; fi
  # wait for a free GPU instead of failing every job in a row (e.g. the voice agent came back after a reboot)
  w=0
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 12000 ]; do
    [ $w = 0 ] && echo "$(date -Is) waiting for 12 GB of free GPU memory"; w=$((w + 1)); sleep 30
    [ $((w % 4)) = 0 ] && ./gpu_pause.sh pause >/dev/null 2>&1   # every 2 min: pause the voice agent again
  done
  echo "$(date -Is) START $job"
  if $py generate.py --model "$model" --site "$site" --cond "$cond" --n "$n" --k "$k" --kloc "$kloc" \
       >> "/mnt/hdd/cxr_vlm/logs/gen_${model}_${site}_${cond/:/_}.log" 2>&1; then
    echo "$job" >> jobs_done.txt; sync jobs_done.txt; echo "$(date -Is) DONE $job"
  else
    echo "$job" >> jobs_failed.txt; echo "$(date -Is) FAILED $job"
  fi
done
