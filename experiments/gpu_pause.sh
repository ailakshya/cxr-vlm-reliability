#!/usr/bin/env bash
# Free the GPU for a batch by pausing the CSMS staging voice services, then put them back.
#   gpu_pause.sh pause   stop the health timer, voice agent and its Ollama server (cameras are not touched)
#   gpu_pause.sh resume  start them again through systemd
#   gpu_pause.sh run CMD...   pause, run CMD, always resume afterwards (also on error or Ctrl-C)
# A copy of the voice agent started by the CI runner outside systemd (port 8765) is stopped on pause;
# on resume the systemd unit csms-voice.service takes over that port, which is its normal setup.
set -u
UNITS="csms-voice.service csms-voice-llm.service"
STATE=/mnt/hdd/cxr_vlm/logs/gpu_pause.state

pause() {
  echo "$(date -Is) pause" >> "$STATE"
  systemctl --user stop csms-voice-health.timer
  systemctl --user stop $UNITS
  pkill -f "/mnt/hdd/csms-agents/voice/code/.venv/bin/python -u service.py" 2>/dev/null
  pkill -f "^.venv/bin/python -u service.py" 2>/dev/null
  for i in $(seq 30); do
    pgrep -f "csms-agents/voice" >/dev/null || pgrep -f "^.venv/bin/python -u service.py" >/dev/null || break; sleep 1
  done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

resume() {
  systemctl --user start $UNITS
  systemctl --user start csms-voice-health.timer
  echo "$(date -Is) resume" >> "$STATE"
  sleep 15; systemctl --user is-active $UNITS csms-voice-health.timer | tr '\n' ' '; echo
}

case "${1:-}" in
  pause) pause ;;
  resume) resume ;;
  run) shift; pause; trap resume EXIT INT TERM; "$@" ;;
  *) echo "usage: $0 pause|resume|run CMD..." >&2; exit 2 ;;
esac
