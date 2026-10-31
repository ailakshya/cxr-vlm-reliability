#!/usr/bin/env bash
# Free the GPU for a batch by pausing the CSMS staging voice services and the camera grid, then put them back.
#   gpu_pause.sh pause   stop the health timer, voice agent, its Ollama server and frigate-grid (camera ffmpeg)
# frigate-grid is a system unit: stopping it needs /etc/sudoers.d/cxr-cameras (NOPASSWD for systemctl stop/start
# frigate-grid.service); without that rule the cameras are left running.
#   gpu_pause.sh resume  start them again through systemd
#   gpu_pause.sh run CMD...   pause, run CMD, always resume afterwards (also on error or Ctrl-C)
# A copy of the voice agent started by the CI runner outside systemd (port 8765) is stopped on pause;
# on resume the systemd unit csms-voice.service takes over that port, which is its normal setup.
set -u
UNITS="csms-voice.service csms-voice-llm.service"
STATE=/mnt/hdd/cxr_vlm/logs/gpu_pause.state

pause() {
  echo "$(date -Is) pause" >> "$STATE"
  # the health check restarts the agent if it fires while we stop it (seen 2026-10-31): stop it too, then retry
  systemctl --user stop csms-voice-health.timer csms-voice-health.service
  for i in 1 2 3 4 5; do
    systemctl --user stop $UNITS
    systemctl --user is-active --quiet $UNITS || break
    sleep 5
  done
  pkill -f "/mnt/hdd/csms-agents/voice/code/.venv/bin/python -u service.py" 2>/dev/null
  pkill -f "^.venv/bin/python -u service.py" 2>/dev/null
  for i in $(seq 30); do
    pgrep -f "csms-agents/voice" >/dev/null || pgrep -f "^.venv/bin/python -u service.py" >/dev/null || break; sleep 1
  done
  sudo -n /usr/bin/systemctl stop frigate-grid.service 2>/dev/null && echo "cameras stopped"   # user asked 2026-10-31
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

resume() {
  systemctl --user start $UNITS
  systemctl --user start csms-voice-health.timer
  sudo -n /usr/bin/systemctl start frigate-grid.service 2>/dev/null
  echo "$(date -Is) resume" >> "$STATE"
  sleep 15; systemctl --user is-active $UNITS csms-voice-health.timer | tr '\n' ' '; echo
}

case "${1:-}" in
  pause) pause ;;
  resume) resume ;;
  run) shift; pause; trap resume EXIT INT TERM; "$@" ;;
  *) echo "usage: $0 pause|resume|run CMD..." >&2; exit 2 ;;
esac
