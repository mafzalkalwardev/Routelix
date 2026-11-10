#!/usr/bin/env bash
# PulseAudio or PipeWire for the routelix service user.
# Chromium and the Python process share XDG_RUNTIME_DIR=/run/routelix.
set -euo pipefail
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/routelix}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

load_awake_sink() {
  local name="$1"
  local description="$2"
  local props="device.description=${description} session.suspend-timeout-seconds=0 node.always-process=true node.passive=false node.suspend-on-idle=false"
  if pactl list short sinks 2>/dev/null | awk '{print $2}' | grep -qx "$name"; then
    pactl suspend-sink "$name" 0 || true
    return 0
  fi
  pactl load-module module-null-sink "sink_name=${name}" "sink_properties=${props}"
  pactl suspend-sink "$name" 0 || true
}

if command -v pipewire >/dev/null 2>&1 && command -v pipewire-pulse >/dev/null 2>&1; then
  pipewire &
  pipewire_pid=$!
  sleep 0.3
  if command -v wireplumber >/dev/null 2>&1; then
    wireplumber &
    wireplumber_pid=$!
  else
    wireplumber_pid=""
  fi
  sleep 0.3
  pipewire-pulse &
  pulse_pid=$!
  trap 'kill "$pipewire_pid" ${wireplumber_pid:+"$wireplumber_pid"} "$pulse_pid" 2>/dev/null || true' EXIT
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if pactl info >/dev/null 2>&1; then
      break
    fi
    sleep 0.3
  done
  # Null sinks with suspend timeout 0. Python reloads an older sink that lacks it.
  load_awake_sink routelix_to_gv Routelix_TTS_to_Google_Voice || true
  load_awake_sink routelix_from_gv Routelix_Google_Voice_Speaker || true
  wait "$pulse_pid"
  exit $?
fi

if command -v pulseaudio >/dev/null 2>&1; then
  exec pulseaudio --daemonize=no --exit-idle-time=-1 --log-target=stderr
fi

echo "Neither pipewire-pulse nor pulseaudio is installed." >&2
exit 1
