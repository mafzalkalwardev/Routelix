#!/usr/bin/env bash
# PulseAudio or PipeWire for the routelix service user.
# Chromium and the Python process share XDG_RUNTIME_DIR=/run/routelix.
set -euo pipefail
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/routelix}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

if command -v pipewire >/dev/null 2>&1 && command -v pipewire-pulse >/dev/null 2>&1; then
  pipewire &
  pipewire_pid=$!
  sleep 0.4
  pipewire-pulse &
  pulse_pid=$!
  trap 'kill "$pipewire_pid" "$pulse_pid" 2>/dev/null || true' EXIT
  wait "$pulse_pid"
  exit $?
fi

if command -v pulseaudio >/dev/null 2>&1; then
  exec pulseaudio --daemonize=no --exit-idle-time=-1 --log-target=stderr
fi

echo "Neither pipewire-pulse nor pulseaudio is installed." >&2
exit 1
