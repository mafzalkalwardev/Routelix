"""How many live calls one Routelix server can place.

Google Voice in the browser is one number and one call panel. A second tab,
profile, or Selenium session on that same account does not create a second
live call: the number is already busy, or the UI steals the single call.
Concurrency above one call needs another phone line. This module does not
add that.

The 24 GB figure is a RAM budget from process sizes measured on a headless
Linux host (PipeWire plus one Python interpreter) plus the resident size of
the CPU models the server actually loads. It is not a count of calls placed.
"""
from __future__ import annotations

import os

# voice.google.com exposes one active call per number.
GOOGLE_VOICE_CONCURRENT_CALLS = 1

# Resident sizes observed while bringing up the headless audio path, plus the
# published working set of the CPU models selected in deploy/linux/.env.example
# (faster-whisper base int8, Piper lessac, one Chromium). Rounded up.
# RSS of pipewire + pipewire-pulse + wireplumber on a headless Linux host: ~42 MB.
_MEASURED_PIPEWIRE_AND_HELPERS_MB = 42
_MEASURED_PYTHON_INTERPRETER_MB = 45
_CHROMIUM_RSS_MB = 450
_WHISPER_BASE_INT8_MB = 250
_PIPER_VOICE_MB = 80
_AGENT_WORKING_SET_MB = 120

AGENT_RSS_MB = (
    _MEASURED_PIPEWIRE_AND_HELPERS_MB
    + _MEASURED_PYTHON_INTERPRETER_MB
    + _CHROMIUM_RSS_MB
    + _WHISPER_BASE_INT8_MB
    + _PIPER_VOICE_MB
    + _AGENT_WORKING_SET_MB
)

# Leave this much RAM for the OS, nginx, and the audio daemon.
_RESERVED_RAM_GB = 2.0


def google_voice_concurrent_calls() -> int:
    """Live calls one Google Voice number can carry at the same time."""
    return GOOGLE_VOICE_CONCURRENT_CALLS


def agent_rss_mb() -> int:
    return AGENT_RSS_MB


def theoretical_agents(ram_gb: float | None = None) -> int:
    """How many isolated agents would fit if each had its own phone line.

    The default budget is the 4 OCPU / 24 GB host. The effective live-call
    cap stays at one until another number exists.
    """
    if ram_gb is None:
        raw = os.getenv("ROUTELIX_RAM_GB", "24").strip() or "24"
        try:
            ram_gb = float(raw)
        except ValueError:
            ram_gb = 24.0
    usable_mb = max(0.0, (float(ram_gb) - _RESERVED_RAM_GB) * 1024.0)
    if AGENT_RSS_MB <= 0:
        return 1
    return max(1, int(usable_mb // AGENT_RSS_MB))


def capacity_report(ram_gb: float | None = None) -> str:
    agents = theoretical_agents(ram_gb)
    return (
        f"One Google Voice number supports {GOOGLE_VOICE_CONCURRENT_CALLS} live call. "
        f"Each agent is about {AGENT_RSS_MB} MB "
        f"(Chromium {_CHROMIUM_RSS_MB} MB, faster-whisper base int8 {_WHISPER_BASE_INT8_MB} MB, "
        f"Piper {_PIPER_VOICE_MB} MB, interpreter and helpers "
        f"{_MEASURED_PYTHON_INTERPRETER_MB + _MEASURED_PIPEWIRE_AND_HELPERS_MB} MB, "
        f"conversation working set {_AGENT_WORKING_SET_MB} MB). "
        f"A {ram_gb or os.getenv('ROUTELIX_RAM_GB', '24')} GB server could run about {agents} "
        f"such agents if each had its own phone line. This installation dials one line."
    )
