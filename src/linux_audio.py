"""PulseAudio / PipeWire routing for headless Google Voice on Linux.

TTS is played into the ``routelix_to_gv`` null sink. Chromium's microphone is
that sink's monitor, so the call hears the agent. Chromium's speaker is the
``routelix_from_gv`` null sink, and speech-to-text records that sink's monitor.

PipeWire accepts the same ``pactl`` commands through ``pipewire-pulse``.
Windows VB-CABLE setup stays in ``src.audio_setup``.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GoogleVoiceAgent")

SINK_TO_GV = "routelix_to_gv"
SINK_FROM_GV = "routelix_from_gv"
TTS_MONITOR = f"{SINK_TO_GV}.monitor"
CAPTURE_MONITOR = f"{SINK_FROM_GV}.monitor"
_LISTEN_TAG = "routelix_listen_in"


def pactl_available() -> bool:
    return shutil.which("pactl") is not None


def _pactl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["pactl", *args],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def list_sinks() -> list[str]:
    if not pactl_available():
        return []
    proc = _pactl(["list", "short", "sinks"])
    names: list[str] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            names.append(parts[1].strip())
    return names


def list_modules() -> list[tuple[str, str]]:
    """Return (module_id, module_name_and_args) rows from pactl."""
    if not pactl_available():
        return []
    proc = _pactl(["list", "short", "modules"])
    rows: list[tuple[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0].strip(), "\t".join(parts[1:])))
    return rows


def ensure_null_sink(name: str, description: str) -> bool:
    if name in list_sinks():
        return True
    proc = _pactl(
        [
            "load-module",
            "module-null-sink",
            f"sink_name={name}",
            f"sink_properties=device.description={description}",
        ]
    )
    if proc.returncode != 0:
        logger.warning(
            "Could not load null sink %s: %s",
            name,
            (proc.stderr or proc.stdout or "").strip(),
        )
        return False
    return name in list_sinks() or proc.returncode == 0


def chromium_pulse_env() -> dict[str, str]:
    """Environment for the Chromium process only.

    The browser plays call audio into ``SINK_FROM_GV`` and reads the agent
    voice from ``TTS_MONITOR``. Python TTS/STT use explicit devices instead,
    so they are not pinned to these defaults.
    """
    return {
        "PULSE_SINK": SINK_FROM_GV,
        "PULSE_SOURCE": TTS_MONITOR,
    }


def _listen_module_ids() -> list[str]:
    ids: list[str] = []
    for module_id, rest in list_modules():
        if _LISTEN_TAG in rest:
            ids.append(module_id)
    return ids


def listen_in_active() -> Optional[bool]:
    """True when call audio is looped to the default speakers. None if Pulse is down."""
    if not pactl_available():
        return None
    return bool(_listen_module_ids())


def set_linux_listen_in(enabled: bool) -> bool:
    """Loop the Google Voice monitor to the default sink, or remove that loop.

    A VPS often has no physical speakers. Unloading the loop is the silent
    default. Recordings and STT read the monitor directly and do not depend
    on this switch. Returns False when Pulse is unavailable.
    """
    if not pactl_available():
        return False
    for module_id in _listen_module_ids():
        _pactl(["unload-module", module_id])
    if not enabled:
        return True
    proc = _pactl(
        [
            "load-module",
            "module-loopback",
            f"source={CAPTURE_MONITOR}",
            "sink=@DEFAULT_SINK@",
            "latency_msec=40",
            f"sink_input_properties=media.name={_LISTEN_TAG}",
        ]
    )
    if proc.returncode != 0:
        logger.warning(
            "Listen-in loopback was not loaded (no default sink is common on a VPS): %s",
            (proc.stderr or proc.stdout or "").strip(),
        )
        return False
    return True


def linux_speakers_muted() -> Optional[bool]:
    active = listen_in_active()
    if active is None:
        return None
    return not active


def play_wav_to_pulse_sink(wav_path: str | Path, sink_name: str) -> float:
    """Play a WAV into a Pulse/PipeWire sink. Returns duration in seconds."""
    paplay = shutil.which("paplay")
    if not paplay:
        raise RuntimeError(
            "paplay is not installed. Install pulseaudio-utils so TTS can reach Chromium."
        )
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate() or 1
            duration = wf.getnframes() / float(rate)
    except wave.Error:
        duration = 0.0
    proc = subprocess.run(
        [paplay, f"--device={sink_name}", str(wav_path)],
        capture_output=True,
        text=True,
        timeout=max(30, int(duration) + 15),
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "paplay failed").strip()
        raise RuntimeError(
            f"paplay could not play to Pulse sink '{sink_name}': {detail}. "
            "Create the sinks with pactl (the live run does this automatically)."
        )
    logger.info("TTS played %.2fs to Pulse sink %s", duration, sink_name)
    return duration


def capture_pulse_monitor(source: str, samplerate: int, frame_size: int, stop_event, put_frame) -> None:
    """Block, reading float32 mono frames from a Pulse monitor source into put_frame."""
    import numpy as np

    parec = shutil.which("parec")
    if not parec:
        raise RuntimeError(
            "parec is not installed. Install pulseaudio-utils so STT can hear the call."
        )
    proc = subprocess.Popen(
        [
            parec,
            f"--device={source}",
            "--format=float32le",
            f"--rate={int(samplerate)}",
            "--channels=1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    frame_bytes = int(frame_size) * 4
    try:
        assert proc.stdout is not None
        while not stop_event.is_set():
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            put_frame(np.frombuffer(buf, dtype=np.float32).copy())
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


def ensure_linux_call_audio(
    *,
    loopback_hint: str = SINK_TO_GV,
    capture_hint: str = CAPTURE_MONITOR,
    chrome_profile_dir: Path | None = None,
    set_windows_devices: bool = True,
    live_listen_enabled: bool = False,
):
    """Create the virtual sinks and Chromium mic prefs. Windows args are ignored."""
    from src.audio_setup import AudioSetupResult, patch_chrome_mic_prefs

    del loopback_hint, capture_hint, set_windows_devices
    result = AudioSetupResult(ok=False)
    result.loopback_hint = SINK_TO_GV
    result.capture_hint = CAPTURE_MONITOR
    result.mic_name = TTS_MONITOR
    result.speaker_name = SINK_FROM_GV

    if not pactl_available():
        result.messages.append(
            "pactl not found. Install pulseaudio-utils and run PulseAudio or PipeWire "
            "(see deploy/linux/README.md). TTS will not enter the call until the sinks exist."
        )
        return result

    tts_ok = ensure_null_sink(SINK_TO_GV, "Routelix_TTS_to_Google_Voice")
    call_ok = ensure_null_sink(SINK_FROM_GV, "Routelix_Google_Voice_Speaker")
    result.mic_set = tts_ok
    result.speaker_set = call_ok
    if tts_ok:
        result.messages.append(
            f"TTS sink {SINK_TO_GV} ready. Chromium microphone should be {TTS_MONITOR}."
        )
    else:
        result.messages.append(f"Failed to create Pulse sink {SINK_TO_GV}.")
    if call_ok:
        result.messages.append(
            f"Call sink {SINK_FROM_GV} ready. STT captures {CAPTURE_MONITOR}."
        )
    else:
        result.messages.append(f"Failed to create Pulse sink {SINK_FROM_GV}.")

    if set_linux_listen_in(bool(live_listen_enabled)):
        result.speaker_muted = not bool(live_listen_enabled)
        result.messages.append(
            "Listen-in looped to the default sink."
            if live_listen_enabled
            else "Listen-in off. Call audio stays on the virtual monitor (recordings still capture it)."
        )
    else:
        result.messages.append(
            "Could not change listen-in. A headless VPS often has no default speaker; "
            "use the Recordings tab to hear the call."
        )

    if chrome_profile_dir is not None:
        result.chrome_prefs_ok = patch_chrome_mic_prefs(Path(chrome_profile_dir))
        if result.chrome_prefs_ok:
            result.messages.append("Chromium profile allows the microphone for voice.google.com.")
        else:
            result.messages.append("Chromium microphone preference was not written.")

    result.ok = bool(tts_ok and call_ok)
    for msg in result.messages:
        logger.info("[AUDIO-SETUP] %s", msg)
    return result
