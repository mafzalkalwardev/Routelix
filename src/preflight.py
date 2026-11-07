"""Preflight checks — run before dialing to verify all dependencies are ready."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.paths import runtime_base

BASE_DIR = runtime_base()


@dataclass
class CheckResult:
    name: str
    status: str  # "ok" | "warn" | "fail"
    message: str


def check_env() -> CheckResult:
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return CheckResult("ENV File", "warn", ".env not found — using system env vars only")
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return CheckResult("ENV File", "fail", ".env found but GROQ_API_KEY is missing or placeholder")
    return CheckResult("ENV File", "ok", ".env present and GROQ_API_KEY is set")


def check_groq_api(api_key: Optional[str] = None) -> CheckResult:
    if os.getenv("OFFLINE_MODE", "").lower() in ("1", "true", "yes", "on"):
        return CheckResult("Groq API", "warn", "Offline mode enabled — external Groq call skipped")
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("your_"):
        return CheckResult("Groq API", "fail", "GROQ_API_KEY not set — cannot connect")
    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=key)
        models = client.models.list()
        count = len(list(models.data))
        return CheckResult("Groq API", "ok", f"Connected — {count} models available")
    except Exception as exc:
        short = str(exc)[:120]
        return CheckResult("Groq API", "fail", f"Connection failed: {short}")


def check_contacts(contacts_file: Optional[Path] = None) -> CheckResult:
    if contacts_file is None:
        contacts_file = Path(
            os.getenv("CONTACTS_FILE", str(BASE_DIR / "data" / "contacts.xlsx"))
        )
    contacts_file = Path(contacts_file)
    if not contacts_file.exists():
        try:
            display = contacts_file.relative_to(BASE_DIR)
        except ValueError:
            display = contacts_file.name
        return CheckResult("Contacts File", "fail", f"Not found: {display}")
    try:
        from src.contacts import load_contacts  # type: ignore
        rows = load_contacts(contacts_file)
        if not rows:
            return CheckResult("Contacts File", "warn", f"Loaded but empty: {contacts_file.name}")
        return CheckResult("Contacts File", "ok", f"{len(rows)} contacts in {contacts_file.name}")
    except Exception as exc:
        return CheckResult("Contacts File", "fail", f"Parse error: {exc}")


def check_chrome_profile(profile_name: Optional[str] = None) -> CheckResult:
    name = profile_name or os.getenv("PROFILE_NAME", "sales_profile")
    profile_path = BASE_DIR / "chrome_profiles" / name
    if not profile_path.exists():
        return CheckResult(
            "Chrome Profile", "warn",
            f"chrome_profiles/{name} not found — will be created on first launch"
        )
    return CheckResult("Chrome Profile", "ok", f"chrome_profiles/{name} found")


def check_audio_loopback(device_hint: Optional[str] = None) -> CheckResult:
    hint = device_hint or os.getenv("LOOPBACK_DEVICE", "CABLE Input")
    try:
        from src.voice_playback import find_loopback_devices, list_audio_devices, probe_output_device  # type: ignore
        devices = list_audio_devices()
        if not devices:
            return CheckResult(
                "Audio Loopback", "fail",
                "No audio devices found — is sounddevice installed?"
            )
        matches = find_loopback_devices(hint)
        if not matches:
            return CheckResult(
                "Audio Loopback", "fail",
                f"'{hint}' not found. Install VB-CABLE from vb-audio.com/Cable/"
            )
        failures = []
        for idx in matches:
            ok, detail = probe_output_device(idx)
            if ok:
                return CheckResult("Audio Loopback", "ok", f"Loopback device [{idx}] is playable for '{hint}'")
            failures.append(f"[{idx}] {detail}")
        return CheckResult(
            "Audio Loopback", "fail",
            f"Matched '{hint}', but no output device could be opened: {'; '.join(failures)}"
        )
    except Exception as exc:
        return CheckResult("Audio Loopback", "fail", f"Audio check error: {exc}")


def check_capture_device(device_hint: Optional[str] = None) -> CheckResult:
    """
    Verify Python's prospect-audio capture path for realtime STT.
    CAPTURE_DEVICE is not the Chrome/Google Voice microphone selector.

    Audio flow: TTS → CABLE Input (output) → VB-CABLE → CABLE Output (input) → Chrome mic.
    Chrome must use CABLE Output as its microphone for voice.google.com; if it uses the
    laptop mic instead, Tony's voice never reaches the call.
    """
    hint = device_hint or os.getenv("CAPTURE_DEVICE", "default")
    if not hint or hint.lower() in ("default", "wasapi", "default (wasapi)"):
        try:
            import soundcard as sc  # type: ignore
            speaker = sc.default_speaker()
            speaker_name = getattr(speaker, "name", "unknown speaker")
            status = "warn" if "cable input" in str(speaker_name).lower() else "ok"
            message = (
                f"CAPTURE_DEVICE='{hint}' uses WASAPI loopback of Windows default speaker: "
                f"{speaker_name}. Chrome's Google Voice mic still must be CABLE Output."
            )
            if status == "warn":
                message += (
                    " Default speaker is the TTS cable, so STT may hear Tony/silence. "
                    "Use real speakers for single-cable capture or a second cable for prospect audio."
                )
            return CheckResult("Capture Device", status, message)
        except ImportError:
            return CheckResult(
                "Capture Device", "fail",
                "CAPTURE_DEVICE='default' needs soundcard for WASAPI loopback. "
                "Run: pip install soundcard",
            )
        except Exception as exc:
            return CheckResult("Capture Device", "warn", f"WASAPI capture check error: {exc}")
        return CheckResult(
            "Capture Device", "warn",
            f"CAPTURE_DEVICE is '{hint}' — Chrome will fall back to the Windows default mic. "
            "Set CAPTURE_DEVICE to 'CABLE Output' so Tony's TTS reaches the call, "
            "then set Chrome mic to 'CABLE Output' for voice.google.com.",
        )
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        hint_lower = hint.lower()
        matches = [
            d for d in devices
            if hint_lower in d["name"].lower() and d["max_input_channels"] > 0
        ]
        if not matches:
            return CheckResult(
                "Capture Device", "fail",
                f"'{hint}' not found as a recording device. "
                "Install VB-CABLE (https://vb-audio.com/Cable/), then set Chrome mic "
                "to 'CABLE Output' for voice.google.com.",
            )
        dev = matches[0]
        return CheckResult(
            "Capture Device", "ok",
            f"'{dev['name']}' found as recording device — "
            "confirm Chrome mic is set to this device for voice.google.com.",
        )
    except Exception as exc:
        return CheckResult("Capture Device", "warn", f"Capture device check error: {exc}")


def check_piper_voice(model_path: Optional[str] = None) -> CheckResult:
    """Confirm the local Piper voice is installed when that engine is selected."""
    engine = os.getenv("TTS_ENGINE", "edge").strip().lower()
    if engine not in {"piper", "piper-tts"}:
        return CheckResult("Piper Voice", "ok", f"TTS engine is '{engine or 'edge'}'; Piper model is not required")
    try:
        import piper  # noqa: F401
    except ImportError:
        return CheckResult(
            "Piper Voice", "fail",
            "piper-tts is not installed. Run: pip install -r requirements.txt",
        )
    raw = model_path or os.getenv("PIPER_MODEL_PATH", "models/piper/en_US-lessac-medium.onnx")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    config_path = Path(str(path) + ".json")
    if not path.is_file():
        return CheckResult("Piper Voice", "fail", f"Model not found: {path}")
    if not config_path.is_file():
        return CheckResult("Piper Voice", "fail", f"Voice config not found: {config_path}")
    return CheckResult("Piper Voice", "ok", f"Piper voice ready: {path.name}")


def check_local_models() -> CheckResult:
    """Confirm on-device answer, hearing, and voice files when those backends are selected."""
    llm = os.getenv("LLM_BACKEND", "groq").strip().lower()
    stt = os.getenv("STT_BACKEND", "groq").strip().lower()
    engine = os.getenv("TTS_ENGINE", "edge").strip().lower()
    if llm != "local" and stt != "local" and engine != "kokoro":
        return CheckResult("Local Models", "ok", "Cloud backends selected; local model files are not required")
    missing: list[str] = []
    if llm == "local":
        model = Path(os.getenv("LOCAL_LLM_PATH", "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf")).expanduser()
        if not model.is_absolute():
            model = BASE_DIR / model
        if not model.is_file():
            missing.append(str(model))
    if engine == "kokoro":
        voice = Path(os.getenv("KOKORO_MODEL_PATH", "models/kokoro/kokoro-v1.0.onnx")).expanduser()
        voices = Path(os.getenv("KOKORO_VOICES_PATH", "models/kokoro/voices-v1.0.bin")).expanduser()
        if not voice.is_absolute():
            voice = BASE_DIR / voice
        if not voices.is_absolute():
            voices = BASE_DIR / voices
        if not voice.is_file():
            missing.append(str(voice))
        if not voices.is_file():
            missing.append(str(voices))
    if stt == "local":
        root = Path(os.getenv("STT_MODEL_DIR", "models/stt")).expanduser()
        if not root.is_absolute():
            root = BASE_DIR / root
        if not root.exists() or not any(root.rglob("model.bin")):
            missing.append(str(root))
    if missing:
        return CheckResult("Local Models", "fail", "Missing local model files: " + "; ".join(missing))
    return CheckResult("Local Models", "ok", "Local answer, speech recognition, and voice models are on disk")


def check_callback_number() -> CheckResult:
    number = os.getenv("CALLBACK_NUMBER", os.getenv("GOOGLE_VOICE_NUMBER", ""))
    # also check dialer_config.json
    if not number:
        config_file = BASE_DIR / "dialer_config.json"
        if config_file.exists():
            import json
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
                number = data.get("callback_number", "")
            except Exception:
                pass
    if not number:
        return CheckResult("Callback Number", "fail", "CALLBACK_NUMBER not configured")
    masked = number[:3] + "***" + number[-2:] if len(number) > 5 else "***"
    return CheckResult("Callback Number", "ok", f"Callback number configured: {masked}")


def run_all(
    groq_api_key: Optional[str] = None,
    contacts_file: Optional[Path] = None,
    profile_name: Optional[str] = None,
    loopback_device: Optional[str] = None,
    capture_device: Optional[str] = None,
) -> List[CheckResult]:
    return [
        check_env(),
        check_groq_api(groq_api_key),
        check_contacts(contacts_file),
        check_chrome_profile(profile_name),
        check_audio_loopback(loopback_device),
        check_capture_device(capture_device),
        check_piper_voice(),
        check_local_models(),
        check_callback_number(),
    ]
