"""Linux server helpers must import without Windows-only modules."""
import ast
import sys
from pathlib import Path
from unittest.mock import patch

import src.audio_setup  # noqa: F401
import src.google_voice  # noqa: F401
import src.gv_accounts  # noqa: F401
import src.web_app  # noqa: F401
from src.browser_launch import _is_snap, find_chromium_binary
from src.linux_audio import SINK_FROM_GV, SINK_TO_GV, ensure_linux_call_audio
from src.web_app import RESERVED_PORTS, _server_port


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_windows_modules_are_not_imported_at_module_level():
    root = Path(__file__).resolve().parents[1] / "src"
    for name in ("web_app.py", "audio_setup.py", "google_voice.py", "gv_accounts.py", "audio_capture.py"):
        imported = _top_level_imports(root / name)
        assert "winreg" not in imported, name
        assert "pycaw" not in imported, name
    assert "src.web_app" in sys.modules


def test_snap_chromium_is_rejected(tmp_path):
    stub = tmp_path / "chromium"
    stub.write_text("#!/bin/sh\nexec snap run chromium\n", encoding="utf-8")
    assert _is_snap(str(stub)) is True


def test_env_chromium_binary_is_used(tmp_path, monkeypatch):
    binary = tmp_path / "chromium"
    binary.write_text("#!/bin/sh\nexec chromium\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("CHROMIUM_BINARY", str(binary))
    monkeypatch.delenv("CHROME_BINARY", raising=False)
    assert find_chromium_binary() == str(binary)


def test_linux_audio_route_uses_pactl():
    sinks = {SINK_TO_GV, SINK_FROM_GV}

    def fake_sinks():
        return sorted(sinks)

    with (
        patch("src.linux_audio.pactl_available", return_value=True),
        patch("src.linux_audio.list_sinks", side_effect=fake_sinks),
        patch("src.linux_audio.ensure_null_sink", return_value=True),
        patch("src.linux_audio.set_linux_listen_in", return_value=True),
        patch("src.audio_setup.patch_chrome_mic_prefs", return_value=True),
    ):
        result = ensure_linux_call_audio(chrome_profile_dir=None)
    assert result.ok is True
    assert result.loopback_hint == SINK_TO_GV
    assert result.capture_hint == f"{SINK_FROM_GV}.monitor"


def test_named_pulse_sink_does_not_use_portaudio_names(monkeypatch):
    """sounddevice only lists pulse/default. The route test must still select the sink."""
    import src.voice_playback as playback

    monkeypatch.setattr(playback, "_pulse_sink_hint", lambda hint: hint.startswith("routelix_"))
    with (
        patch("src.linux_audio.ensure_null_sink", return_value=True) as ensure,
        patch("src.linux_audio.probe_pulse_sink", return_value=(True, "Pulse sink routelix_to_gv accepted playback")),
        patch("src.voice_playback.find_loopback_devices", side_effect=AssertionError("PortAudio lookup")),
    ):
        index = playback.find_playable_loopback_device("routelix_to_gv")
    assert index == playback.PULSE_OUTPUT_INDEX
    ensure.assert_called_once()
    assert "routelix_to_gv" in playback.describe_audio_device(index)


def test_pulse_playback_sets_pulse_sink_env(tmp_path, monkeypatch):
    wav = tmp_path / "tone.wav"
    import wave
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x10" * 160)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env") or {}
        class Result:
            returncode = 0
            stderr = ""
            stdout = ""
        return Result()

    with (
        patch("src.linux_audio.shutil.which", return_value="/usr/bin/paplay"),
        patch("src.linux_audio.list_sinks", return_value=["routelix_to_gv"]),
        patch("src.linux_audio.wake_sink"),
        patch("src.linux_audio.subprocess.run", side_effect=fake_run),
    ):
        from src.linux_audio import play_wav_to_pulse_sink
        play_wav_to_pulse_sink(wav, "routelix_to_gv")
    assert seen["cmd"][0].endswith("paplay")
    assert "--device=routelix_to_gv" in seen["cmd"]
    assert seen["env"]["PULSE_SINK"] == "routelix_to_gv"


def test_preflight_reports_suspended_sink(monkeypatch):
    from src.preflight import check_audio_loopback

    monkeypatch.setattr("src.linux_audio.pactl_available", lambda: True)
    monkeypatch.setattr("src.linux_audio.ensure_null_sink", lambda name, description: True)
    monkeypatch.setattr(
        "src.linux_audio.probe_pulse_sink",
        lambda sink, timeout_s=5.0: (False, "Playback to routelix_to_gv timed out because PipeWire suspended the idle sink."),
    )
    monkeypatch.setattr("src.preflight._linux_virtual_hint", lambda hint: True)
    result = check_audio_loopback("routelix_to_gv")
    assert result.status == "fail"
    assert "suspended" in result.message


def test_reserved_ports_are_refused(monkeypatch):
    monkeypatch.setenv("ROUTELIX_PORT", "9222")
    try:
        _server_port()
    except SystemExit as exc:
        assert "9222" in str(exc)
    else:
        raise AssertionError("reserved port was accepted")
    assert 27017 in RESERVED_PORTS
    assert 8787 not in RESERVED_PORTS
