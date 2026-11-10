#!/usr/bin/env python3
"""Download a small public audio set for detection and transcription evals.

Nothing this script writes is meant to be committed. Files land in
``data/eval_audio/`` (gitignored) or a directory passed with ``--dest``.

Sources, all freely redistributable for research:

* LibriSpeech test-clean (OpenSLR SLR12) — read English speech, with transcripts
* ESC-50 (Karol Piczak, CC BY-NC) — environmental noise, non-tonal classes only
* Synthetic silence, voicemail beeps, and espeak-ng IVR/voicemail prompts

espeak-ng must be on PATH for the machine-speech prompts. The script still
builds the speech, noise, silence, and beep clips if espeak-ng is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - download path expects soundfile
    sf = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "eval_audio"
LIBRISPEECH_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
ESC50_CSV_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{name}"

# Non-tonal ESC-50 targets. Sirens, alarms, horns, and bells are excluded so a
# pure tone is not labeled "noise".
ESC50_NOISE_TARGETS = {
    10,  # rain
    11,  # sea waves
    12,  # crackling fire
    16,  # wind
    17,  # pouring water
    19,  # thunderstorm
    32,  # keyboard typing
    35,  # washing machine
    36,  # vacuum cleaner
    40,  # helicopter
    44,  # engine
    45,  # train
    47,  # airplane
}

MACHINE_PROMPTS = (
    "Hi, you have reached Sam. I can't come to the phone right now. Please leave a message after the beep.",
    "The person you are trying to reach is not available. Please leave your name and number after the tone.",
    "You have reached the voicemail box of Jordan Lee. Please record your message after the tone.",
    "Telephone number four one seven, five five five, zero one zero one is not available to take your call.",
    "Your call has been forwarded to an automatic voice message system. Leave your message after the tone.",
    "The mailbox is full. Please hang up and try your call again later.",
    "The number you have reached has been disconnected or is no longer in service.",
    "Thank you for calling. For English, press one. Para espanol, press two.",
    "Please listen carefully. Our menu options have changed. Press one for sales. Press two for support.",
    "If you know your party's extension, you may dial it at any time. Press star to return to the main menu.",
    "To speak with an operator, press zero. To repeat this message, press nine.",
    "Hello. This is an automated attendant. Press one to continue. Press two to leave a voicemail.",
)

# Short human replies a live pickup actually says. Used as oracle transcripts
# on real LibriSpeech audio in the fusion eval, and as their own phrase rows.
HUMAN_PHRASES = (
    "Hello?",
    "Yeah, who is this?",
    "Speaking.",
    "Okay, I can hear you.",
    "Yep, this is Mike.",
    "Hey, what's this about?",
    "Hold on one second.",
    "Not interested, thanks.",
    "Who are you with?",
    "Can I help you?",
    "Go ahead.",
    "Sorry, say that again?",
    "Nah, we already have somebody.",
    "Alright, I'm listening.",
    "What do you want?",
    "Hi, this is Dana.",
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    print(f"GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _write_wav(path: Path, audio: np.ndarray, samplerate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    if sf is not None:
        sf.write(str(path), pcm.astype(np.float32), samplerate)
        return
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((pcm * 32767.0).astype(np.int16).tobytes())


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32)
    dst_n = max(1, int(round(audio.size * dst_sr / src_sr)))
    positions = np.linspace(0, audio.size - 1, dst_n)
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile is required to read flac/wav eval audio")
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def _clip(audio: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(sr * seconds)
    if audio.size >= n:
        return audio[:n].astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: audio.size] = audio
    return out


def _load_transcripts(root: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in root.rglob("*.trans.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1].strip()
    return texts


def download_librispeech(dest: Path, limit: int) -> list[dict]:
    archive = dest / "cache" / "test-clean.tar.gz"
    extract = dest / "cache" / "LibriSpeech"
    _download(LIBRISPEECH_URL, archive)
    if not any(extract.rglob("*.flac")):
        print("extracting LibriSpeech flac files")
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".flac") or m.name.endswith(".trans.txt")]
            # Keep the archive extraction bounded: transcripts plus the first files.
            flacs = [m for m in members if m.name.endswith(".flac")][: max(limit * 3, limit)]
            texts = [m for m in members if m.name.endswith(".trans.txt")]
            tar.extractall(extract, members=texts + flacs)
    transcripts = _load_transcripts(extract)
    rows = []
    flacs = sorted(extract.rglob("*.flac"))
    for path in flacs:
        if len(rows) >= limit:
            break
        text = transcripts.get(path.stem, "")
        if not text:
            continue
        audio, sr = _read_audio(path)
        audio = _resample(audio, sr)
        # Keep a few seconds. Long chapters dominate the RMS stats.
        audio = _clip(audio, 16000, 4.0)
        out = dest / "prepared" / "human" / f"{path.stem}.wav"
        _write_wav(out, audio)
        rows.append(
            {
                "id": f"human-{path.stem}",
                "label": "human",
                "path": str(out.relative_to(dest)),
                "transcript": "",
                "reference": text,
                "source": "librispeech-test-clean",
            }
        )
    return rows


def download_esc50(dest: Path, limit: int) -> list[dict]:
    csv_path = dest / "cache" / "esc50.csv"
    _download(ESC50_CSV_URL, csv_path)
    chosen: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                target = int(row["target"])
            except (KeyError, ValueError):
                continue
            if target in ESC50_NOISE_TARGETS:
                chosen.append(row["filename"])
            if len(chosen) >= limit:
                break
    rows = []
    raw_dir = dest / "cache" / "esc50"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        raw = raw_dir / name
        _download(ESC50_AUDIO_URL.format(name=name), raw)
        audio, sr = _read_audio(raw)
        audio = _resample(audio, sr)
        audio = _clip(audio, 16000, 3.0)
        out = dest / "prepared" / "noise" / name
        _write_wav(out, audio)
        rows.append(
            {
                "id": f"noise-{Path(name).stem}",
                "label": "noise",
                "path": str(out.relative_to(dest)),
                "transcript": "",
                "reference": "",
                "source": "esc50",
            }
        )
    return rows


def build_silence(dest: Path, count: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(7)
    for index in range(count):
        if index % 2 == 0:
            audio = np.zeros(16000 * 2, dtype=np.float32)
            kind = "digital"
        else:
            # Below the production energy threshold (~0.015 RMS).
            audio = rng.normal(0.0, 0.0008, 16000 * 2).astype(np.float32)
            kind = "near"
        out = dest / "prepared" / "silence" / f"{kind}-{index:02d}.wav"
        _write_wav(out, audio)
        rows.append(
            {
                "id": f"silence-{kind}-{index:02d}",
                "label": "silence",
                "path": str(out.relative_to(dest)),
                "transcript": "",
                "reference": "",
                "source": "synthetic-silence",
            }
        )
    return rows


def build_beeps(dest: Path, count: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(11)
    for index in range(count):
        freq = float(rng.uniform(760, 1480))
        duration = float(rng.uniform(0.42, 0.8))
        t = np.arange(int(16000 * duration)) / 16000.0
        tone = (0.28 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        # Real beeps are not perfect lab sines: a weak harmonic plus a little hiss.
        tone += (0.03 * np.sin(2 * np.pi * 2 * freq * t)).astype(np.float32)
        tone += rng.normal(0.0, 0.008, tone.shape).astype(np.float32)
        audio = np.zeros(16000 * 2, dtype=np.float32)
        start = int(16000 * 0.35)
        audio[start : start + tone.size] = np.clip(tone, -1, 1)
        out = dest / "prepared" / "machine" / f"beep-{index:02d}.wav"
        _write_wav(out, audio)
        rows.append(
            {
                "id": f"machine-beep-{index:02d}",
                "label": "machine",
                "path": str(out.relative_to(dest)),
                "transcript": "",
                "reference": "",
                "source": "synthetic-beep",
                "beep_hz": round(freq, 1),
            }
        )
    return rows


def build_speech_then_beep(dest: Path, human_rows: list[dict], count: int) -> list[dict]:
    rows = []
    for index, human in enumerate(human_rows[:count]):
        audio, sr = _read_audio(dest / human["path"])
        audio = _resample(audio, sr)[: int(16000 * 1.6)]
        gap = np.zeros(int(16000 * 0.25), dtype=np.float32)
        t = np.arange(int(16000 * 0.55)) / 16000.0
        beep = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        mixed = np.concatenate([audio, gap, beep]).astype(np.float32)
        out = dest / "prepared" / "machine" / f"speech-beep-{index:02d}.wav"
        _write_wav(out, mixed)
        rows.append(
            {
                "id": f"machine-speech-beep-{index:02d}",
                "label": "machine",
                "path": str(out.relative_to(dest)),
                "transcript": "",
                "reference": human.get("reference", ""),
                "source": "librispeech-plus-beep",
            }
        )
    return rows


def build_espeak(dest: Path) -> list[dict]:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        print("espeak-ng not found; skipping synthesized mailbox prompts")
        return []
    rows = []
    for index, text in enumerate(MACHINE_PROMPTS):
        raw = dest / "cache" / "espeak" / f"machine-{index:02d}.wav"
        raw.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [exe, "-s", "150", "-v", "en-us", "-w", str(raw), text],
            check=True,
            capture_output=True,
        )
        audio, sr = _read_audio(raw)
        audio = _resample(audio, sr)
        audio = _clip(audio, 16000, 8.0)
        out = dest / "prepared" / "machine" / f"espeak-{index:02d}.wav"
        _write_wav(out, audio)
        rows.append(
            {
                "id": f"machine-espeak-{index:02d}",
                "label": "machine",
                "path": str(out.relative_to(dest)),
                "transcript": text,
                "reference": text,
                "source": "espeak-ng",
            }
        )
    return rows


def build_wer_set(dest: Path, human_rows: list[dict], noise_rows: list[dict], limit: int) -> list[dict]:
    """Full-length LibriSpeech references, plus the same clip at about 8 dB SNR."""
    rows = []
    if not human_rows:
        return rows
    noise_audio = None
    if noise_rows:
        chunks = []
        for row in noise_rows[:8]:
            audio, sr = _read_audio(dest / row["path"])
            chunks.append(_resample(audio, sr))
        noise_audio = np.concatenate(chunks) if chunks else None
    # Re-read a longer slice from the cached flac when we still have it.
    extract = dest / "cache" / "LibriSpeech"
    flacs = {path.stem: path for path in extract.rglob("*.flac")} if extract.exists() else {}
    rng = np.random.default_rng(19)
    for human in human_rows[:limit]:
        stem = human["id"].split("human-", 1)[-1]
        source = flacs.get(stem)
        if source is None:
            audio, sr = _read_audio(dest / human["path"])
            audio = _resample(audio, sr)
        else:
            audio, sr = _read_audio(source)
            audio = _resample(audio, sr)
        audio = audio[: 16000 * 18]
        clean = dest / "prepared" / "wer" / f"{stem}-clean.wav"
        _write_wav(clean, audio)
        rows.append(
            {
                "id": f"wer-clean-{stem}",
                "label": "human",
                "path": str(clean.relative_to(dest)),
                "transcript": "",
                "reference": human["reference"],
                "source": "librispeech-wer-clean",
                "wer": True,
                "condition": "clean",
            }
        )
        if noise_audio is not None and noise_audio.size > 1000:
            tiled = np.resize(noise_audio, audio.size).astype(np.float32)
            speech_rms = float(np.sqrt(np.mean(audio**2)) + 1e-8)
            noise_rms = float(np.sqrt(np.mean(tiled**2)) + 1e-8)
            target = speech_rms / (10 ** (8.0 / 20.0))
            mixed = audio + tiled * (target / noise_rms)
            # A little room tone on each end, the way a phone segment arrives.
            pad = rng.normal(0.0, max(target, 0.01), int(16000 * 0.35)).astype(np.float32)
            mixed = np.concatenate([pad, mixed, pad]).astype(np.float32)
            noisy = dest / "prepared" / "wer" / f"{stem}-noisy.wav"
            _write_wav(noisy, np.clip(mixed, -1, 1))
            rows.append(
                {
                    "id": f"wer-noisy-{stem}",
                    "label": "human",
                    "path": str(noisy.relative_to(dest)),
                    "transcript": "",
                    "reference": human["reference"],
                    "source": "librispeech-wer-noisy-8db",
                    "wer": True,
                    "condition": "noisy",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--human", type=int, default=40)
    parser.add_argument("--noise", type=int, default=40)
    parser.add_argument("--silence", type=int, default=20)
    parser.add_argument("--beeps", type=int, default=15)
    parser.add_argument("--speech-beeps", type=int, default=10)
    parser.add_argument("--wer", type=int, default=24)
    args = parser.parse_args()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    human = download_librispeech(dest, args.human)
    manifest.extend(human)
    manifest.extend(download_esc50(dest, args.noise))
    manifest.extend(build_silence(dest, args.silence))
    manifest.extend(build_beeps(dest, args.beeps))
    manifest.extend(build_speech_then_beep(dest, human, args.speech_beeps))
    manifest.extend(build_espeak(dest))
    manifest.extend(build_wer_set(dest, human, [r for r in manifest if r["label"] == "noise"], args.wer))
    phrases = []
    for text in HUMAN_PHRASES:
        phrases.append({"text": text, "label": "human"})
    for text in MACHINE_PROMPTS:
        phrases.append({"text": text, "label": "machine"})
    phrases.extend(
        [
            {"text": "Please state your name and the reason for your call.", "label": "human"},
            {"text": "Who is calling?", "label": "human"},
            {"text": "This call may be screened.", "label": "human"},
            {"text": "", "label": "silence"},
            {"text": "   ", "label": "silence"},
        ]
    )
    payload = {"clips": manifest, "phrases": phrases, "human_phrases": list(HUMAN_PHRASES)}
    out = dest / "manifest.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in manifest:
        if row.get("wer"):
            counts["wer"] = counts.get("wer", 0) + 1
        else:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(json.dumps({"dest": str(dest), "counts": counts, "phrases": len(phrases)}, indent=2))


if __name__ == "__main__":
    main()
