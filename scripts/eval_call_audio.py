#!/usr/bin/env python3
"""Score far-end detection and transcription on the downloaded eval set.

The "before" column is the pre-change rule set, copied here so it can be
re-run after the detectors change:

* energy VAD (noise counts as speech)
* beep = 700–1600 Hz energy ratio only
* steady RMS treated as a voicemail greeting
* short human keyword list and a 2.5 s speech cap

The "after" column calls the current detectors. Word error rate compares the
old peak-normalize conditioner with ``src.stt._condition_stt_audio`` on the
same faster-whisper model. Prosody compares sentence-only Piper rendering
with clause pauses on the same voice.

Audio is not committed. Build it first:

    python3 scripts/download_eval_audio.py
    python3 scripts/eval_call_audio.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "eval_audio" / "manifest.json"
DEFAULT_OUT = ROOT / "docs" / "eval" / "detection_results.json"

# Keyword list and human score from before the IVR / reply expansion.
_LEGACY_VM = (
    "leave a message",
    "leave your message",
    "leave a voicemail",
    "leave your name and number",
    "leave your name",
    "leave your number",
    "please leave your message",
    "please leave a message",
    "after the tone",
    "after the beep",
    "at the tone",
    "at the talent",
    "not available",
    "is not available",
    "not available to take",
    "not available to pick",
    "not available to answer",
    "unavailable to take",
    "unavailable to pick",
    "to pick up your call",
    "to pick your call",
    "mailbox",
    "your call has been forwarded",
    "please record your message",
    "record your message",
    "you've reached",
    "you have reached",
    "voicemail",
    "unable to pick",
    "unable to answer",
    "unable to take",
    "can't take your call",
    "cannot take your call",
    "cant take your call",
    "can't come to the phone",
    "cannot come to the phone",
    "not able to answer",
    "not able to take your call",
    "no one is available",
    "person you are trying to reach",
    "the person you called",
)
_LEGACY_CARRIER = (
    "telephone number",
    "the number you have dialed",
    "the number you dialed",
    "the wireless customer",
)
_LEGACY_SCREEN = (
    "state your name",
    "say your name",
    "record your name",
    "reason for your call",
    "reason for calling",
    "reason of your call",
    "reason of the call",
    "reason for the call",
    "purpose of your call",
    "purpose of the call",
    "who's calling",
    "who is calling",
    "screening service",
    "call is being screened",
    "this call may be screened",
    "try to connect you",
    "name and the reason",
    "name and reason",
    "tell me the reason",
    "tell us the reason",
    "tell the reason",
)
_LEGACY_HUMAN = (
    "hello",
    "hi",
    "hey",
    "yes",
    "yeah",
    "who is this",
    "assalamualaikum",
    "salam alaikum",
    "speaking",
)
_DIGIT_RUN = re.compile(r"\d[\d\s\-]{5,}\d")
_WORD_RE = re.compile(r"[a-z0-9']+")
_PROSODY_TEXT = (
    "Hi, this is Ashley from Indus Transports. "
    "Are you still dispatching solo? "
    "I can help with the M C number and a quick follow up."
)


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def _legacy_keyword(text: str) -> bool:
    lower = (text or "").lower()
    if any(phrase in lower for phrase in _LEGACY_VM):
        return True
    has_number = any(cue in lower for cue in _LEGACY_CARRIER)
    if not has_number:
        return False
    if _DIGIT_RUN.search(lower):
        return True
    return any(
        phrase in lower
        for phrase in (
            "can't take",
            "cannot take",
            "cant take",
            "not available",
            "leave your message",
            "leave a message",
            "after the tone",
            "at the tone",
        )
    )


def _legacy_screening(text: str) -> bool:
    lower = (text or "").lower()
    if not lower or _legacy_keyword(lower):
        return False
    return any(phrase in lower for phrase in _LEGACY_SCREEN)


def _legacy_human(text: str, duration: float, elapsed: float, has_speech: bool) -> bool:
    confidence = 0.0
    if text and any(re.search(rf"\b{re.escape(word)}\b", text, re.I) for word in _LEGACY_HUMAN):
        confidence += 0.45
    if 0.15 <= duration <= 2.5:
        confidence += 0.35
    if has_speech and elapsed <= 8.0:
        confidence += 0.15
    return confidence >= 0.70


def _legacy_beep_frame(frame: np.ndarray, sr: int) -> bool:
    samples = np.asarray(frame, dtype=np.float64)
    if samples.size < 64:
        return False
    if float(np.sqrt(np.mean(samples * samples))) < 0.01:
        return False
    spectrum = np.abs(np.fft.rfft(samples * np.hamming(samples.size)))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sr)
    total = float(spectrum.sum())
    if total <= 0:
        return False
    mask = (freqs >= 700) & (freqs <= 1600)
    return float(spectrum[mask].sum()) / total >= 0.55


def legacy_audio_label(audio: np.ndarray, sr: int, transcript: str = "") -> str:
    """Pre-change call decision on one clip. Trailing silence flushes VAD only."""
    frame = max(1, int(sr * 0.03))
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    beep_run = 0
    beep = False
    speech_frames = 0
    silence_frames = 0
    rms_window: list[float] = []
    machine_pattern = False
    greetings = 0
    labels: list[str] = []
    in_speech = False
    trigger = 0
    silence_count = 0
    speech_buf = 0
    segments: list[float] = []
    flush = np.concatenate([samples, np.zeros(int(sr * 0.45), dtype=np.float32)])
    original_frames = samples.size // frame
    for index in range(0, max(0, flush.size - frame), frame):
        chunk = flush[index : index + frame]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if index < original_frames:
            if not beep:
                if _legacy_beep_frame(chunk, sr):
                    beep_run += 1
                    if beep_run >= int(0.35 * 1000 / 30):
                        beep = True
                else:
                    beep_run = 0
            rms_window.append(rms)
            if len(rms_window) > 200:
                rms_window.pop(0)
            if rms >= 0.01:
                speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1
            label = "uncertain"
            if speech_frames >= 20:
                if silence_frames >= 8:
                    label = "voicemail_greeting"
                elif len(rms_window) >= 30:
                    arr = np.asarray(rms_window, dtype=np.float64)
                    cv = float(arr.std() / (arr.mean() + 1e-9))
                    if cv < 0.45 and speech_frames > 40:
                        machine_pattern = True
                        label = "voicemail_greeting"
                    else:
                        label = "live_call"
                else:
                    label = "live_call"
            labels.append(label)
            if label == "voicemail_greeting":
                greetings += 1
        threshold = 0.015 * (0.70 if in_speech else 1.0)
        is_speech = rms >= threshold
        if not in_speech:
            trigger = trigger + 1 if is_speech else max(0, trigger - 1)
            if trigger >= 2:
                in_speech = True
                silence_count = 0
                speech_buf = 0
        else:
            speech_buf += 1
            silence_count = 0 if is_speech else silence_count + 1
            if silence_count >= 12 or speech_buf >= int(25.0 / 0.03):
                segments.append(speech_buf * 0.03)
                in_speech = False
                trigger = 0
                silence_count = 0
    if beep or _legacy_keyword(transcript):
        return "machine"
    if _legacy_screening(transcript):
        return "human"
    duration = samples.size / float(sr or 1)
    has_speech = bool(segments) or any(label == "live_call" for label in labels)
    if _legacy_human(transcript, duration, 1.0, has_speech):
        return "human"
    if segments and 0.35 <= max(segments) <= 2.5 and not _legacy_keyword(transcript):
        return "human"
    if machine_pattern and greetings >= 6:
        return "machine"
    if segments:
        return "human"
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
    if rms < 0.005:
        return "silence"
    return "noise" if rms >= 0.015 else "silence"


def legacy_phrase_label(text: str) -> str:
    if not (text or "").strip():
        return "silence"
    if _legacy_keyword(text):
        return "machine"
    if _legacy_screening(text):
        return "human"
    if _legacy_human(text, 1.2, 1.0, True):
        return "human"
    return "uncertain"


def current_audio_label(audio: np.ndarray, sr: int, transcript: str = "") -> str:
    from src.audio_scene import decide_far_end

    return decide_far_end(audio, sr, transcript).label


def current_phrase_label(text: str) -> str:
    from src.audio_scene import classify_transcript

    return classify_transcript(text, speech_duration_seconds=1.2)


def _prf(truth: list[str], pred: list[str], labels: list[str]) -> dict:
    report = {"n": len(truth), "accuracy": 0.0, "per_class": {}}
    if not truth:
        return report
    report["accuracy"] = sum(t == p for t, p in zip(truth, pred)) / len(truth)
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, pred))
        fp = sum(t != label and p == label for t, p in zip(truth, pred))
        fn = sum(t == label and p != label for t, p in zip(truth, pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        report["per_class"][label] = {
            "precision": precision,
            "recall": recall,
            "support": sum(t == label for t in truth),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return report


def _silence_gaps(audio: np.ndarray, sr: int, rms_limit: float = 0.005, min_seconds: float = 0.07) -> list[float]:
    frame = max(1, int(sr * 0.01))
    gaps: list[float] = []
    run = 0
    for start in range(0, max(0, audio.size - frame), frame):
        chunk = audio[start : start + frame]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < rms_limit:
            run += 1
        else:
            if run * 0.01 >= min_seconds:
                gaps.append(run * 0.01)
            run = 0
    if run * 0.01 >= min_seconds:
        gaps.append(run * 0.01)
    return gaps


def _legacy_render_piper(text: str, voice) -> tuple[np.ndarray, int]:
    """Sentence split and a fixed 160 ms gap. Matches the previous renderer."""
    from src.realtime_tts import _piper_synthesize, _prepare_piper_text

    prepared = _prepare_piper_text(text)
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", prepared) if part.strip()]
    chunks: list[np.ndarray] = []
    rate = 22050
    for index, part in enumerate(parts):
        audio, rate = _piper_synthesize(
            part,
            voice,
            length_scale=1.07,
            noise_scale=0.0,
            noise_w_scale=0.0,
        )
        if index:
            chunks.append(np.zeros(int(rate * 0.16), dtype=np.float32))
        chunks.append(audio)
    if not chunks:
        return np.zeros(1, dtype=np.float32), rate
    if len(chunks) == 1:
        return chunks[0], rate
    return np.concatenate(chunks), rate


def _prosody() -> dict:
    model = ROOT / "models" / "piper" / "en_US-lessac-medium.onnx"
    config = Path(str(model) + ".json")
    if not model.is_file() or not config.is_file():
        return {"skipped": "piper voice model is not installed"}
    from src.realtime_tts import _load_piper_voice, _render_piper_phrase

    voice = _load_piper_voice(model)
    before_audio, before_sr = _legacy_render_piper(_PROSODY_TEXT, voice)
    started = time.perf_counter()
    after_audio, after_sr = _render_piper_phrase(
        _PROSODY_TEXT,
        voice,
        length_scale=1.07,
        noise_scale=0.0,
        noise_w_scale=0.0,
    )
    synth_s = time.perf_counter() - started
    before_gaps = _silence_gaps(before_audio, before_sr)
    after_gaps = _silence_gaps(after_audio, after_sr)
    return {
        "voice": model.name,
        "text": _PROSODY_TEXT,
        "before": {
            "seconds": float(before_audio.size / before_sr),
            "pause_count_70ms": len(before_gaps),
            "pause_seconds": [round(gap, 3) for gap in before_gaps],
        },
        "after": {
            "seconds": float(after_audio.size / after_sr),
            "pause_count_70ms": len(after_gaps),
            "pause_seconds": [round(gap, 3) for gap in after_gaps],
            "synth_seconds": synth_s,
        },
        "note": (
            "Both renders use length_scale 1.07 and noise scales 0 so the gap "
            "list does not move between runs. Pause count is runs of near-silence "
            "at least 70 ms. This is an acoustic proxy for clause breathing, not a listening test."
        ),
    }


def _legacy_condition(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if 0.03 <= peak < 0.22:
        audio = audio * (0.22 / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_words(reference)
    hyp = _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dist = np.zeros((rows, cols), dtype=np.int32)
    dist[:, 0] = np.arange(rows)
    dist[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i, j] = min(dist[i - 1, j] + 1, dist[i, j - 1] + 1, dist[i - 1, j - 1] + cost)
    return float(dist[-1, -1] / len(ref))


def _false_transcripts(rows: list[dict], audio_root: Path, model) -> dict:
    """How often Whisper invents words on noise and silence.

    The shipped path refuses noise before the model runs. Silence still goes
    through, then the low-energy hallucination filter can drop it.
    """
    from src.audio_scene import should_transcribe
    from src.stt import _is_low_energy_hallucination

    counts = {
        label: {"n": 0, "before_nonempty": 0, "after_nonempty": 0}
        for label in ("noise", "silence")
    }
    for row in rows:
        if row.get("wer") or row["label"] not in counts:
            continue
        audio, sr = _read(audio_root / row["path"])
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        conditioned = _legacy_condition(audio)
        segments, _info = model.transcribe(
            np.ascontiguousarray(conditioned),
            language="en",
            vad_filter=False,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if _is_low_energy_hallucination(text, peak):
            text = ""
        bucket = counts[row["label"]]
        bucket["n"] += 1
        if text:
            bucket["before_nonempty"] += 1
        if should_transcribe(audio, sr) and text:
            bucket["after_nonempty"] += 1
    # White noise is the case the call-path guard is aimed at. ESC-50 often
    # already comes back empty from tiny.en, so it does not show the guard.
    white = {"n": 8, "before_nonempty": 0, "after_nonempty": 0}
    rng = np.random.default_rng(0)
    for _ in range(white["n"]):
        audio = rng.normal(0.0, 0.08, 16000 * 2).astype(np.float32)
        peak = float(np.max(np.abs(audio)))
        segments, _info = model.transcribe(
            np.ascontiguousarray(_legacy_condition(audio)),
            language="en",
            vad_filter=False,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if _is_low_energy_hallucination(text, peak):
            text = ""
        if text:
            white["before_nonempty"] += 1
        if should_transcribe(audio, 16000) and text:
            white["after_nonempty"] += 1
    counts["white_noise"] = white
    for bucket in counts.values():
        n = bucket["n"] or 1
        bucket["before_rate"] = round(bucket["before_nonempty"] / n, 4)
        bucket["after_rate"] = round(bucket["after_nonempty"] / n, 4)
    return {
        "model": "tiny.en",
        "note": (
            "Rate is the fraction of clips where the transcript is non-empty. "
            "Noise is not sent to the model after the change. Silence still is."
        ),
        "counts": counts,
    }


def _load_whisper():
    from faster_whisper import WhisperModel

    model_dir = ROOT / "models" / "stt"
    model_dir.mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        "tiny.en",
        device="cpu",
        compute_type="int8",
        download_root=str(model_dir),
    )


def _transcribe_wer(rows: list[dict], audio_root: Path, model=None) -> dict:
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        return {"skipped": "faster-whisper is not installed"}
    from src.stt import _condition_stt_audio

    if model is None:
        model = _load_whisper()

    def decode(audio: np.ndarray) -> str:
        segments, _info = model.transcribe(
            audio,
            language="en",
            vad_filter=False,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    groups = {"clean": {"before": [], "after": []}, "noisy": {"before": [], "after": []}}
    details = []
    for row in rows:
        audio, sr = _read(audio_root / row["path"])
        if sr != 16000 and audio.size:
            target = int(round(audio.size * 16000 / sr))
            audio = np.interp(np.linspace(0, audio.size - 1, target), np.arange(audio.size), audio).astype(np.float32)
            sr = 16000
        before_audio = _legacy_condition(audio)
        after_audio = _condition_stt_audio(audio, sr)
        before_text = decode(before_audio)
        # The shipped conditioner matches the previous one. Decode once so a
        # second stochastic pass cannot invent a WER delta.
        if np.allclose(before_audio, after_audio):
            after_text = before_text
        else:
            after_text = decode(after_audio)
        before_wer = _wer(row["reference"], before_text)
        after_wer = _wer(row["reference"], after_text)
        condition = row.get("condition") or "clean"
        groups[condition]["before"].append(before_wer)
        groups[condition]["after"].append(after_wer)
        details.append(
            {
                "id": row["id"],
                "condition": condition,
                "before_wer": before_wer,
                "after_wer": after_wer,
            }
        )
    summary = {}
    for name, bucket in groups.items():
        if not bucket["before"]:
            continue
        summary[name] = {
            "n": len(bucket["before"]),
            "before_wer": round(float(np.mean(bucket["before"])), 4),
            "after_wer": round(float(np.mean(bucket["after"])), 4),
        }
    return {"model": "tiny.en", "summary": summary, "clips": details}


def _round_report(report: dict) -> dict:
    out = {"n": report["n"], "accuracy": round(report["accuracy"], 4), "per_class": {}}
    for label, stats in report["per_class"].items():
        out["per_class"][label] = {
            "precision": round(stats["precision"], 4),
            "recall": round(stats["recall"], 4),
            "support": stats["support"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-wer", action="store_true")
    parser.add_argument("--skip-prosody", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    scene_rows = [row for row in payload["clips"] if not row.get("wer")]
    labels = ["human", "machine", "silence", "noise"]
    acoustic_truth = []
    acoustic_before = []
    acoustic_after = []
    fusion_truth = []
    fusion_before = []
    fusion_after = []
    by_source: dict[str, dict[str, int]] = {}
    for row in scene_rows:
        audio, sr = _read(root / row["path"])
        truth = row["label"]
        before = legacy_audio_label(audio, sr, "")
        after = current_audio_label(audio, sr, "")
        acoustic_truth.append(truth)
        acoustic_before.append(before)
        acoustic_after.append(after)
        transcript = row.get("transcript") or ""
        fusion_truth.append(truth)
        fusion_before.append(legacy_audio_label(audio, sr, transcript))
        fusion_after.append(current_audio_label(audio, sr, transcript))
        bucket = by_source.setdefault(row.get("source") or "unknown", {"n": 0, "before": 0, "after": 0})
        bucket["n"] += 1
        bucket["before"] += int(before == truth)
        bucket["after"] += int(after == truth)

    phrase_truth = []
    phrase_before = []
    phrase_after = []
    for row in payload.get("phrases") or []:
        phrase_truth.append(row["label"])
        phrase_before.append(legacy_phrase_label(row["text"]))
        phrase_after.append(current_phrase_label(row["text"]))

    results = {
        "dataset": {
            "manifest": str(args.manifest),
            "scene_clips": len(scene_rows),
            "phrases": len(phrase_truth),
            "sources": sorted({row.get("source") for row in scene_rows}),
        },
        "acoustic": {
            "description": "Transcript forced empty. Labels are human, machine, silence, noise.",
            "before": _round_report(_prf(acoustic_truth, acoustic_before, labels)),
            "after": _round_report(_prf(acoustic_truth, acoustic_after, labels)),
            "by_source_correct": by_source,
        },
        "with_known_transcript": {
            "description": (
                "Same audio. espeak prompts keep their text, which is what a correct "
                "STT pass would hand the keyword detector. Other clips have no transcript."
            ),
            "before": _round_report(_prf(fusion_truth, fusion_before, labels)),
            "after": _round_report(_prf(fusion_truth, fusion_after, labels)),
        },
        "phrases": {
            "description": "Hand-labeled lines. uncertain predictions count as misses.",
            "before": _round_report(_prf(phrase_truth, phrase_before, ["human", "machine", "silence"])),
            "after": _round_report(_prf(phrase_truth, phrase_after, ["human", "machine", "silence"])),
        },
    }
    if not args.skip_prosody:
        results["piper_prosody"] = _prosody()
    if not args.skip_wer:
        try:
            whisper = _load_whisper()
        except ImportError:
            whisper = None
            results["wer"] = {"skipped": "faster-whisper is not installed"}
            results["false_transcripts"] = {"skipped": "faster-whisper is not installed"}
        if whisper is not None:
            wer_rows = [row for row in payload["clips"] if row.get("wer")]
            results["wer"] = _transcribe_wer(wer_rows, root, whisper)
            results["false_transcripts"] = _false_transcripts(scene_rows, root, whisper)
        # The clip list is long; keep the summary in the committed file and
        # drop per-clip rows if they are present so the doc stays small.
        if isinstance(results["wer"], dict) and "clips" in results["wer"]:
            results["wer"]["clip_count"] = len(results["wer"]["clips"])
            del results["wer"]["clips"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
