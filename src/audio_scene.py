"""Frame-level far-end audio scene labels.

The call path used to treat any loud frame as speech and any steady loud
stretch as a voicemail greeting. These helpers separate silence, noise,
live speech, and a narrow voicemail beep using only numpy. No model file
is required, and nothing here changes a call decision by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FrameFeatures:
    rms: float
    zcr: float
    flatness: float
    centroid_hz: float
    harmonicity: float
    dominant_hz: float
    band_ratio: float
    lobe: float


@dataclass(frozen=True)
class FarEndDecision:
    label: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


def analyze_frame(
    frame: np.ndarray,
    samplerate: int,
    *,
    beep_low_hz: float = 700.0,
    beep_high_hz: float = 1600.0,
) -> FrameFeatures:
    """Spectral snapshot of one 20–40 ms frame."""
    samples = np.asarray(frame, dtype=np.float64).reshape(-1)
    if samples.size < 32 or samplerate <= 0:
        return FrameFeatures(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    samples = samples - float(samples.mean())
    rms = float(np.sqrt(np.mean(samples * samples)))
    signs = np.sign(samples)
    signs[signs == 0] = 1.0
    zcr = float(np.mean(np.abs(np.diff(signs)) > 0)) if samples.size > 1 else 0.0

    power = np.abs(np.fft.rfft(samples * np.hamming(samples.size))) ** 2
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / float(samplerate))
    total = float(power.sum()) + 1e-12
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (float(np.mean(power)) + 1e-12))
    centroid = float(np.sum(freqs * power) / total)
    peak_i = int(np.argmax(power))
    dominant = float(freqs[peak_i])
    lobe_band = (freqs >= dominant - 80.0) & (freqs <= dominant + 80.0)
    lobe = float(power[lobe_band].sum() / total)
    beep_band = (freqs >= beep_low_hz) & (freqs <= beep_high_hz)
    band_ratio = float(power[beep_band].sum() / total)

    corr = np.correlate(samples, samples, mode="full")
    corr = corr[len(corr) // 2 :]
    corr = corr / (float(corr[0]) + 1e-12)
    low = max(1, int(samplerate / 420))
    high = min(len(corr) - 1, int(samplerate / 65))
    harmonicity = float(max(0.0, corr[low:high].max())) if high > low else 0.0
    return FrameFeatures(rms, zcr, flatness, centroid, harmonicity, dominant, band_ratio, lobe)


def is_tone_frame(
    features: FrameFeatures,
    *,
    beep_low_hz: float = 650.0,
    beep_high_hz: float = 1750.0,
    band_ratio_min: float = 0.45,
) -> bool:
    """A voicemail beep is one stable partial, not a vowel with many harmonics."""
    return bool(
        beep_low_hz <= features.dominant_hz <= beep_high_hz
        and features.lobe >= 0.70
        and features.harmonicity >= 0.45
        and features.flatness < 0.12
        and features.band_ratio >= band_ratio_min
        and features.rms >= 0.008
    )


def frame_kind(frame: np.ndarray, samplerate: int, *, speech_rms: float = 0.01) -> str:
    """Return silence, noise, speech, or tone."""
    features = analyze_frame(frame, samplerate)
    if features.rms < speech_rms * 0.45:
        return "silence"
    if is_tone_frame(features):
        return "tone"
    voiced = features.harmonicity >= 0.32 and features.flatness < 0.30 and features.lobe < 0.70
    voiced = voiced or (
        features.harmonicity >= 0.28
        and features.flatness < 0.38
        and 90.0 <= features.centroid_hz <= 3800.0
        and features.lobe < 0.70
    )
    if voiced:
        return "speech"
    if features.rms >= speech_rms:
        return "noise"
    return "silence"


def is_speech_frame(frame: np.ndarray, samplerate: int, *, speech_rms: float = 0.01) -> bool:
    return frame_kind(frame, samplerate, speech_rms=speech_rms) == "speech"


def _frame_kinds(
    audio: np.ndarray,
    samplerate: int,
    *,
    speech_rms: float = 0.01,
) -> list[str]:
    frame = max(1, int(samplerate * 0.03))
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    kinds: list[str] = []
    for start in range(0, max(0, samples.size - frame + 1), frame):
        kinds.append(frame_kind(samples[start : start + frame], samplerate, speech_rms=speech_rms))
    return kinds


def should_transcribe(audio: np.ndarray, samplerate: int, *, speech_rms: float = 0.01) -> bool:
    """False when the clip is noise rather than speech or silence.

    Silence still returns True. Callers and tests use a quiet buffer as a
    stand-in segment; noise is what makes Whisper invent a transcript.
    """
    kinds = _frame_kinds(audio, samplerate, speech_rms=speech_rms)
    if not kinds:
        return True
    speech = kinds.count("speech") / len(kinds)
    noise = kinds.count("noise") / len(kinds)
    if speech >= 0.18 and speech >= noise:
        return True
    if noise >= 0.45 and speech < 0.15:
        return False
    return True


def is_speech_audio(audio: np.ndarray, samplerate: int, *, speech_rms: float = 0.01) -> bool:
    """True when the clip is mostly harmonic speech rather than noise or a tone."""
    kinds = _frame_kinds(audio, samplerate, speech_rms=speech_rms)
    if not kinds:
        return False
    speech = kinds.count("speech")
    noise = kinds.count("noise")
    return speech >= 2 and speech > noise and (speech / len(kinds)) >= 0.18


def flat_synthetic_pitch(audio: np.ndarray, samplerate: int) -> bool:
    """Formant-poor TTS such as espeak holds a nearly fixed pitch.

    Live read speech moves. The gate stays tight so a calm person is not
    marked as a machine; the eval set is what justifies the cutoff.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(1, int(samplerate * 0.04))
    hop = max(1, int(samplerate * 0.015))
    pitches: list[float] = []
    for start in range(0, max(0, samples.size - frame + 1), hop):
        chunk = samples[start : start + frame]
        if frame_kind(chunk, samplerate) != "speech":
            continue
        features = analyze_frame(chunk, samplerate)
        if features.harmonicity < 0.45 or features.dominant_hz <= 0:
            continue
        # Prefer the autocorrelation pitch already implied by harmonicity.
        # dominant_hz on a vowel is often a harmonic, so estimate F0 directly.
        pitched = _frame_f0(chunk, samplerate)
        if pitched > 0:
            pitches.append(pitched)
    if len(pitches) < int(1.2 / 0.015):
        return False
    values = np.asarray(pitches, dtype=np.float64)
    spread = float(np.percentile(values, 90) - np.percentile(values, 10))
    return float(values.std()) <= 12.0 and spread <= 32.0


def _frame_f0(frame: np.ndarray, samplerate: int) -> float:
    samples = np.asarray(frame, dtype=np.float64).reshape(-1)
    samples = samples - float(samples.mean())
    if samples.size < 32:
        return 0.0
    corr = np.correlate(samples, samples, mode="full")
    corr = corr[len(corr) // 2 :]
    corr = corr / (float(corr[0]) + 1e-12)
    low = max(1, int(samplerate / 420))
    high = min(len(corr) - 1, int(samplerate / 65))
    if high <= low:
        return 0.0
    region = corr[low:high]
    if float(region.max()) < 0.45:
        return 0.0
    lag = low + int(np.argmax(region))
    return float(samplerate / lag) if lag else 0.0


def classify_transcript(text: str, *, speech_duration_seconds: float = 1.0) -> str:
    """Label words already transcribed. Empty text is silence, not a person."""
    from src.human_detector import HumanDetector
    from src.voicemail_detector import VoicemailDetector

    spoken = (text or "").strip()
    if not spoken:
        return "silence"
    if VoicemailDetector.keyword_count(spoken) > 0:
        return "machine"
    if VoicemailDetector.is_screening_transcript(spoken):
        return "human"
    human = HumanDetector().classify(
        transcript=spoken,
        speech_duration_seconds=speech_duration_seconds,
        answer_elapsed_seconds=1.0,
        has_speech_like=True,
    )
    if human.detected:
        return "human"
    return "uncertain"


def decide_far_end(
    audio: np.ndarray,
    samplerate: int = 16000,
    transcript: str = "",
    *,
    speech_rms: float = 0.01,
) -> FarEndDecision:
    """Decide human, machine, silence, or noise for one far-end clip.

    A mailbox phrase or a real beep wins over a leading "hello". Screening
    and a live reply win over acoustic guesses. Noise is not speech.
    """
    from src.human_detector import HumanDetector
    from src.voicemail_detector import VoicemailDetector

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    kinds = _frame_kinds(samples, samplerate, speech_rms=speech_rms)
    total = max(1, len(kinds))
    speech_ratio = kinds.count("speech") / total
    noise_ratio = kinds.count("noise") / total
    silence_ratio = kinds.count("silence") / total
    tone_run = 0
    longest_tone = 0
    for kind in kinds:
        if kind == "tone":
            tone_run += 1
            longest_tone = max(longest_tone, tone_run)
        else:
            tone_run = 0
    tone_seconds = longest_tone * 0.03
    spoken = (transcript or "").strip()

    if VoicemailDetector.keyword_count(spoken) > 0:
        return FarEndDecision("machine", 0.93, ["mailbox or ivr phrase"])
    if tone_seconds >= 0.30:
        return FarEndDecision("machine", 0.95, ["beep tone"])
    if VoicemailDetector.is_screening_transcript(spoken):
        return FarEndDecision("human", 0.84, ["call screening"])

    duration = samples.size / float(samplerate or 1)
    human = HumanDetector().classify(
        transcript=spoken,
        speech_duration_seconds=duration,
        answer_elapsed_seconds=min(duration, 8.0),
        has_speech_like=speech_ratio >= 0.18,
    )
    if spoken and human.detected:
        return FarEndDecision("human", human.confidence, human.reasons or ["human reply"])

    if speech_ratio >= 0.25 and flat_synthetic_pitch(samples, samplerate):
        return FarEndDecision("machine", 0.78, ["flat synthetic pitch"])

    if speech_ratio >= 0.22 and speech_ratio >= noise_ratio:
        confidence = min(0.9, 0.55 + speech_ratio)
        return FarEndDecision("human", confidence, ["speech-like audio"])
    if noise_ratio >= 0.30 and speech_ratio < 0.15 and tone_seconds < 0.20:
        return FarEndDecision("noise", min(0.9, 0.55 + noise_ratio), ["inharmonic noise"])
    if silence_ratio >= 0.75 and speech_ratio < 0.08 and noise_ratio < 0.15:
        return FarEndDecision("silence", min(0.95, 0.6 + silence_ratio), ["near silence"])
    if noise_ratio > speech_ratio and noise_ratio >= 0.2:
        return FarEndDecision("noise", 0.6, ["noise dominates"])
    if silence_ratio >= 0.5:
        return FarEndDecision("silence", 0.6, ["mostly silence"])
    return FarEndDecision("silence", 0.5, ["no clear speech"])
