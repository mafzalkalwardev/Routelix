"""Far-end scene labels: speech, noise, silence, and a real beep."""

import numpy as np

from src.audio_scene import classify_transcript, decide_far_end, frame_kind, should_transcribe
from src.human_detector import HumanDetector
from src.vad import EnergyVAD, VADConfig
from src.voicemail_detector import BeepDetector, VoicemailAudioClassifier, VoicemailDetector


def _tone(freq: float, seconds: float, sr: int = 16000, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _vowel(seconds: float, f0: float = 140.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    wave = np.zeros(t.shape, dtype=np.float64)
    for harmonic in range(1, 10):
        wave += (0.22 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    return wave.astype(np.float32)


def _frames(audio: np.ndarray, sr: int = 16000):
    frame = int(sr * 0.03)
    for start in range(0, len(audio) - frame, frame):
        yield audio[start : start + frame]


def test_beep_detects_a_pure_tone_and_ignores_a_vowel():
    beep = _tone(1000, 0.6)
    vowel = _vowel(1.0)
    # Harmonics stacked inside the beep band used to look like a tone.
    t = np.arange(int(16000 * 0.8)) / 16000
    stacked = (
        0.4 * np.sin(2 * np.pi * 1000 * t)
        + 0.25 * np.sin(2 * np.pi * 500 * t)
        + 0.15 * np.sin(2 * np.pi * 1500 * t)
    ).astype(np.float32)

    def detected(audio: np.ndarray) -> bool:
        det = BeepDetector()
        return any(det.process_frame(frame, samplerate=16000) for frame in _frames(audio))

    assert detected(beep) is True
    assert detected(vowel) is False
    assert detected(stacked) is False


def test_scene_labels_silence_noise_speech_and_beep():
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    noise = np.random.default_rng(1).normal(0, 0.08, sr * 2).astype(np.float32)
    speech = _vowel(1.2)
    beep = np.concatenate([np.zeros(int(sr * 0.2), dtype=np.float32), _tone(980, 0.5)])

    assert decide_far_end(silence, sr).label == "silence"
    assert decide_far_end(noise, sr).label == "noise"
    assert decide_far_end(speech, sr).label == "human"
    assert decide_far_end(beep, sr).label == "machine"
    assert frame_kind(np.zeros(480, dtype=np.float32), sr) == "silence"


def test_steady_noise_and_steady_vowel_are_not_a_mailbox_greeting():
    clf = VoicemailAudioClassifier()
    noise = np.random.default_rng(2).normal(0, 0.08, 16000 * 3).astype(np.float32)
    for frame in _frames(noise):
        clf.process_frame(frame, samplerate=16000)
    assert clf.machine_pattern_detected is False

    clf.reset()
    vowel = _vowel(4.0)
    labels = [clf.process_frame(frame, samplerate=16000) for frame in _frames(vowel)]
    assert clf.machine_pattern_detected is False
    assert "voicemail_greeting" not in labels


def test_ivr_press_one_is_machine_and_okay_is_human():
    assert VoicemailDetector.keyword_count("For English, press one.") >= 1
    assert classify_transcript("For English, press one.") == "machine"
    assert classify_transcript("Okay, I can hear you.") == "human"
    assert classify_transcript("Not interested, thanks.") == "human"
    assert classify_transcript("Please state your name and the reason for your call.") == "human"
    assert classify_transcript("") == "silence"
    # A mailbox phrase still beats a leading hello.
    assert classify_transcript("Hi, you've reached Sam. Please leave a message after the beep.") == "machine"


def test_human_detector_accepts_a_normal_length_reply():
    heard = HumanDetector().classify(
        transcript="Okay, I can hear you.",
        speech_duration_seconds=3.2,
        answer_elapsed_seconds=2.0,
        has_speech_like=True,
    )
    assert heard.detected is True
    assert heard.confidence >= 0.70


def test_noise_is_not_sent_to_stt_and_silence_still_can_be():
    sr = 16000
    noise = np.random.default_rng(4).normal(0, 0.08, sr * 2).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    speech = _vowel(0.8)
    assert should_transcribe(noise, sr) is False
    assert should_transcribe(silence, sr) is True
    assert should_transcribe(speech, sr) is True


def test_vad_speech_gate_rejects_noise_and_keeps_a_vowel():
    cfg = VADConfig(
        speech_threshold=0.01,
        speech_trigger_frames=2,
        silence_trigger_frames=5,
        pre_speech_pad_frames=0,
        speech_like_gate=True,
    )
    vad = EnergyVAD(cfg)
    noise = np.random.default_rng(3).normal(0, 0.2, 480 * 20).astype(np.float32)
    segment = None
    frame = 480
    for start in range(0, len(noise) - frame, frame):
        segment = vad.process_frame(noise[start : start + frame]) or segment
    for _ in range(8):
        flushed = vad.process_frame(np.zeros(frame, dtype=np.float32))
        if flushed is not None:
            segment = flushed
    assert segment is None

    vad.reset()
    vowel = _vowel(0.8)
    heard = None
    for start in range(0, len(vowel) - frame, frame):
        heard = vad.process_frame(vowel[start : start + frame]) or heard
    for _ in range(8):
        flushed = vad.process_frame(np.zeros(frame, dtype=np.float32))
        if flushed is not None:
            heard = flushed
    assert heard is not None
    assert heard.size > 0
