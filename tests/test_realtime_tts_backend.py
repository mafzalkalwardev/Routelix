import wave

import numpy as np
import pytest

from src import realtime_tts


def test_unknown_tts_engine_is_rejected(monkeypatch):
    monkeypatch.setattr(realtime_tts, "_edge_available", lambda: True)

    with pytest.raises(ValueError, match="Unsupported TTS_ENGINE"):
        realtime_tts.RealtimeTTS(device_index=0, engine="unknown")


def test_piper_wav_is_decoded_to_float32(monkeypatch, tmp_path):
    class FakeVoice:
        def synthesize_wav(self, text, wav_file):
            assert text == "Hello"
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(np.zeros(2205, dtype=np.int16).tobytes())

    data, rate = realtime_tts._piper_synthesize("Hello", FakeVoice())

    assert data.dtype == np.float32
    assert data.shape == (2205,)
    assert rate == 22050
