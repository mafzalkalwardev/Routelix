"""Clause breaths for Piper and SSML for Edge, without a network call."""

import numpy as np

from src.realtime_tts import _render_piper_phrase
from src.speech_style import split_spoken_clauses, to_edge_ssml


def test_split_keeps_decimals_and_breathes_on_commas():
    parts = split_spoken_clauses("Hi, the rate is 3.5 dollars. Are you interested?")
    texts = [text for text, _pause in parts]
    assert texts[0].startswith("Hi")
    assert any("3.5" in text for text in texts)
    assert texts[-1].endswith("?")
    assert parts[-1][1] == 0.0
    assert any(pause >= 0.1 for _text, pause in parts[:-1])


def test_edge_ssml_adds_a_clause_break():
    markup = to_edge_ssml("Hi there. Are you still dispatching?")
    assert markup.startswith("<speak")
    assert "<break time=" in markup
    assert "Hi," in markup
    assert "dispatching?" in markup


def test_piper_render_inserts_a_real_gap_between_clauses(monkeypatch):
    def fake_synthesize(text, voice, **kwargs):
        del voice, kwargs
        assert text.strip()
        return np.full(int(22050 * 0.2), 0.2, dtype=np.float32), 22050

    monkeypatch.setattr("src.realtime_tts._piper_synthesize", fake_synthesize)
    audio, rate = _render_piper_phrase("Hi, this is Alex.", voice=object())
    assert rate == 22050
    quiet = np.abs(audio) < 0.02
    # A run of inserted silence between the two clauses.
    run = 0
    longest = 0
    for sample_is_quiet in quiet:
        if sample_is_quiet:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    assert longest >= int(rate * 0.08)
