"""English and Spanish voices only.

French and the other Kokoro languages are not offered. Edge, Piper, and
Kokoro each have a short list the operator can select. Piper Spanish files
are downloaded on the server; they are not stored in git.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceChoice:
    engine: str
    voice_id: str
    language: str
    label: str


# Edge neural voices. These need network at synthesis time.
ENGLISH_EDGE = (
    VoiceChoice("edge", "en-US-GuyNeural", "en", "English, US, Guy"),
    VoiceChoice("edge", "en-US-JennyNeural", "en", "English, US, Jenny"),
    VoiceChoice("edge", "en-GB-SoniaNeural", "en", "English, UK, Sonia"),
    VoiceChoice("edge", "en-GB-RyanNeural", "en", "English, UK, Ryan"),
)
SPANISH_EDGE = (
    VoiceChoice("edge", "es-MX-DaliaNeural", "es", "Spanish, Mexico, Dalia"),
    VoiceChoice("edge", "es-MX-JorgeNeural", "es", "Spanish, Mexico, Jorge"),
    VoiceChoice("edge", "es-ES-ElviraNeural", "es", "Spanish, Spain, Elvira"),
    VoiceChoice("edge", "es-ES-AlvaroNeural", "es", "Spanish, Spain, Alvaro"),
)

# Piper ONNX names. install.sh can fetch these from rhasspy/piper-voices.
ENGLISH_PIPER = (
    VoiceChoice("piper", "en_US-lessac-medium", "en", "English, US, Lessac"),
    VoiceChoice("piper", "en_US-amy-medium", "en", "English, US, Amy"),
    VoiceChoice("piper", "en_GB-alan-medium", "en", "English, UK, Alan"),
    VoiceChoice("piper", "en_GB-alba-medium", "en", "English, UK, Alba"),
)
SPANISH_PIPER = (
    VoiceChoice("piper", "es_ES-davefx-medium", "es", "Spanish, Spain, Davefx"),
    VoiceChoice("piper", "es_ES-sharvard-medium", "es", "Spanish, Spain, Sharvard"),
    VoiceChoice("piper", "es_MX-ald-medium", "es", "Spanish, Mexico, Ald"),
)

# Kokoro ids inside voices-v1.0.bin. Spanish is two voices in that file.
# French (ff_siwis) and the other language packs are intentionally absent.
ENGLISH_KOKORO = (
    VoiceChoice("kokoro", "af_heart", "en", "English, US, Heart"),
    VoiceChoice("kokoro", "af_bella", "en", "English, US, Bella"),
    VoiceChoice("kokoro", "am_adam", "en", "English, US, Adam"),
    VoiceChoice("kokoro", "am_michael", "en", "English, US, Michael"),
    VoiceChoice("kokoro", "bf_emma", "en", "English, UK, Emma"),
    VoiceChoice("kokoro", "bm_george", "en", "English, UK, George"),
)
SPANISH_KOKORO = (
    VoiceChoice("kokoro", "ef_dora", "es", "Spanish, Dora"),
    VoiceChoice("kokoro", "em_alex", "es", "Spanish, Alex"),
)

CATALOG = ENGLISH_EDGE + SPANISH_EDGE + ENGLISH_PIPER + SPANISH_PIPER + ENGLISH_KOKORO + SPANISH_KOKORO


def voices_for(language: str, engine: str | None = None) -> tuple[VoiceChoice, ...]:
    lang = (language or "en").strip().lower()[:2]
    if lang not in {"en", "es"}:
        raise ValueError("Routelix voices are English and Spanish only")
    chosen = tuple(item for item in CATALOG if item.language == lang)
    if engine:
        key = engine.strip().lower()
        chosen = tuple(item for item in chosen if item.engine == key)
    return chosen


def language_of_voice(voice_id: str) -> str:
    for item in CATALOG:
        if item.voice_id == voice_id:
            return item.language
    lowered = (voice_id or "").lower()
    if lowered.startswith("es_") or lowered.startswith("es-"):
        return "es"
    return "en"
