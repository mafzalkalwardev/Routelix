"""Human pickup scoring copied from the Auto Dialer detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
import re


# Live pickup lines. Mailbox and IVR phrases are scored separately and win
# when both could match ("hi, you've reached" is a mailbox, not a hello).
HUMAN_KEYWORDS = (
    "hello",
    "hi",
    "hey",
    "yes",
    "yeah",
    "yep",
    "yup",
    "okay",
    "ok",
    "sure",
    "alright",
    "all right",
    "speaking",
    "this is",
    "who is this",
    "who's this",
    "whos this",
    "who is it",
    "who are you",
    "what's this",
    "whats this",
    "what is this",
    "what do you want",
    "can i help",
    "how can i help",
    "may i help",
    "hold on",
    "hang on",
    "one second",
    "one moment",
    "go ahead",
    "i'm listening",
    "im listening",
    "not interested",
    "no thanks",
    "no thank you",
    "nope",
    "nah",
    "pardon",
    "excuse me",
    "say that again",
    "come again",
    "what",
    "huh",
    "assalamualaikum",
    "salam alaikum",
)


@dataclass(frozen=True)
class HumanDetection:
    detected: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)


class HumanDetector:
    def classify(
        self,
        *,
        transcript: str = "",
        speech_duration_seconds: float = 0.0,
        answer_elapsed_seconds: float = 0.0,
        has_speech_like: bool = False,
    ) -> HumanDetection:
        reasons: list[str] = []
        confidence = 0.0

        if self.has_human_keyword(transcript):
            confidence += 0.58
            reasons.append("human greeting keyword")
        # A live answer is often one short word, but it is also often a full
        # sentence. The old 2.5 s cap dropped "okay, I can hear you" once the
        # utterance ran long, and a keyword by itself never cleared 0.70.
        if 0.18 <= speech_duration_seconds <= 8.0:
            confidence += 0.22
            reasons.append("conversational length")
        elif speech_duration_seconds > 8.0 and self.has_human_keyword(transcript):
            confidence += 0.15
            reasons.append("extended live turn")
        if has_speech_like and answer_elapsed_seconds <= 12.0:
            confidence += 0.18
            reasons.append("speech during human-first window")
        if self.has_human_keyword(transcript) and speech_duration_seconds <= 0.0:
            confidence += 0.15
            reasons.append("short reply text")

        confidence = min(1.0, confidence)
        return HumanDetection(
            detected=confidence >= 0.70,
            confidence=confidence,
            reasons=reasons,
        )

    @staticmethod
    def has_human_keyword(text: str) -> bool:
        if not text:
            return False
        return any(re.search(rf"\b{re.escape(word)}\b", text, re.I) for word in HUMAN_KEYWORDS)
