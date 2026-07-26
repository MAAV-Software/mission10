"""Speech to text behind a narrow interface: PCM in, transcript and per-word
confidences out.

The boundary keeps a change of engine cheap. All the code after Transcription is
pure (grammar.py) and imports no engine. Thus a different engine touches only this
file. vosk operates on the CM5: it publishes a manylinux2014_aarch64 wheel.

The audio contract is 16 kHz mono signed 16-bit little-endian PCM, with no
container. The browser makes exactly this format (static/pcm-worklet.js). Thus no
code in this file decodes the audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Protocol

SAMPLE_RATE_HZ = 16000


@dataclass(frozen=True)
class Transcription:
    text: str
    word_confidences: list[float] = field(default_factory=list)


class SpeechEngine(Protocol):
    def transcribe(self, pcm: bytes) -> Transcription: ...


class VoskEngine:
    """vosk with a closed grammar. This class makes one recognizer for each
    utterance. It loads the model only once, because the load is the slow part."""

    def __init__(self, model_path: Path, vocabulary: tuple[str, ...]) -> None:
        from vosk import Model  # here, thus the module loads without vosk

        if not model_path.exists():
            raise FileNotFoundError(
                f"vosk model not found at {model_path}. "
                "Run mission10/scripts/fetch_speech_models.sh."
            )
        self._model = Model(str(model_path))
        # The "[unk]" token is necessary. Without it, the recognizer matches all
        # audio to the nearest sequence of words in the vocabulary. Thus a cough or
        # a person nearby becomes a command. With it, unknown audio becomes the
        # token, and grammar.py rejects the utterance.
        self._grammar = json.dumps(list(vocabulary) + ["[unk]"])

    def transcribe(self, pcm: bytes) -> Transcription:
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self._model, float(SAMPLE_RATE_HZ), self._grammar)
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(pcm)
        result = json.loads(recognizer.FinalResult() or "{}")
        words = result.get("result", [])
        return Transcription(
            text=(result.get("text") or "").strip(),
            word_confidences=[float(w["conf"]) for w in words if "conf" in w],
        )
