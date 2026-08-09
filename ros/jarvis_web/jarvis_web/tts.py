"""Piper text to speech. Text in, WAV bytes out.

This module makes the audio on demand and keeps no cache. The set of phrases is
small, thus a cache saves no measurable time. A cache also fixes the set of words
that Jarvis can say at build time. That becomes wrong as soon as a response
contains a variable.
"""

from __future__ import annotations

import io
from pathlib import Path
import wave


class PiperVoice:
    def __init__(self, model_path: Path) -> None:
        from piper import PiperVoice as _PiperVoice

        if not model_path.exists():
            raise FileNotFoundError(
                f"piper voice not found at {model_path}. "
                "Run mission10/scripts/fetch_speech_models.sh."
            )
        self._voice = _PiperVoice.load(str(model_path))

    def say(self, text: str) -> bytes:
        """Make a complete WAV file in memory. The caller sends the bytes to the
        browser, and the browser plays them. This method writes nothing to disk."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            self._voice.synthesize_wav(text, wav)
        return buffer.getvalue()
