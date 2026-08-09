"""Speech in, intent out, through the real engines.

piper says a phrase, vosk transcribes it, and grammar.py classifies the result.
This is the only test that shows what the recognizer does with sound. The tests
with fakes cannot: they supply the transcript that they want.

The negation cases below are the reason that an unknown token rejects the full
utterance. This test found that behaviour.

The models are large and the repository does not hold them. Without them these
tests skip. Run mission10/scripts/fetch_speech_models.sh to get them.

piper gives 22.05 kHz audio and vosk needs 16 kHz, thus this file resamples. The
webapp never does: the browser gives 16 kHz directly (static/pcm-worklet.js).
"""

import array
import io
import os
from pathlib import Path
import unittest
import wave

from jarvis_web.grammar import ABORT, COME_HOME, EXECUTE, LAUNCH, Accepted, recognize

TARGET_RATE = 16000
THRESHOLD = 0.5

MODELS = Path(
    os.environ.get(
        "JARVIS_MODELS_DIR",
        Path(__file__).resolve().parents[3] / "models" / "speech" / "assets",
    )
)
VOSK_MODEL = MODELS / "vosk-model-small-en-us-0.15"
PIPER_VOICE = MODELS / "en_US-lessac-medium.onnx"


def models_are_present():
    return VOSK_MODEL.is_dir() and PIPER_VOICE.is_file()


def to_16k_mono(wav_bytes):
    """Take one channel and resample to 16 kHz with linear interpolation."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as source:
        rate = source.getframerate()
        width = source.getsampwidth()
        channels = source.getnchannels()
        frames = source.readframes(source.getnframes())

    assert width == 2, f"expected 16-bit samples, got {width * 8}-bit"
    samples = memoryview(frames).cast("h")
    if channels > 1:
        samples = samples[::channels]
    if rate == TARGET_RATE:
        return bytes(samples)

    ratio = rate / TARGET_RATE
    out = array.array("h")
    for i in range(int(len(samples) / ratio)):
        position = i * ratio
        low = int(position)
        first = samples[low]
        second = samples[low + 1] if low + 1 < len(samples) else first
        out.append(int(first + (second - first) * (position - low)))
    return out.tobytes()


@unittest.skipUnless(models_are_present(), f"speech models not found in {MODELS}")
class TestRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jarvis_web.grammar import VOCABULARY
        from jarvis_web.stt import VoskEngine
        from jarvis_web.tts import PiperVoice

        cls.voice = PiperVoice(PIPER_VOICE)
        cls.engine = VoskEngine(VOSK_MODEL, VOCABULARY)

    def outcome_of(self, said):
        heard = self.engine.transcribe(to_16k_mono(self.voice.say(said)))
        return heard, recognize(heard.text, heard.word_confidences, THRESHOLD)

    def test_each_spoken_phrase_gives_its_intent(self):
        cases = [
            ("launch", LAUNCH),
            ("jarvis launch", LAUNCH),
            ("execute", EXECUTE),
            ("jarvis execute", EXECUTE),
            ("come home", COME_HOME),
            ("jarvis come home", COME_HOME),
            ("abort", ABORT),
            ("jarvis abort", ABORT),
        ]
        for said, intent in cases:
            with self.subTest(said=said):
                heard, result = self.outcome_of(said)
                self.assertIsInstance(
                    result.outcome, Accepted, f"heard {heard.text!r}"
                )
                self.assertIs(result.outcome.intent, intent, f"heard {heard.text!r}")

    def test_speech_outside_the_vocabulary_commands_nothing(self):
        for said in (
            "the weather is nice today",
            "how much battery is left",
            "put the kettle on",
        ):
            with self.subTest(said=said):
                heard, result = self.outcome_of(said)
                self.assertNotIsInstance(
                    result.outcome, Accepted, f"heard {heard.text!r}"
                )

    def test_a_spoken_negation_never_launches(self):
        # The safety property, against sound rather than a supplied transcript.
        # Each of these contains the word "launch", thus a recognizer that drops
        # the words it does not know would start the mission.
        for said in (
            "don't launch",
            "do not launch",
            "no launch",
            "cancel launch",
            "stop the launch",
        ):
            with self.subTest(said=said):
                heard, result = self.outcome_of(said)
                self.assertNotIsInstance(
                    result.outcome, Accepted, f"heard {heard.text!r}"
                )


if __name__ == "__main__":
    unittest.main()
