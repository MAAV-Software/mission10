"""Tests for the routes, with a fake engine, a fake voice and a fake ROS rim.

These tests hold the contract of the /utterance route: an accepted phrase
publishes once, and any rejection publishes nothing. test_publish.py holds the
behaviour against a real subscriber.
"""

from pathlib import Path
import tempfile
import unittest

from jarvis_web.app import create_app, describe, newest_result
from jarvis_web.grammar import (
    LAUNCH,
    Accepted,
    LowConfidence,
    NoMatch,
    OutOfVocabulary,
    Rejected,
)
from jarvis_web.stt import Transcription

CONFIDENT = [0.9, 0.9]


class FakeEngine:
    def __init__(self, text, confidences=CONFIDENT):
        self.transcription = Transcription(text, confidences)

    def transcribe(self, pcm):
        return self.transcription


class FakeVoice:
    def say(self, text):
        return b"RIFF" + text.encode()


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, line):
        self.lines.append(line)

    warn = info


class FakeGates:
    def __init__(self):
        self.published = []
        self.logger = FakeLogger()

    def publish(self, intent):
        self.published.append(intent)


def build(text, confidences=CONFIDENT, results_dir=Path("/nonexistent"), threshold=0.5):
    gates = FakeGates()
    app = create_app(FakeEngine(text, confidences), FakeVoice(), gates, results_dir, threshold)
    app.config.update(TESTING=True)
    return app.test_client(), gates


class TestUtterance(unittest.TestCase):
    def test_a_known_phrase_publishes_one_message(self):
        client, gates = build("jarvis launch")
        body = client.post("/utterance", data=b"\x00\x01").get_json()

        self.assertEqual(gates.published, [LAUNCH])
        self.assertTrue(body["accepted"])
        self.assertEqual(body["intent"], "launch")
        self.assertEqual(body["response"], "LAUNCHING.")

    def test_a_rejection_publishes_nothing(self):
        for text, confidences, cause in (
            ("banana", CONFIDENT, "no_match"),
            ("launch", [0.2], "low_confidence"),
            ("[unk] launch", CONFIDENT, "out_of_vocabulary"),
        ):
            client, gates = build(text, confidences)
            body = client.post("/utterance", data=b"\x00").get_json()

            self.assertEqual(gates.published, [], text)
            self.assertFalse(body["accepted"], text)
            self.assertEqual(body["reason"], cause, text)
            self.assertEqual(body["response"], "SAY AGAIN.", text)


class TestDescribe(unittest.TestCase):
    def test_each_outcome_gives_a_name_and_a_cause(self):
        # describe() ends with assert_never. A new outcome without a branch fails
        # here.
        cases = [
            (Accepted(LAUNCH), ("launch", "ok")),
            (Rejected(OutOfVocabulary()), (None, "out_of_vocabulary")),
            (Rejected(NoMatch()), (None, "no_match")),
            (Rejected(LowConfidence(LAUNCH)), ("launch", "low_confidence")),
        ]
        for outcome, expected in cases:
            self.assertEqual(describe(outcome), expected, outcome)


class TestResult(unittest.TestCase):
    def test_a_missing_directory_is_not_an_error_page(self):
        client, _ = build("launch", results_dir=Path("/nonexistent"))
        response = client.get("/result")
        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.get_json()["available"])

    def test_the_route_gives_the_most_recent_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os

            results = Path(tmp)
            (results / "old.png").write_bytes(b"\x89PNGold")
            (results / "new.png").write_bytes(b"\x89PNGnew")
            (results / "dump.json").write_text("{}")
            os.utime(results / "old.png", (1, 1))

            self.assertEqual(newest_result(results), results / "new.png")

            client, _ = build("launch", results_dir=results)
            self.assertEqual(client.get("/result").data, b"\x89PNGnew")


class TestPage(unittest.TestCase):
    def test_the_browser_can_get_the_page_and_each_static_file(self):
        # This fails if the data_files glob in setup.py stops installing static/.
        client, _ = build("launch")
        self.assertIn(b"HOLD TO TALK", client.get("/").data)
        for asset in ("app.js", "pcm-worklet.js", "style.css"):
            self.assertEqual(client.get(f"/static/{asset}").status_code, 200, asset)


if __name__ == "__main__":
    unittest.main()
