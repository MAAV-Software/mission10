"""Tests for the pure layer. No speech engine and no ROS.

These tests hold only the properties that a change to the phrase table can break.
The behaviour against real audio is in test_round_trip.py, and the behaviour
against a real subscriber is in test_publish.py. Those two matter more.
"""

import unittest

from jarvis_web.grammar import (
    CALLSIGN,
    INTENTS,
    LAUNCH,
    PHRASES,
    REJECT_RESPONSE,
    VOCABULARY,
    Accepted,
    LowConfidence,
    Match,
    NoMatch,
    OutOfVocabulary,
    Rejected,
    recognize,
    utterance_confidence,
)

CONFIDENT = [1.0, 1.0, 1.0]


class OutcomeTest(unittest.TestCase):
    def assert_accepted(self, result, intent, note=""):
        self.assertIsInstance(result.outcome, Accepted, note)
        self.assertIs(result.outcome.intent, intent, note)
        self.assertEqual(result.response, intent.response, note)

    def assert_rejected(self, result, rejection_type, note=""):
        self.assertIsInstance(result.outcome, Rejected, note)
        self.assertIsInstance(result.outcome.rejection, rejection_type, note)
        self.assertEqual(result.response, REJECT_RESPONSE, note)


class TestTheTable(OutcomeTest):
    """Properties of the phrase table itself."""

    def test_each_phrase_matches_with_and_without_the_callsign(self):
        for phrase, intent in PHRASES.items():
            for said in (phrase, f"{CALLSIGN} {phrase}"):
                self.assert_accepted(recognize(said, CONFIDENT, 0.5), intent, said)

    def test_the_topics_are_all_different(self):
        # A copy and paste error in the table sends two commands to one gate.
        topics = [intent.topic for intent in INTENTS]
        self.assertEqual(len(topics), len(set(topics)))

    def test_each_word_of_each_phrase_is_in_the_vocabulary(self):
        # A phrase with a word outside the vocabulary is unreachable: the
        # recognizer can never produce it.
        for phrase in PHRASES:
            for word in phrase.split():
                self.assertIn(word, VOCABULARY, phrase)

    def test_messy_text_still_reaches_the_phrase(self):
        # vosk gives bare lower-case words. This rule protects a change of engine.
        self.assert_accepted(recognize("  Jarvis,   LAUNCH!  ", CONFIDENT, 0.5), LAUNCH)


class TestSafety(OutcomeTest):
    """The rules that stop a gate from opening when it must not."""

    def test_the_word_land_publishes_nothing(self):
        # "land" is a word of habit. It is not in the vocabulary, thus it gives
        # SAY AGAIN and not a descent.
        self.assert_rejected(recognize("land", CONFIDENT, 0.5), NoMatch)

    def test_negation_does_not_reach_the_bare_verb(self):
        # vosk made these transcripts for the spoken negations, against this
        # vocabulary. Each one has the unknown token at a confidence of 1.00. If
        # the code ignored the token, all five would publish /start_mission.
        for said in (
            "don't launch",
            "do not launch",
            "no launch",
            "cancel launch",
            "stop the launch",
        ):
            self.assert_rejected(
                recognize("[unk] launch", CONFIDENT, 0.5), OutOfVocabulary, said
            )

    def test_the_unknown_token_rejects_at_any_position(self):
        for transcript in ("[unk] jarvis abort", "abort [unk]", "come [unk] home"):
            self.assert_rejected(
                recognize(transcript, CONFIDENT, 0.5), OutOfVocabulary, transcript
            )

    def test_an_average_would_accept_where_the_minimum_rejects(self):
        # 0.95 and 0.20 give an average of 0.575, which is more than the
        # threshold. The weak word must still reject the phrase.
        confidences = [0.95, 0.20]
        self.assertGreater(sum(confidences) / len(confidences), 0.5)
        self.assert_rejected(recognize("come home", confidences, 0.5), LowConfidence)

    def test_no_confidences_gives_zero_and_not_one(self):
        # An engine that sends no per-word data must give a rejection, not a pass.
        self.assertEqual(utterance_confidence([]), 0.0)


class TestResponseIsTotal(unittest.TestCase):
    def test_every_outcome_gives_a_response(self):
        # Match.response ends with assert_never. A new outcome without a branch
        # fails here.
        outcomes = [
            Accepted(LAUNCH),
            Rejected(OutOfVocabulary()),
            Rejected(NoMatch()),
            Rejected(LowConfidence(LAUNCH)),
        ]
        for outcome in outcomes:
            response = Match("t", "t", 1.0, outcome).response
            self.assertTrue(response, outcome)


if __name__ == "__main__":
    unittest.main()
