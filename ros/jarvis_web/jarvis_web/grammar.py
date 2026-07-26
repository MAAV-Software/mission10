"""Speech to mission gate. This module is pure. It has no speech engine, no ROS
and no input or output.

The recognizer accepts only the words in VOCABULARY. Thus a transcript is always
a sequence of these words. This module puts the transcript into a normal form and
compares the result with the phrases. If the result is not a phrase, the operator
hears SAY AGAIN and the webapp sends nothing.

This is the correct result when the recognizer hears only part of a command.
Silence is safer than a gate that the operator did not ask for.

LAUNCH and ABORT have opposite effects. Thus they start with different sounds and
they have a different number of syllables (qualifier RFD 9.6). The word "land" is
not in the vocabulary. If the operator says "land", the drones stay in the air and
the operator hears SAY AGAIN.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import assert_never

__all__ = [
    "ABORT",
    "Accepted",
    "CALLSIGN",
    "COME_HOME",
    "EXECUTE",
    "INTENTS",
    "Intent",
    "LAUNCH",
    "LowConfidence",
    "Match",
    "NoMatch",
    "Outcome",
    "OutOfVocabulary",
    "PHRASES",
    "REJECT_RESPONSE",
    "Rejected",
    "Rejection",
    "UNKNOWN_TOKEN",
    "VOCABULARY",
    "normalize",
    "recognize",
    "utterance_confidence",
]


@dataclass(frozen=True)
class Intent:
    """One command: its name, its topic and the words that Jarvis says back."""

    name: str
    topic: str
    response: str


LAUNCH = Intent("launch", "/start_mission", "LAUNCHING.")
EXECUTE = Intent("execute", "/begin_orbit", "EXECUTING.")
COME_HOME = Intent("come_home", "/end_mission", "COMING HOME.")
ABORT = Intent("abort", "/abort_mission", "ABORTING.")

INTENTS: tuple[Intent, ...] = (LAUNCH, EXECUTE, COME_HOME, ABORT)

PHRASES: dict[str, Intent] = {
    "launch": LAUNCH,
    "execute": EXECUTE,
    "come home": COME_HOME,
    "abort": ABORT,
}

REJECT_RESPONSE = "SAY AGAIN."


@dataclass(frozen=True)
class OutOfVocabulary:
    """The recognizer heard audio that is not one of the known words."""


@dataclass(frozen=True)
class NoMatch:
    """The words are all known, but together they are not a phrase."""


@dataclass(frozen=True)
class LowConfidence:
    """The words are a phrase, but one word is too weak to act on.

    This class holds the intent for the log only. It gives no permission to
    publish. Only an Accepted gives that permission.
    """

    intent: Intent


Rejection = OutOfVocabulary | NoMatch | LowConfidence


@dataclass(frozen=True)
class Accepted:
    """The transcript is a phrase and each word is strong enough."""

    intent: Intent


@dataclass(frozen=True)
class Rejected:
    """The webapp publishes nothing and the operator hears SAY AGAIN."""

    rejection: Rejection


# There are two levels because there are two different questions. The outer level
# decides if the webapp publishes. The inner level records the cause, for the log.
# An Accepted always holds an intent and a Rejected never holds one. Thus the code
# cannot make a state that publishes a gate for a command it did not recognize.
Outcome = Accepted | Rejected


# The recognizer writes this token for audio that it cannot match to a word in the
# vocabulary. If a transcript contains this token, the webapp rejects the full
# utterance.
#
# This rule is necessary for safety. Against this vocabulary, the phrases "don't
# launch", "do not launch", "no launch", "cancel launch" and "stop the launch" all
# become "[unk] launch". The test test_negation_does_not_reach_the_bare_verb holds
# these five examples. If the code ignored the token to accept more noise, all five
# phrases would start a launch.
#
# The cost of the rule is that the webapp sometimes rejects a good phrase, because
# noise before the phrase becomes a token. This cost is small: the operator says
# the command again. The other risk is four aircraft in the air after a command
# that told them not to fly.
UNKNOWN_TOKEN = "[unk]"

# The callsign. Push-to-talk does the work of a wake word. Thus the callsign is
# optional in each phrase, and the code removes it before it compares the phrase.
CALLSIGN = "jarvis"

# The closed vocabulary for the recognizer. Six words is most of the reason that
# the recognizer is accurate outdoors. Each new word makes the other words less
# accurate.
VOCABULARY: tuple[str, ...] = (
    CALLSIGN,
    "launch",
    "execute",
    "come",
    "home",
    "abort",
)

_NON_LETTER = re.compile(r"[^a-z]+")


def normalize(text: str) -> str:
    """Make the text lower case, put one space between the words, and remove a
    callsign at the start."""
    words = _NON_LETTER.sub(" ", text.strip().lower()).split()
    if words and words[0] == CALLSIGN:
        words = words[1:]
    return " ".join(words)


def utterance_confidence(word_confidences: list[float]) -> float:
    """Give the confidence of the weakest word.

    An average is not safe here. With an average, a word with a low confidence can
    pass because a different word has a high confidence. In a phrase of two words,
    one word is one half of the data.

    If there are no confidences, there is no data. The result is then 0.0, and the
    caller rejects the utterance.
    """
    if not word_confidences:
        return 0.0
    return min(word_confidences)


@dataclass(frozen=True)
class Match:
    """The result of one utterance.

    The outcome is the only field that gives permission to publish. The other
    three fields are for the log and for the operator display.
    """

    transcript: str
    normalized: str
    confidence: float
    outcome: Outcome

    @property
    def response(self) -> str:
        """The words that Jarvis says aloud."""
        match self.outcome:
            case Accepted(intent):
                return intent.response
            case Rejected():
                return REJECT_RESPONSE
        assert_never(self.outcome)


def recognize(
    transcript: str, word_confidences: list[float], threshold: float
) -> Match:
    """Turn a transcript into an intent, or into a rejection with its cause."""
    normalized = normalize(transcript)
    confidence = utterance_confidence(word_confidences)

    # The code examines the raw transcript, before it uses normalize(). normalize()
    # removes the brackets and leaves the word "unk", which then fails to match a
    # phrase only by chance.
    if UNKNOWN_TOKEN in transcript:
        return Match(transcript, normalized, confidence, Rejected(OutOfVocabulary()))

    intent = PHRASES.get(normalized)
    if intent is None:
        return Match(transcript, normalized, confidence, Rejected(NoMatch()))
    if confidence < threshold:
        return Match(
            transcript, normalized, confidence, Rejected(LowConfidence(intent))
        )
    return Match(transcript, normalized, confidence, Accepted(intent))
