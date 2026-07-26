"""The Flask rim and the entry point.

The browser sends one POST for each utterance. Push-to-talk shows when the speech
stops, because the operator releases the button. Thus there is no stream session,
and the server keeps no state between two requests.

The path from an intent to a gate is direct: transcribe, recognize, publish. No
code between them decides if the command is legal. The mission nodes control
their own state.
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import assert_never

from flask import Flask, jsonify, request, send_file, send_from_directory

from jarvis_web import grammar

# This module imports rclpy, vosk and piper in main(), not here. create_app()
# receives its collaborators as arguments. Thus the tests can supply fakes, and
# the Flask layer runs on a machine that has none of the three packages.

DEFAULT_MODELS_DIR = Path(
    os.environ.get("JARVIS_MODELS_DIR", "models/speech/assets")
)
DEFAULT_RESULTS_DIR = Path(os.environ.get("JARVIS_RESULTS_DIR", "/tmp/maav_results"))
DEFAULT_THRESHOLD = 0.5


def static_dir() -> Path:
    """Give the installed share directory, or the source tree if colcon did not
    install the package. The --symlink-install option makes the two the same."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("jarvis_web")) / "static"
    except Exception:
        return Path(__file__).parent / "static"


def describe(outcome: grammar.Outcome) -> tuple[str | None, str]:
    """Give the intent name and the cause, for the JSON body and for the log.

    JSON has no sum types. Thus this function makes the outcome flat at the wire
    boundary, and only at that boundary.
    """
    match outcome:
        case grammar.Accepted(intent):
            return intent.name, "ok"
        case grammar.Rejected(grammar.OutOfVocabulary()):
            return None, "out_of_vocabulary"
        case grammar.Rejected(grammar.NoMatch()):
            return None, "no_match"
        case grammar.Rejected(grammar.LowConfidence(intent)):
            # The name shows what the webapp almost heard. It is for the log. It
            # is not permission to publish.
            return intent.name, "low_confidence"
    assert_never(outcome)


def newest_result(results_dir: Path) -> Path | None:
    """Give the most recent render, by the time of the last change to the file.
    The pathfinder controls this directory. This function selects a file, and it
    never reads the contents."""
    if not results_dir.is_dir():
        return None
    renders = [p for p in results_dir.glob("*.png") if p.is_file()]
    if not renders:
        return None
    return max(renders, key=lambda p: p.stat().st_mtime)


def create_app(engine, voice, gates, results_dir: Path, threshold: float) -> Flask:
    """Build the webapp.

    The `engine` transcribes PCM (stt.SpeechEngine). The `voice` makes WAV bytes
    (tts.PiperVoice). The `gates` publish an accepted intent (node.GatePublisher).
    The caller supplies all three, thus a test can replace each one.
    """
    static = static_dir()
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(static, "index.html")

    @app.get("/static/<path:name>")
    def asset(name: str):
        return send_from_directory(static, name)

    @app.post("/utterance")
    def utterance():
        pcm = request.get_data()
        heard = engine.transcribe(pcm)
        result = grammar.recognize(heard.text, heard.word_confidences, threshold)
        intent_name, cause = describe(result.outcome)

        # Only this statement decides if the webapp publishes. It has one branch
        # for each outcome. If a new outcome appears later, `accepted` stays
        # unbound and the request fails. The webapp does not publish by accident.
        match result.outcome:
            case grammar.Accepted(intent):
                gates.publish(intent)
                accepted = True
            case grammar.Rejected():
                gates.logger.info(
                    f"rejected ({cause}): {result.transcript!r} "
                    f"conf={result.confidence:.2f}"
                )
                accepted = False

        return jsonify(
            transcript=result.transcript,
            intent=intent_name,
            accepted=accepted,
            reason=cause,
            confidence=result.confidence,
            response=result.response,
            audio=base64.b64encode(voice.say(result.response)).decode("ascii"),
        )

    @app.get("/result")
    def result():
        render = newest_result(results_dir)
        if render is None:
            return jsonify(available=False), 404
        return send_file(render, mimetype="image/png")

    return app


def main() -> None:
    from jarvis_web.node import GatePublisher
    from jarvis_web.stt import VoskEngine
    from jarvis_web.tts import PiperVoice

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--vosk-model", type=Path, default=None)
    parser.add_argument("--piper-voice", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    # A browser gives access to the microphone only in a secure context. Thus
    # these two options are necessary in the field. The package README gives the
    # mkcert procedure.
    parser.add_argument("--cert", type=Path, default=None)
    parser.add_argument("--key", type=Path, default=None)
    args = parser.parse_args()

    vosk_model = args.vosk_model or args.models_dir / "vosk-model-small-en-us-0.15"
    piper_voice = args.piper_voice or args.models_dir / "en_US-lessac-medium.onnx"

    engine = VoskEngine(vosk_model, grammar.VOCABULARY)
    voice = PiperVoice(piper_voice)
    gates = GatePublisher()
    app = create_app(engine, voice, gates, args.results_dir, args.threshold)

    ssl_context = (str(args.cert), str(args.key)) if args.cert and args.key else None
    if ssl_context is None:
        gates.logger.warn(
            "no TLS: a browser refuses access to the microphone on an insecure "
            "origin, thus voice does not operate from a phone"
        )

    try:
        app.run(host=args.host, port=args.port, ssl_context=ssl_context, threaded=True)
    finally:
        gates.shutdown()


if __name__ == "__main__":
    main()
