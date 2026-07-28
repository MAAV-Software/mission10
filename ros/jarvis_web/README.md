# jarvis_web

The operator webapp for the master drone. Voice in, mission gates out, and the
result map at the end.

This webapp only publishes. It subscribes to nothing and it holds no mission
state. No code comes between a known phrase and the publish. The mission nodes
control their own state.

## Vocabulary

| Phrase | Topic | Response |
|---|---|---|
| `[jarvis] launch` | `/start_mission` | `LAUNCHING.` |
| `[jarvis] execute` | `/begin_orbit` | `EXECUTING.` |
| `[jarvis] come home` | `/end_mission` | `COMING HOME.` |
| `[jarvis] abort` | `/abort_mission` | `ABORTING.` |

The callsign `jarvis` is optional, because push-to-talk does the work of a wake
word. For all other speech the operator hears `SAY AGAIN`, and the webapp
publishes nothing.

`come home` starts the peel-off, the return to the launch anchor, and the land.
`abort` lands each drone where it is, immediately. The `OffboardController` base
class subscribes to both, thus every mission gets them.

### Why an unknown word rejects the full utterance

The recognizer knows six words and the token `[unk]`. If a transcript contains
the token, the webapp rejects the full utterance. This rule is necessary for
safety.

Against this vocabulary, the phrases "don't launch", "do not launch", "no
launch", "cancel launch" and "stop the launch" all become `[unk] launch`. The
token has a confidence of 1.00 in each one. If the code ignored the token to
accept more noise, all five phrases would start a launch.

The cost of the rule is that the webapp sometimes rejects a good phrase, because
noise before the phrase becomes a token. This happened one time with a clean
`jarvis abort`. The operator then says the command again, which is the cheaper
failure.

The test `test_negation_does_not_reach_the_bare_verb` holds these five examples.

## Setup

Get the speech models. The repository does not hold them.

```bash
scripts/fetch_speech_models.sh
```

The Python extras go in a uv venv for this package, as `flake.nix` specifies.
Create the venv from inside the nix `sim` shell, and give uv that interpreter:

```bash
nix develop .#sim
cd ros/jarvis_web
uv venv --python "$(which python3)" --system-site-packages
uv pip install -r requirements.txt
```

The `--python` option is necessary. Without it, uv selects its own interpreter,
which is a different minor version from the one in the nix shell. The venv then
cannot load the `rclpy` C extension, and the webapp stops at the first import.

Do this test after you make the venv:

```bash
python -c "import rclpy, vosk, piper, flask"
```

All three packages have wheels for linux aarch64. A resolution against
`aarch64-manylinux_2_28` gives vosk 0.3.45, piper-tts 1.5.0 and onnxruntime
1.27.0. The aarch64 wheels for onnxruntime need glibc 2.27 or later, thus the CM5
needs a bookworm userland or a later one.

`stt.py` is an interface, and `grammar.py` imports no engine. Thus a different
engine, for example sherpa-onnx, touches only that one file.

## HTTPS

A browser refuses access to the microphone on an insecure origin. Thus TLS is
necessary in the field.

```bash
mkcert -install                       # one time, on the machine with the CA
mkcert <drone-ap-address>             # a leaf certificate for the AP address
```

Install the mkcert CA on the Android phone of the operator. Do this one time. Go
to Settings, then Security, then Encryption & credentials, then Install a
certificate, then CA certificate.

Chrome on Android obeys a CA that the user installed. Thus the origin is clean
and the browser shows no warning page. Android then shows a permanent
notification about the network. This notification has no effect.

## Running

The order is important. The nix shell gives rclpy. The venv gives vosk, piper and
flask. colcon gives the package.

```bash
nix develop .#sim
source ros/jarvis_web/.venv/bin/activate
source install/setup.bash

ros2 run jarvis_web jarvis_web \
  --cert <name>.pem --key <name>-key.pem \
  --results-dir /tmp/maav_results
```

The default paths are `models/speech/assets` and `/tmp/maav_results`, relative to
the root of the workspace. To change them, use `--models-dir`, `--vosk-model`,
`--piper-voice`, `--results-dir` and `--threshold`.

Without `--cert` and `--key`, the webapp uses plain HTTP and writes a warning. A
desktop browser on localhost can use this mode for development, and the space bar
operates push-to-talk. A phone in this mode gives no access to the microphone.

### Drone companion service

The Drone companion image supplies ROS in `/opt/ros/jazzy`. Run Jarvis directly
from the source checkout with:

```bash
ros/jarvis_web/run_jarvis.sh
```

Install the checked-in system service once on the Drone:

```bash
sudo install -m 0644 \
  ros/jarvis_web/systemd/jarvis-web.service \
  /etc/systemd/system/jarvis-web.service
sudo systemctl daemon-reload
sudo systemctl enable --now jarvis-web.service
```

The service runs as `maav`, starts during boot without an interactive login,
and restarts after a process failure. It expects the venv, speech models, and
TLS files described above. Inspect it with:

```bash
systemctl status jarvis-web.service
journalctl -u jarvis-web.service
```

## Layout

- `grammar.py` — pure. The vocabulary, the phrases, the normal form and the
  confidence rule. No engine, no ROS and no input or output. An utterance becomes
  an `Accepted` or a `Rejected`. Only an `Accepted` holds an intent. Thus the code
  cannot make a state that publishes a gate for a command it did not recognize.
- `stt.py` — the engine interface and the vosk implementation behind it.
- `tts.py` — the piper wrapper. Text in, WAV bytes out.
- `node.py` — the ROS rim. Four publishers, and one publish for each accepted
  intent.
- `app.py` — the Flask routes and the entry point. It makes the outcome flat only
  at the JSON boundary, because JSON has no sum types.
- `static/` — the page, the push-to-talk client and the PCM worklet. The CSS holds
  only the rules that push-to-talk needs on a touchscreen. It has no colours and
  no fonts.

## Tests

Run these from inside the nix `sim` shell, with the venv active:

```bash
python3 -m pytest test/
```

The two tests that matter most use the real parts:

- `test_round_trip.py` — piper says a phrase, vosk transcribes it, and grammar.py
  classifies the result. This is the only test that shows what the recognizer does
  with sound. It skips without the speech models.
- `test_publish.py` — a real node, a real subscriber and real DDS. It shows that
  each topic name is correct, and that one publish is sufficient.

The other two use fakes. `test_grammar.py` holds the properties of the phrase
table that an edit can break. `test_app.py` holds the contract of the `/utterance`
route.

`colcon test` does not work here. It starts `python -m unittest` with no
arguments, which finds nothing. This is true for the other packages in the
workspace too, thus it is not specific to this package.
