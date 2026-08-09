// Push-to-talk. Hold the button to capture the speech, release it to send. The
// browser sends one request for each utterance.
//
// The browser keeps no queue. If a request fails, the operator says the command
// again. A command that waits in a queue during a bad link, and goes to the
// drones some time later, is a surprise. The mission moved on.

const TARGET_RATE = 16000;

// The label of the button is the full state display. There is no style to
// change, and a word is more clear than a colour.
const LABELS = { idle: "HOLD TO TALK", listening: "LISTENING", thinking: "WORKING" };

const talk = document.getElementById("talk");
const heard = document.getElementById("heard");
const verdict = document.getElementById("verdict");
const mapImage = document.getElementById("map");
const mapEmpty = document.getElementById("map-empty");

let audioContext = null;
let stream = null;
let chunks = [];
let capturing = false;

async function startCapture() {
  if (capturing) return;
  capturing = true;
  chunks = [];
  setState("listening");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false },
    });

    // A request for a 16 kHz context lets the resampler of the browser do the
    // rate conversion. That resampler is better than one in the worklet.
    audioContext = new AudioContext({ sampleRate: TARGET_RATE });
    await audioContext.audioWorklet.addModule("/static/pcm-worklet.js");

    const source = audioContext.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(audioContext, "pcm-worklet");
    worklet.port.onmessage = (event) => chunks.push(event.data);
    source.connect(worklet);
  } catch (err) {
    capturing = false;
    setState("idle");
    verdict.textContent = "MICROPHONE UNAVAILABLE";
    heard.textContent = String(err);
  }
}

async function stopCapture() {
  if (!capturing) return;
  capturing = false;
  setState("thinking");

  if (stream) stream.getTracks().forEach((t) => t.stop());

  const pcm = concat(chunks);
  chunks = [];

  if (pcm.length === 0) {
    setState("idle");
    return;
  }

  try {
    const response = await fetch("/utterance", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: pcm,
    });
    const result = await response.json();
    render(result);
    if (result.audio) await play(result.audio);
  } catch (err) {
    verdict.textContent = "NO RESPONSE";
    heard.textContent = String(err);
  } finally {
    setState("idle");
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
  }
}

function render(result) {
  heard.textContent = result.transcript ? `"${result.transcript}"` : "(nothing heard)";
  // The response text shows the result without help. LAUNCHING and SAY AGAIN
  // are sufficiently different, and the browser also speaks the response.
  verdict.textContent = result.response;
}

async function play(base64Wav) {
  const bytes = Uint8Array.from(atob(base64Wav), (c) => c.charCodeAt(0));
  // The press on the button is a user gesture, thus the browser lets the page
  // play audio. A call to Audio().play() after an await fails on a phone.
  const context = new AudioContext();
  const buffer = await context.decodeAudioData(bytes.buffer);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start();
  source.onended = () => context.close();
}

function concat(parts) {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Int16Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

function setState(state) {
  talk.textContent = LABELS[state];
}

talk.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  startCapture();
});
for (const event of ["pointerup", "pointercancel", "pointerleave"]) {
  talk.addEventListener(event, (e) => {
    e.preventDefault();
    stopCapture();
  });
}
// A held space bar does the same as the button, for work at a desk.
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat) startCapture();
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") stopCapture();
});

async function refreshMap() {
  try {
    const response = await fetch("/result", { cache: "no-store" });
    if (!response.ok) throw new Error("no result");
    const blob = await response.blob();
    if (mapImage.src.startsWith("blob:")) URL.revokeObjectURL(mapImage.src);
    mapImage.src = URL.createObjectURL(blob);
    mapImage.hidden = false;
    mapEmpty.hidden = true;
  } catch {
    mapImage.hidden = true;
    mapEmpty.hidden = false;
  }
}

refreshMap();
setInterval(refreshMap, 5000);
