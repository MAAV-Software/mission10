// This worklet changes the microphone frames to 16 kHz mono signed 16-bit PCM.
// That is the format that stt.py sends to the recognizer. The conversion is here,
// thus there is no container format and no decode step on the CM5.
//
// app.js requests an AudioContext at 16 kHz. Thus `sampleRate` is usually 16000,
// `ratio` is 1, and the code below does not resample. The linear path is only for
// a browser that ignores the requested rate.
//
// The linear path has no anti-alias filter, and it does not interpolate across a
// block boundary. Both limits are acceptable, because this path is the second
// choice. When the code uses it, the browser already refused to resample. Also,
// the recognizer does not use data above 8 kHz.

const TARGET_RATE = 16000;

class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    // The position of the next output sample, from the start of the current
    // block. The carry at the end of process() keeps it in the range [0, ratio).
    this.cursor = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    if (this.ratio === 1) {
      this.port.postMessage(toInt16(channel.length, (i) => channel[i]));
      return true;
    }

    const count = Math.max(0, Math.ceil((channel.length - this.cursor) / this.ratio));
    if (count > 0) {
      const cursor = this.cursor;
      const ratio = this.ratio;
      this.port.postMessage(
        toInt16(count, (i) => {
          const pos = cursor + i * ratio;
          const low = Math.floor(pos);
          const a = channel[low];
          const b = channel[low + 1];
          return b === undefined ? a : a + (b - a) * (pos - low);
        })
      );
    }
    this.cursor = this.cursor + count * this.ratio - channel.length;
    return true;
  }
}

function toInt16(count, sampleAt) {
  const out = new Int16Array(count);
  for (let i = 0; i < count; i++) {
    const s = Math.max(-1, Math.min(1, sampleAt(i)));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

registerProcessor("pcm-worklet", PcmWorklet);
