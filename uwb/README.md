# Mission 10 UWB software

This Cargo workspace owns the DWM3001CDK's nRF52833 application and the wire
protocol shared with the CM5. The DW3110's opaque internal code is firmware;
the Rust in this directory is ordinary Mission 10 software.

## Workspace

- `dwm3001/` — Embassy application, DW3110 driver, DS-TWR state machine, J20
  USB CDC adapter, and J9 debug metadata.
- `protocol/` — `no_std` native air frames, isolated DW1000 bench frames,
  ranging arithmetic, and the typed bidirectional host protocol.
- `protocol/python/` — independent host decoder and cross-language golden test.

The host stream is exactly:

```text
COBS( Hubpack(HostToRadioEnvelope | RadioToHostEnvelope) || CRC-32/ISO-HDLC ) || 0x00
```

The same envelopes are intended for J20 USB CDC and the later UART adapter. The
temporary 18-byte over-the-air format is deliberately separate and selected by
the default `dw1000-bench` build feature. It exists only to range against the
DW1000 bench node on `bigrpi5` while the second DWM3001CDK is unavailable.

## Build and test

```sh
nix develop .#uwb
cd uwb

cargo fmt --all --check
cargo test --target x86_64-unknown-linux-gnu -p mission10-uwb-protocol
cargo build --release -p mission10-dwm3001
# Or, for the reverse-role diagnostic instead:
cargo build --release -p mission10-dwm3001 --features initiator
```

`protocol/testdata/host_protocol_v3.frames` is the shared wire contract: the
Rust `committed_fixture_matches_the_encoders` test regenerates it from the
encoders and fails if the checked-in copy is stale, while the Python suite
decodes the same file and asserts its meaning. The two codecs stay independent;
only the bytes have one owner. Never hand-edit the file — after an intended
format change, regenerate it (which normally accompanies a protocol version
bump):

```sh
UPDATE_GOLDEN=1 cargo test -p mission10-uwb-protocol committed_fixture
```

The default build is the `A1:C1` responder used with the DW1000
initiator and waits on the DW3110's active-high IRQ at nRF `P1.02`.
`--features initiator` builds the reverse-role `A0:C0` diagnostic.
`engineering-sample` must only be enabled for a module physically marked E1.0.
Build the role you intend to flash last; both variants use the same ELF path.

Flash the workstation-built ELF through J9:

```sh
scp target/thumbv7em-none-eabihf/release/mission10-dwm3001 bigrpi5:/tmp/
ssh bigrpi5 \
  '~/.local/bin/probe-rs download --chip nRF52833_xxAA /tmp/mission10-dwm3001 && \
   ~/.local/bin/probe-rs reset --chip nRF52833_xxAA'
```

Decode J20 on the Pi with only Python's standard library:

```sh
python3 uwb/protocol/python/mission10_uwb_protocol.py \
  /dev/serial/by-id/usb-MAAV_Mission_10_DWM3001_bring-up_DWM3001-01-if00

# Exercise the bidirectional link with an immediate health request:
python3 uwb/protocol/python/mission10_uwb_protocol.py \
  --request-health \
  /dev/serial/by-id/usb-MAAV_Mission_10_DWM3001_bring-up_DWM3001-01-if00

# Or measure aggregate range rate without printing every radio event:
python3 uwb/protocol/python/mission10_uwb_protocol.py \
  --summary-interval 2 \
  /dev/serial/by-id/usb-MAAV_Mission_10_DWM3001_bring-up_DWM3001-01-if00
```

The decoder sets the tty to raw mode. Any other host reader must do the same;
otherwise Linux's terminal line discipline will consume binary control bytes.
A writer sends an empty COBS boundary after opening the stream so stale bytes
from a prior USB/UART connection cannot corrupt its first command; the radio
treats this empty boundary as synchronization, not a malformed frame.

Host protocol version 3 is bidirectional. The native DWM3001 mode can configure
a 16-bit node address and up to three peers; all modes can receive fixed-point
ego state and health requests. The temporary DW1000 bench mode rejects
`Configure` with `UnsupportedInMode` because its peer bytes remain compile-time
interoperability constants.
The radio reports configuration readback, addressed ranges, peer state, typed
diagnostics, and cumulative radio/transport/queue counters. Independent bounded
queues keep range and peer-state delivery ahead of discardable diagnostics.

Native air frames use PAN `0x4d10` and IEEE 802.15.4 short source/destination
addresses. Drone addresses are `0..4`; bench/development addresses occupy
`0x8000..0x80ff`. The native four-message ranging FSM remains gated until a
second DWM3001CDK is available, but its codec and shared timestamp arithmetic
are covered by workstation tests now.

Runtime PHY/timing failures abandon only the active exchange. Finish-state and
SPI preparation failures receive bounded retries; an error from a consuming
`dw3000-ng` start API deliberately resets the nRF because that API does not
return ownership of the radio. The triggering diagnostic survives in GPREGRET
and is emitted before radio initialization after reboot. A three-second,
watchdog-petted backoff repeats the retained cause at bounded intervals so it
remains visible while USB enumerates and the host opens the tty, even when a
permanently absent radio triggers another reset. A four-second hardware watchdog
independently resets a stalled radio task and is paused while a debugger halts
the CPU. The `recoveries` health counter includes both recovered exchanges and
successful finish-state retries.

The committed `protocol/testdata/host_protocol_v3.frames` fixture is the single
golden wire contract consumed by both the Rust encoder tests and Python codec
tests.

## Bench result (2026-07-18)

The connected production module reports DW3110 ID `DECA/3/0/2`, OTP transmit
power `0x61616161`, antenna delays `0x3ff0/0x3ff0`, crystal value
`0x00be0019`, and OTP revision `0x00010201`.

Both role directions completed DS-TWR against the DW1000 bench node on
`bigrpi5`. The responder direction produced 201 samples in 30.2 seconds
(approximately 6.7 Hz). The reverse direction sustained repeated 8–25 cm bench
measurements, reported typed Hubpack ranges over J20, timed out visibly when the
DW1000 process stopped, and resumed after only the peer process restarted.

These are bring-up observations, not calibration results. Surveyed-distance,
orientation, signal-level, warm-up, and two-DWM3001CDK testing remain required.

## IRQ bench result (2026-07-18)

The radio wait path uses a race-safe `status -> wait_for_high/deadline -> status`
loop. The driver interrupt masks are changed in `Ready` before each TX or RX;
the retained one-second deadline diagnoses a missing interrupt rather than
acting as the normal progress mechanism.

The IRQ responder sustained 341 ranges in 41.3 seconds (8.2 Hz), reported IRQ
wakes with zero spurious wakes, and resumed after its DW1000 initiator was
stopped and restarted. The IRQ initiator also ranged in the reverse direction,
reported health counters, and resumed after its DW1000 responder was stopped
for four seconds and restarted. The default responder was finally downloaded
and reset into standalone operation on `bigrpi5`.

## High-rate one-pair bench result (2026-07-18)

After initialization at the required 4 MHz, the DWM3001 application now raises
SPIM3 to 32 MHz. Both delayed DS-TWR legs default to 2 ms, delayed-send deadline
failures recover to the next exchange instead of panicking, and the diagnostic
initiator waits 5 ms rather than 100 ms between exchanges.

The DW1000 on `bigrpi5` was used as initiator with its post-init SPI
clock at 20 MHz. Its event wait now observes the configured poll deadline rather
than imposing a hidden 50 ms floor, and its stalled-exchange watchdog is
configurable with `--watchdog-ms`.

Measured close-range regimes:

| Requested period | Delayed legs | Completed | Result | Recoveries |
| --- | --- | ---: | ---: | ---: |
| 20 ms | 2 ms / 2 ms | 486 / 489 | 48.4 Hz | 2 POLL_ACK |
| 10 ms | 2 ms / 2 ms | 921 / 931 | 90.0 Hz | 8 POLL_ACK, 2 RANGE_REPORT |
| 5 ms stress | 2 ms / 1 ms | 1382 / 1396 | 172.0 Hz | 9 POLL_ACK, 4 RANGE_REPORT |

J20 independently tracked the active 50 Hz and 100 Hz regimes without invalid
host frames. The 5 ms result is a stress observation, not the fleet operating
point: Mission 10 should schedule close pairs at 50 Hz, selectively promote an
urgent pair toward 100 Hz, and retain airtime for the other five possible pairs.
