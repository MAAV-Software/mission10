# Mission 10 UWB software

This Cargo workspace owns the DWM3001CDK's nRF52833 application and the wire
protocol shared with the CM5. The DW3110's opaque internal code is firmware;
the Rust in this directory is ordinary Mission 10 software.

## Workspace

- `dwm3001/` — Embassy application, DW3110 driver, DS-TWR state machine, J20
  USB CDC adapter, and J9 debug metadata.
- `protocol/` — `no_std` legacy DW1000 air frames and the typed host envelope.
- `protocol/python/` — independent host decoder and cross-language golden test.

The host stream is exactly:

```text
COBS( Hubpack(HostEnvelope) || CRC-32/ISO-HDLC ) || 0x00
```

The same envelope is intended for J20 USB CDC and the later UART adapter. The
temporary 18-byte over-the-air format is deliberately separate: it exists only
to range against the surviving DW1000 while the second DWM3001CDK is unavailable.

## Build and test

Run all compilation on a workstation, never on a Pi:

```sh
nix develop .#uwb
cd uwb

cargo fmt --all --check
cargo test --target x86_64-unknown-linux-gnu -p mission10-uwb-protocol
cargo build --release -p mission10-dwm3001
# Or, for the reverse-role diagnostic instead:
cargo build --release -p mission10-dwm3001 --features initiator
```

The default build is the `A1:C1` responder used with the legacy DW1000
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
```

The decoder sets the tty to raw mode. Any other host reader must do the same;
otherwise Linux's terminal line discipline will consume binary control bytes.

Host protocol version 2 reports typed `Diagnostic` variants instead of reused
numeric error codes. Its `Health` message carries cumulative IRQ wakes,
spurious IRQ wakes, software wait timeouts, and receive recoveries. Health is
emitted at startup and every 32 completed ranges.

## Bench result (2026-07-18)

The connected production module reports DW3110 ID `DECA/3/0/2`, OTP transmit
power `0x61616161`, antenna delays `0x3ff0/0x3ff0`, crystal value
`0x00be0019`, and OTP revision `0x00010201`.

Both role directions completed DS-TWR against the surviving DW1000 on
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
