# Mission 10 UWB software

This Cargo workspace owns both UWB radio applications and the wire protocol
shared with the CM5. The DW3110's opaque internal code is firmware; the Rust in
this directory is ordinary Mission 10 software.

## Workspace

- `dwm3001/` — Embassy application, DW3110 driver, DS-TWR state machine, J20
  USB CDC adapter, and J9 debug metadata.
- `dw1000/` — Linux application for the CM5-attached DW1000, including the
  SPI/GPIO adapters and native addressed DS-TWR CLI.
- `protocol/` — `no_std` native air frames, ranging arithmetic, and the typed
  bidirectional host protocol.
- `protocol/python/` — independent host decoder and cross-language golden test.

The host stream is exactly:

```text
COBS( Hubpack(HostToRadioEnvelope | RadioToHostEnvelope) || CRC-32/ISO-HDLC ) || 0x00
```

The same envelopes are intended for J20 USB CDC and the later UART adapter.
Both radio applications use the native addressed Poll/Response/Final/Report
air protocol. Response and Final use 2 ms delayed turns. Report uses a separate
2 ms delayed turn so that the initiator can start a clean receive attempt after
Final TX.

## Build and test

```sh
nix develop .#uwb
cd uwb

cargo fmt --all --check
cargo test --target x86_64-unknown-linux-gnu -p mission10-uwb-protocol
cargo test --target x86_64-unknown-linux-gnu -p mission10-dw1000
cargo build --release -p mission10-dwm3001
cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.31 -p mission10-dw1000
# Or, for the reverse-role diagnostic instead:
cargo build --release -p mission10-dwm3001 --features initiator
```

`protocol/testdata/host_protocol_v4.frames` is the shared wire contract: the
Rust `committed_fixture_matches_the_encoders` test regenerates it from the
encoders and fails if the checked-in copy is stale, while the Python suite
decodes the same file and asserts its meaning. The two codecs stay independent;
only the bytes have one owner. Never hand-edit the file — after an intended
format change, regenerate it (which normally accompanies a protocol version
bump):

```sh
UPDATE_GOLDEN=1 cargo test -p mission10-uwb-protocol committed_fixture
```

`protocol/testdata/air_protocol_v1.frames` independently pins the complete
MAC-plus-Hubpack bytes for Poll, Response, Final, and Report. These vectors do
not include the two-byte PHY FCS.

The default build is the native short-address `1` responder for peer `0`. It
waits on the DW3110's active-high IRQ at nRF `P1.02`;
`--features initiator` builds the reverse-role address `0` diagnostic for peer
`1`.
`engineering-sample` must only be enabled for a module physically marked E1.0.
Both roles use the same default ELF path. Build them sequentially or give each
role a separate `CARGO_TARGET_DIR`.

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

Host protocol version 4 is bidirectional. It carries configuration for a
16-bit node address and up to three peers, fixed-point ego state, and health
requests. The current fixed-role DWM3001 diagnostic builds reject `Configure`
with `UnsupportedInMode`; the fleet scheduler will apply it when runtime peer
selection lands.
The radio reports configuration readback, addressed ranges, peer state, typed
diagnostics, and cumulative radio/transport/queue counters. Independent bounded
queues keep range and peer-state delivery ahead of discardable diagnostics.

Native air frames use PAN `0x4d10` and IEEE 802.15.4 short source/destination
addresses. Drone addresses are `0..4`; bench/development addresses occupy
`0x8000..0x80ff`. Current ranging messages require a unicast destination;
`0xffff` is reserved for a future broadcast message with a scheduler consumer.
The protocol and ranging arithmetic are shared by the DWM3001 and direct-DW1000
applications; each radio retains its own effectful state machine. The direct
application round-robins configured higher-address peers and responds to
configured lower-address peers.

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

The committed `protocol/testdata/host_protocol_v4.frames` fixture is the single
golden wire contract consumed by both the Rust encoder tests and Python codec
tests.

## DW1000 Linux flight-candidate spike

The direct-CM5 application is built on the workstation for aarch64 Linux. Zig
links it against a glibc 2.31 compatibility floor and records the standard
`/lib/ld-linux-aarch64.so.1` interpreter, so the artifact runs on the fleet Pis
without a Nix installation. The known `bigrpi5` wiring is CE0 at
`/dev/spidev0.0`, active-high IRQ on GPIO24, and open-drain RSTn on GPIO25.
Logical reset-high releases GPIO25 as an input with a pull-up.

```sh
nix develop .#uwb
cd uwb
cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.31 -p mission10-dw1000
scp target/aarch64-unknown-linux-gnu/release/mission10-dw1000 bigrpi5:/tmp/
ssh bigrpi5 'chmod 755 /tmp/mission10-dw1000'

ssh bigrpi5 '/tmp/mission10-dw1000 probe --address 0x8000'
ssh bigrpi5 \
  '/tmp/mission10-dw1000 range --address 0x8000 --peer 2 --duration 60 --quiet'

# RF-link diagnostic for two direct DW1000 nodes:
ssh bigrpi5 \
  '/tmp/mission10-dw1000 range --address 0x8000 --peer 2 --robust-phy \
   --poll-period-ms 100 --reply-delay-us 5000 --timeout-ms 100'

# Close-range diagnostic:
ssh bigrpi5 \
  '/tmp/mission10-dw1000 range --address 0x8000 --peer 2 --low-tx-power \
   --duration 10 --quiet'
```

`probe` performs a hardware reset, initializes at 2 MHz, verifies device ID
`0xDECA0130`, and then raises SPI to 20 MHz. `range` defaults to the DWM3001
native PHY, 2 ms delayed legs, a 10 ms poll period, and a 10 ms response bound.
Lower-address nodes initiate; every node also answers a configured peer's Poll.
Exchange failures re-arm receive and preserve the process for peer recovery.
Every frame carries source and destination short addresses. A node answers all
configured higher-address peers and responds to configured lower-address
peers.
`--robust-phy` selects 110 kbps and a 2048-symbol preamble while retaining
Channel 5, 64 MHz PRF, and preamble code 10. It is a link-diagnosis profile;
both DW1000 endpoints must select it, and its longer frames require relaxed
poll and response timing.
Recovery follows the terminal radio status. Preamble and SFD timeouts re-arm
RX without a receiver reset. Header, frame-check, Reed-Solomon, frame-wait,
LDE, filtering, overrun, and frame-read failures reset the receiver before
re-arm. The standalone `rx-oracle` command provides receiver-state diagnostics
without perturbing the ranging state machine.
`--low-tx-power` retains the operational PHY while reducing the DW1000's
transmit gain by 8.5 dB for close-range diagnosis.

The application uses `dw1000-rs` 0.2.0. Its exact upstream source is retained
under `dw1000/vendor/` with documented blocking and async additions for
device-ID readout, typed delayed deadlines and on-air timestamps, absolute
delayed TX, raw receive timestamps, receiver reset, bounded radio-state
snapshots, and raw transmit-power control. Initialization also applies
Decawave's crystal-clock OTP reads,
crystal-PLL lock detector, programmed LDO tune, and AON cleanup. Receiver
configuration derives `DRX_SFDTOC` from the active preamble, SFD, and PAC.
Explicit RX follows Decawave's force-off, status-clear, and buffer-pointer
synchronization sequence, and exposes the complete terminal/clear event masks
used by the Linux event service. The TX entry point writes the unadjusted
`DX_TIME` deadline while returning the antenna-adjusted timestamp carried by
DS-TWR. Raw RX timestamps keep bias and antenna calibration at the shared
subsystem layer. Remove the local patch when an upstream release provides
equivalent semantics.

### Rust DW1000 bench result (2026-07-21)

The Rust initiator ranged against the `bigrpi5` DWM3001CDK responder for 60
seconds at the 10 ms operating point. It completed 5,957 of 6,001 polls
(99.27%), delivered 99.3 ranges/s, decoded zero invalid frames, and reported
zero delayed-TX timestamp mismatches. Current unsurveyed bench readings were
approximately 0.3--0.55 m; calibration remains pending.

Ten consecutive J-Link resets of the DWM3001 peer produced the expected bounded
timeouts; the same DW1000 process resumed and completed 804 ranges during the
10-second disturbance run. Under the same 2 ms/10 ms/20 MHz settings, the
existing Python oracle delivered 79.2 ranges/s in a separate 10-second run.

### Native mixed-radio result (2026-07-25)

A DW1000 on `drone2` at native address `0` ranged with the DWM3001 on `drone4`
at native address `1`. It completed 965 of 1,001 polls in 10 seconds (96.5
ranges/s), with zero invalid, wrong-peer, or unexpected frames. J20 decoded the
same protocol-v4 run and reported addressed range and peer-state events.
After the DW1000 receiver diagnostics and Decawave initialization alignment,
a final regression completed 969 of 1,001 polls in 10 seconds (96.9 ranges/s).
It reported zero invalid frames, wrong-peer frames, unexpected events, and
scheduled-transmit misses.

A three-node run then assigned the DW1000 on `drone0` address `0`, the DWM3001
on `drone4` address `1`, and the DW1000 on `drone2` address `2`. Address `0`
round-robined both peers and completed 490 of 1,000 polls in 10 seconds: the
address-1 slots succeeded at their expected 50 Hz share, while every address-2
slot timed out.

Later two-node tests exposed an address-collision confounder: the powered
DWM3001 responder at address `1` had answered nominal DW1000 address-0/1 tests
even while the direct responder reported zero received frames. The earlier
apparent 59.4 Hz and 9.5 Hz reverse-role DW1000 results therefore did not
measure a DW1000-to-DW1000 link.

Fresh tests used addresses `2` and `3` with `drone0` and `drone2` less than
20 cm apart. Both physical role assignments completed zero exchanges on the
operational PHY. The robust PHY, an 8.5 dB transmit-power reduction, Decawave's
missing OTP/LDO initialization steps, and receiver-reset policies `never`,
`recovery`, and `always` also completed zero. Bounded register traces showed
explicit RX entering the undocumented `SYS_STATE` preamble-hunt states without
an RX event. Repeatedly entering automatic WAIT4RESP before explicit RX
partially changed the result to 167 of 488 polls (33.4 Hz), which demonstrates
a state-dependent receive path but is not an acceptable workaround.

The third DW1000 separates the common software path from individual links.
`drone0` and `bigrpi5` completed in both physical role assignments: 69 of 481
polls (13.8 Hz) with `bigrpi5` initiating and 136 of 486 (27.2 Hz) with
`drone0` initiating. `drone2` and `bigrpi5` completed zero in both assignments.

A fourth DW1000 on `drone1` passed device-ID, reset, IRQ, and 20 MHz SPI
bring-up without a boot-configuration change. With the pre-namespace
experimental addresses `6` and `7`, `drone1` initiated 226 completed exchanges
from 986 polls (22.6 Hz)
against `bigrpi5`. In the reverse physical role, `bigrpi5` completed zero of
959 polls and the `drone1` responder observed no receive event. Resetting the
receiver before the responder arm also completed zero of 955 polls.

The apparent cold-responder failure was in the Linux event service. A raw
one-millisecond status oracle on `drone2` observed 534 valid, FCS-good Polls
and five SFD timeouts in 12 seconds, with no header, FCS, Reed-Solomon, LDE,
filter, overrun, or buffer-pointer errors. This isolated the fault above the
radio and RF link.

Two omitted terminal conditions could strand explicit RX after noise:
`RX_PREAMBLE_TIMEOUT` and `RX_SFD_TIMEOUT`. After a valid frame, the service
then cleared `RX_FRAME_READY` but left the interrupt-masked `RX_FRAME_GOOD`
latch asserted. IRQ stayed high, so a later TX-complete event could not produce
a rising edge; the software deadline also ran before the status-poll fallback.
The final path handles every terminal RX condition, clears all status belonging
to the receive attempt, and services latched hardware events before software
deadlines. Start-RX also uses Decawave's force-off, status-clear, and
receive-buffer-pointer synchronization sequence.

With the pre-namespace experimental addresses `6` and `7`, three cold-`drone2`
responder trials completed
623/979, 772/989, and 693/983 initiator polls in 10 seconds (62.3--77.1 Hz).
Reversing the physical roles completed 810/990 polls (81.0 Hz), with
`bigrpi5` reporting 816 responder ranges. All four trials reported zero
invalid frames, wrong-peer frames, unexpected TX events, and scheduled-TX
mismatches. An A/B run gave identical recovery without the forced-TX-clock
implementation of the DW1000 TX-1 erratum workaround, so the final path omits
that experiment. The exact verified build later completed 140/480 polls in a
five-second smoke run (28.0 Hz), so direct-link throughput remains variable
even though both roles now recover and complete exchanges.
Calibration, system load, and repeat tests with the now-unpowered `drone0` and
`drone1` nodes remain open before the direct backend can be promoted.

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
orientation, signal-level, warm-up, and fleet-scheduling tests remain required.

## IRQ bench result (2026-07-18)

The radio wait path uses a race-safe `status -> wait_for_high/deadline -> status`
loop. The driver interrupt masks are changed in `Ready` before each TX or RX;
the retained one-second deadline diagnoses a missing interrupt. Bounded
response legs additionally use the DW3110 receive-frame-wait timer, so a lost
packet ends the active exchange in hardware rather than leaving both peers in
RX until noise happens to raise an SFD/PHY error.

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

## Two-DWM3001CDK bench result (2026-07-18)

The `drone2` initiator and `bigrpi5` responder completed DS-TWR in both radios
using the earlier, now-retired provisional air codec. With no hardware
frame-wait timeout, one lost packet could leave both peers receiving
indefinitely; random RF activity eventually produced SFD/PHY flags and
accidentally restarted the exchange. A 30-second baseline completed only about
52 ranges before this failure mode dominated.

The IEEE short-SFD timeout is now the recommended 129 symbols for the
128-symbol preamble, standard 8-symbol SFD, and PAC8. PollAck, Range, and
RangeReport reception use a 10 ms DW3110 `RX_FWTO` deadline; the responder's
idle wait for Poll remains unbounded. The initiator's one-second missing-IRQ
counter stayed at zero after this change, and frame-wait expirations recovered
the transaction deterministically.

The corrected responder reported 5,267 ranges in 60 seconds (87.8 Hz), 114
recoverable diagnostics, and zero invalid host frames. An upstream USB hub on
`drone2` re-enumerated J-Link, J20, Ethernet, and video together during the
soak; ranging continued, distinguishing that host USB event from a radio reset.
The unsurveyed range was roughly 5.3--5.8 m and is not a calibration result.

A subsequent DW3000 manual audit confirmed that `dw3000-ng` performs the
mandatory PGF/RX calibration during configuration. It also found the documented
delayed-transmit corner where neither HPDWARN nor TXFRS is asserted. TX
completion is bounded and checks `PMSC_STATE` plus `TX_STATE` to identify that
silent failure before aborting the exchange. The initial 20 ms guard produced
3,019 responder and 3,029 initiator ranges in 35 seconds (86.4--86.7 Hz) with
zero invalid host frames.

Reducing the guard to 10 ms produced 3,034 responder and 3,038 initiator ranges
in the same interval (86.8--87.0 Hz). Neither run produced a TX-completion or
radio-state failure, and the initiator software-timeout counter remained zero.
The 10 ms capture encountered one partial host frame immediately after opening
an already-streaming CDC endpoint and none in a warm follow-up. Recoverable
radio errors remained RX frame-wait, PHY, Reed-Solomon, and SFD failures.
