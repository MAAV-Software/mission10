# Mission 10 UWB software

This Cargo workspace contains the Mission 10 UWB radio applications and their
shared protocols. The Rust programs in this directory are software. The opaque
code inside a radio is firmware.

## Components

- `protocol/` contains the `no_std` air and host protocols, pair scheduling,
  clock correlation, ranging arithmetic, and generated golden frames.
- `dwm3001/` contains the Embassy application for the DWM3001CDK nRF52833 and
  DW3110.
- `dw1000/` contains the Linux application for a CM5-attached DW1000.
- `protocol/python/` contains the independent host codec and CDC bench tool.

Both applications use the same air protocol and pair policy. SPI, IRQ, reset,
and radio-state effects remain backend-specific.

The complete development inventory is four aircraft DW1000 modules and two
movable DWM3001CDKs. A configuration therefore supports six nodes and five
peers per node. The competition roster contains four aircraft.

## Pair-local medium access

Air protocol version 6 uses PAN `0x4d10` and IEEE 802.15.4 short addresses.
Aircraft use addresses `0..3`. Development nodes use `0x8000..0x80ff`.
Poll, Response, Final, and Report are unicast. `FleetMode` is broadcast and
contains the selected aircraft master and `field` or `internet` network.

For each configured pair:

- the lower address initiates;
- the responder listens whenever it is not transmitting;
- far pairs target one attempt per 100 ms;
- pairs below 3.0 m target one attempt per 10 ms;
- close state ends at 3.5 m or after 300 ms without a complete exchange;
- startup phase and failure backoff use pair-local pseudorandom state.

The initiator assigns a wrapping 16-bit exchange ID. Every message in the
exchange echoes it. A responder rejects an immediate duplicate.

One radio executes one exchange at a time. If several local pairs are due, it
selects the most overdue pair. Address order breaks a tie. A failed attempt
widens its retry window, up to 400 ms. A successful attempt returns to the
10 ms or 100 ms cadence.

These rates are targets and caps. There is no deterministic fleet-wide rate
under contention. There is also no coordinator, schedule beacon, time cell,
owner election, shared radio clock, or runtime schedule plan. An absent peer
cannot remove another pair's permission to range.

## Time metadata

DS-TWR uses local radio timestamp differences and does not require synchronized
node clocks.

Host clock correlation is passive. It maps the Final event to mission time
after a range is complete. It never admits, delays, or suppresses a radio
exchange. `CompletedExchange` always contains the raw 40-bit radio event time
and contains either:

- a mapped mission time, generation, and error bound; or
- `MissionEventTime::Unavailable`.

The DWM application exchanges non-blocking clock probes with the host and
brackets a DW3110 system-time read when it maps an event. Mapping becomes
unavailable above a 750 us error bound. Ranging continues.

The direct application brackets a DW1000 system-time read with Linux realtime
samples and prints the mapped value when the bracket meets the same bound.

## Host protocol

Host protocol version 9 uses:

```text
COBS(Hubpack(HostToRadioEnvelope | RadioToHostEnvelope) ||
     CRC-32/ISO-HDLC) || 0x00
```

It carries configuration, ego state, `FleetMode`, passive clock correlation,
health requests, completed exchanges, diagnostics, and counters. The host
sender retains the latest unsent exchange for each peer. USB packet boundaries
are not protocol boundaries.

The same byte stream can use USB CDC or UART. Host ego state becomes stale
after 100 ms without an update. The radio keeps ranging and sets `HOST_STALE`.

## Build and test

Run from `mission10/uwb`:

```sh
nix develop ..#uwb

cargo fmt --all --check
cargo test --target x86_64-unknown-linux-gnu -p mission10-uwb-protocol
cargo test --target x86_64-unknown-linux-gnu -p dw1000-rs -p mission10-dw1000
cargo check -p mission10-dwm3001
cargo check -p mission10-dwm3001 --features engineering-sample
cargo build --release -p mission10-dwm3001
# Run this native release build on bigrpi5.
cargo build --release --target aarch64-unknown-linux-gnu -p mission10-dw1000
uv run --with pytest pytest -q protocol/python/test_protocol.py
```

The Rust encoders generate:

- `protocol/testdata/air_protocol_v6.frames`;
- `protocol/testdata/host_protocol_v9.frames`.

The Python tests decode the host fixture independently. Do not hand-edit these
files. Regenerate an intentional format change with:

```sh
UPDATE_AIR_GOLDEN=1 cargo test --target x86_64-unknown-linux-gnu \
  -p mission10-uwb-protocol every_air_message_matches_committed_golden_bytes
UPDATE_GOLDEN=1 cargo test --target x86_64-unknown-linux-gnu \
  -p mission10-uwb-protocol committed_fixture
```

## DWM3001CDK operation

Build on the workstation and flash through J9:

```sh
scp target/thumbv7em-none-eabihf/release/mission10-dwm3001 drone2:/tmp/
ssh drone2 \
  '~/.local/bin/probe-rs download --chip nRF52833_xxAA \
   /tmp/mission10-dwm3001 && \
   ~/.local/bin/probe-rs reset --chip nRF52833_xxAA'
```

Configure the node and every selected peer:

```sh
python3 uwb/protocol/python/mission10_uwb_protocol.py \
  /dev/serial/by-id/usb-MAAV_Mission_10_DWM3001_bring-up_DWM3001-01-if00 \
  --configure-node 0x8000 \
  --peer 0 --peer 1 --peer 2 --peer 3 --peer 0x8001 \
  --request-health
```

Broadcast a fleet selection from a DWM3001 host without ROS:

```sh
python3 uwb/protocol/python/mission10_uwb_protocol.py /dev/ttyACM0 \
  --fleet-master 2 --network field
```

The application initializes the DW3110 at 4 MHz, raises SPI to 32 MHz, reports
OTP and antenna delays, checks actual delayed timestamps, retains reset causes,
and uses a four-second hardware watchdog.

## Direct DW1000 operation

Known CM5 wiring uses `/dev/spidev0.0`, active-high IRQ on GPIO24, and
open-drain reset on GPIO25.

```sh
scp target/aarch64-unknown-linux-gnu/release/mission10-dw1000 drone2:/tmp/
ssh drone2 'chmod 755 /tmp/mission10-dw1000'

ssh drone2 '/tmp/mission10-dw1000 probe --address 2'
ssh drone2 \
  'sudo /tmp/mission10-dw1000 range --rt-priority 80 \
   --address 2 --peer 0 --peer 1 --peer 3 \
   --host-socket /run/uwb/host.sock --duration 60 --quiet'
```

The process reports its scheduler policy, priority, and current CPU at startup.
Qualification runs use `SCHED_FIFO` without CPU affinity. The policy improves
Linux service latency; it does not change DS-TWR timestamp arithmetic.

`--low-tx-power` reduces TX gain by 8.5 dB for close-range diagnosis. The
standalone `rx-oracle` observes the receive pipeline without adding output or
extra SPI work to the ranging state machine.

## Remaining flight gates

The implementation is ready for pair-local bench testing. It is not
flight-qualified. Required evidence includes:

- both physical roles for every installed radio pair;
- 9–11 completed ranges/s for one far pair;
- at least 90 completed ranges/s, and no more than 100 attempts/s, for one
  uncontended close pair;
- three-node progress without synchronized host clocks;
- continued live-pair service when another peer is absent;
- calibrated antenna delays and surveyed accuracy;
- direct-backend results under representative CM5 load and `SCHED_FIFO`;
- UART time-correlation, signal-integrity, and host-stall results;
- multi-hour outage distributions and final airframe packaging.

Git history contains change history. Repeatable bench records are in
`test-evidence/`.
