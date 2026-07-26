# UWB static-grid bench evidence — 2026-07-26

> Superseded design. Keep this file as historical hardware and failure
> evidence. The current pair-local scheduler does not use the static grid or a
> clock-qualified transmission gate.

## Scope

These tests used the operational Channel 5 PHY at 6.8 Mbit/s. The DS-TWR
turnaround was 2 ms. The DWM3001C and DW1000 were both connected to
`bigrpi5`. The direct DW1000 process used `SCHED_FIFO` priority 80 on CPU 3.

The distance was not surveyed. The results are protocol and timing evidence.
They are not calibration evidence.

## Receive oracle

The DWM3001C transmitted Poll frames while the DW1000 receive oracle ran for
5 seconds. The oracle reported:

```text
frames=31
fcs_good=31
header_errors=0
fcs_errors=0
rs_errors=0
overruns=0
```

This result verifies the operational PHY and the DWM3001C-to-DW1000 receive
path.

## DWM3001C initiator and DW1000 responder

Configuration:

- DWM3001C: address 0, peer 1;
- direct DW1000: address 1, peer 0.

The direct process ran for 6.010 seconds and reported:

```text
rate=8.7Hz
base_completions=52
base_timeouts=0
ranges=52
rx_errors=0
invalid=0
scheduled_tx_misses=0
```

The measured ranges were 0.292 m through 0.413 m. The absolute delayed
Response, Final, and Report deadlines were required for this result. Scheduling
these transmissions relative to the time at which Linux finished processing
the preceding frame produced no completions.

## Clock and host transport

Embassy initially used the nRF52833 internal low-frequency RC oscillator.
Selecting the DWM3001C 32.768 kHz crystal made the 50 ppm rate bound valid.

The USB CDC four-timestamp exchange measured error bounds from approximately
0.17 ms through 0.53 ms. The USB bench admission bound is 0.60 ms, and
transmission stops at 0.75 ms. A nominal-rate model with the declared drift
bound remained qualified when USB latency changed.

The first clock loop did not pet the 4-second watchdog. The corrected loop
stayed enumerated for a 12-second unqualified-clock test. The host transmitter
also recovered after a disconnect after its probe notification changed from a
blocking queue to a latest-value signal.

## Reverse-role result

Configuration:

- direct DW1000: address 0, peer 1;
- DWM3001C: address 1, peer 0.

With a 200 µs delayed-Poll request, all 30 attempts timed out before `TXFRS`:

```text
polls=30
tx_completion_timeouts=30
rx_completion_timeouts=0
ranges=0
```

`SCHED_FIFO` priority 80 did not make this deadline viable.

Preparing the Poll 1 ms before its marker removed the transmit failure:

```text
polls=29
tx_completion_timeouts=0
rx_completion_timeouts=23
ranges=6
```

The reverse role is not qualified. The next test must isolate the remaining
Response receive timeouts before changing another timing value.

## Excluded three-node run

The public-NTP offsets were +1.766 ms on `bigrpi5` and -1.971 ms on `drone2`.
The approximately 3.7 ms difference exceeds the 0.75 ms static-grid clock
contract. A run performed in that state produced no cross-host completions and
is not radio or MAC evidence.

DS-TWR ranging does not require synchronized clocks. The common clock is used
only by the static collision-avoidance grid. A MAC without a common time
reference must use a different access policy, such as randomized contention.
