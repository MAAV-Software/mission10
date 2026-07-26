# UWB pair-local scheduler — implementation record — 2026-07-26

## Scope

This record separates prior static-grid measurements from the pair-local
implementation that replaced that design.

Air protocol version 4 and host protocol version 7 remove time cells,
opportunity identifiers, and clock-qualified transmission. The lower address
initiates each pair. Far and close attempt periods are 100 ms and 10 ms.
Failure backoff is randomized and bounded at 400 ms.

The bounded development configuration supports:

- four installed aircraft DW1000 radios;
- two movable DWM3001CDKs;
- five peers per node;
- fifteen pair states.

Mission-time correlation is passive. An unavailable mapping does not stop
ranging.

## Checks completed

- Rust air and host golden fixtures were regenerated from the versioned
  encoders.
- The Python host codec decodes the same host fixture.
- Protocol and direct-backend host tests pass.
- Both DWM3001 build configurations pass their embedded target checks.

## Close-pair smoke test

The DWM3001C on `bigrpi5` used address 0 and initiated to the direct DW1000 at
address 1. The direct process used `SCHED_FIFO` priority 80 on CPU 3. The
distance was not surveyed.

After reserving 1 ms to leave receive mode before a due Poll, the DWM host
reported:

```text
3 s: 296 ranges, 98.3 Hz
6 s: 594 ranges, 98.7 Hz
9 s: 889 ranges, 98.5 Hz
```

Mapped Final event-time error bounds in the sampled output were 393 through
543 us. The direct responder reported:

```text
polls_rx=1381
finals_rx=1377
reports_tx=1377
timeouts=1
scheduled_tx_misses=0
reinitializations=0
```

This result passes the one-direction close-pair rate gate. It is not
calibration evidence.

The physical roles were then reversed. The direct DW1000 at address 0
initiated to the DWM3001C at address 1. The direct process again used
`SCHED_FIFO` priority 80 on CPU 3 and reported:

```text
12 s: 1166 ranges, 97.2 Hz
polls_tx=1176
responses_rx=1170
finals_tx=1170
reports_rx=1166
timeouts=2
scheduled_tx_misses=0
reinitializations=0
```

The DWM host output contained mapped Final event times with sampled error
bounds of 181 through 244 us. This run did not reproduce the earlier
reverse-role Response receive timeout.

## Direct DW1000 pair

The final pair-local build ran between the direct DW1000s on `bigrpi5` and
`drone2`. Both processes used `SCHED_FIFO` priority 80 on CPU 3. Each physical
role assignment ran for 30 seconds. The distance was not surveyed.

With `bigrpi5` as address 0 and initiator:

```text
30.052 s: 2742 ranges, 91.2 Hz
polls_tx=2838
responses_rx=2790
finals_tx=2790
reports_rx=2742
timeouts=71
scheduled_tx_misses=0
reinitializations=0
```

With `drone2` as address 0 and initiator:

```text
30.028 s: 2757 ranges, 91.8 Hz
polls_tx=2834
responses_rx=2781
finals_tx=2781
reports_rx=2757
timeouts=51
scheduled_tx_misses=0
reinitializations=0
```

Both directions pass the 90 Hz close-pair completion gate. No invalid,
wrong-peer, or delayed-transmit-mismatch event occurred. These runs establish
functional bidirectional DS-TWR between these two modules. They do not
establish surveyed range accuracy or the behavior of the other two installed
DW1000 modules.

## Bench work required

1. Extend both physical-role runs to 60 seconds.
2. Verify 9–11 completed ranges/s while far.
3. Run three reachable nodes without synchronized host clocks.
4. Remove one configured peer and verify that the remaining live pair
   continues.
5. Repeat the direct backend under full CM5 load with `SCHED_FIFO` priority 80
   and a reserved CPU.
6. Repeat the direct-pair matrix with the other two installed DW1000 modules.

Record per-leg transmit and receive counters. Do not infer a failure location
from one aggregate completion counter.
