# Installed DW1000 regression matrix — 2026-07-26

## Scope

This record covers the two installed DW1000 modules that were unavailable
during the pair-local implementation test. All ranging processes used commit
`a0b91fe`, `SCHED_FIFO` priority 80, and CPU 3. Distances were not surveyed.

`drone1` initially lacked membership in the `spi` and `gpio` groups. The user
was added to both groups before testing. Both radios then returned device ID
`0xdeca0130` at 20 MHz SPI.

## `drone0` and `drone1`

With `drone0` as initiator, 45 of 154 Poll attempts completed in 30.036
seconds. `drone1` received 102 Polls but only 45 Finals.

With `drone1` as initiator, 88 of 185 Poll attempts completed in 30.037
seconds. `drone0` received 174 Polls and sent 174 Responses, while `drone1`
received 108 Responses. Once `drone0` received a Final, 107 exchanges
completed.

The pair is functional, but these rates do not pass the far- or close-pair
gate.

## Known-good cross-checks

`drone0` ranged with the previously qualified DW1000 on `bigrpi5`:

```text
bigrpi5 initiator: 182/193 completed attempts, 9.1 Hz
drone0 initiator:   178/192 completed attempts, 8.9 Hz
```

The rates are consistent with far-pair service. There were no invalid frames,
wrong-peer frames, delayed-transmit mismatches, or reinitializations.

`drone1` ranged with the same `bigrpi5` module:

```text
bigrpi5 initiator: 230/584 completed attempts, 11.5 Hz
drone1 initiator:  538/1015 completed attempts, 26.9 Hz
```

When `drone1` initiated, `bigrpi5` received 1004 Polls and 680 Finals.
`drone1` received only 686 Responses and 538 Reports. This result indicates a
healthy `drone1` transmit path and a lossy `drone1` receive path.

Reducing transmit power by 8.5 dB did not improve the result. The initiator
completed 225 of 564 attempts at 15.0 Hz. Close-range saturation is therefore
not the leading theory.

## Receive-oracle comparison

The same `bigrpi5` sender stimulated each raw receive oracle. The trials were
not a calibrated RF comparison because node positions differed. The
receive-pipeline progression is still diagnostic.

`drone1`:

```text
preambles=40 sfds=33 headers=23 fcs_good=19
header_errors=10 rs_errors=4 sfd_timeouts=7
clock_pll_loss=true preamble_rejection=true
```

`drone0` control:

```text
preambles=23 sfds=20 headers=19 fcs_good=19
header_errors=1 rs_errors=0 sfd_timeouts=3
clock_pll_loss=false preamble_rejection=true
```

`drone1` loses frames after preamble detection at the SFD, PHY-header, and
Reed-Solomon stages. The clock-PLL-loss latch is unique to this comparison.
Inspect the `drone1` module power, ground, crystal/clock environment, antenna
connection, and soldering before treating it as a flight radio. Repeat the
oracle comparison with fixed placement after any hardware change.

## Disposition

- `drone0`: passes the functional bidirectional direct-DW1000 check.
- `drone1`: enumerates and transmits, but does not pass receive qualification.
- Multi-node scheduler testing should use the qualified radios until
  `drone1` is repaired or the receive fault is otherwise explained.

## Three-node pair-local test

The qualified direct radios on `drone0`, `drone2`, and `bigrpi5` ran
concurrently as addresses 0, 1, and 2. Every node configured both peers. The
common overlap was 35 seconds.

Per-peer completed exchanges were:

```text
drone0  <-> drone2:  372 / 373
drone2  <-> bigrpi5: 2883 / 2946
drone0  <-> bigrpi5:   58 / 62
```

All three pairs made progress without a coordinator or synchronized host
clocks. The previously measured `drone0` to `bigrpi5` link fell from about
9 Hz to about 1.7 Hz while the close `drone2` to `bigrpi5` pair completed at
more than 80 Hz. The current pair-local scheduler therefore does not preserve
the intended far-pair floor under close-pair contention.

The run also exposed an observability defect. Frames addressed to another node
increment the `invalid` counter. `drone0` recorded 9463 invalid frames while
overhearing the busy pair. A valid frame for another destination is expected
medium traffic and must not be classified as malformed input.

### Unsurveyed ranges

A second short three-node run retained the computed distance instead of only
counting completed exchanges:

```text
drone0  <-> drone2:  mean=0.802 m min=0.755 m max=0.930 m
drone2  <-> bigrpi5: mean=2.677 m min=2.583 m max=2.761 m
drone0  <-> bigrpi5: mean=3.441 m min=3.315 m max=3.568 m
```

Both endpoints reported identical aggregates for each pair. These distances
are not calibrated or surveyed. They explain the scheduling state: the first
two pairs are below the 3.0 m close-entry threshold, while the last pair lies
around the 3.5 m close-exit threshold and remained on its far cadence.

## Missing-peer control

`drone0` retained peers 1 and 2 while no process ran at address 1. Address 2
remained online. The live pair completed 168 exchanges in 20 seconds at the
initiator and 174 during the responder's longer window. There were no
reinitializations, invalid frames, wrong-peer frames, or delayed-transmit
mismatches at the initiator.

A missing configured peer reduces aggregate opportunity use through failed
attempts and backoff, but it does not suppress the live pair. The multi-node
rate loss above is contention behavior rather than a dead-peer dependency.
