# Mission 10 patch notes

This directory is the crates.io `dw1000-rs` 0.2.0 source, upstream commit
`08d57d9969df4a385f234cf4cd6ea1cefd718455`, retained under its declared
Apache-2.0 license.

Mission 10 adds these driver interfaces:

- `read_device_id`, so Linux bring-up can verify that SPI reaches a DW1000;
- `schedule_delayed_transmit` and `schedule_delayed_receive`, which distinguish
  the raw `DX_TIME` deadline from the antenna-adjusted RMARKER timestamp;
- `transmit_at`, which accepts one checked `DelayedTransmit` schedule;
- `read_frame_raw`, which retains the hardware RX timestamp for protocols that
  perform calibration after combining timestamps from both radios;
- `reset_receiver`, which applies the DW1000 receiver-only soft-reset sequence
  without discarding the configured PHY and address;
- `read_debug_state`, which captures the bounded register set used to diagnose
  receiver state-machine failures;
- `set_transmit_power`, which permits controlled close-range saturation tests.

Mission 10 removes the upstream tag/anchor protocol and ranging state machine.
The mission application owns its air protocol, schedule, and recovery policy.
Keeping a second coordinator-based protocol in the hardware driver made
ownership ambiguous and added an unused public interface.

The delayed-transmit entry point is required for DS-TWR. Upstream 0.2.0 computes
the timestamp inserted into a Range payload and then computes a later timestamp
when `transmit` writes `DX_TIME`. Even a small difference corrupts the time of
flight. `DxTimeDeadline`, `OnAirTimestamp`, and `DelayedTransmit` make the two
time domains explicit. Delayed RX programs only a raw deadline. Blocking and
async frontends expose the same absolute-TX and raw-RX behavior.

Initialization additionally follows Decawave's reference
sequence for crystal-clock OTP reads, PLL-lock detection, programmed LDO tune,
and AON cleanup. Receiver tuning derives `DRX_SFDTOC` from the configured
preamble, SFD, and PAC. Explicit receive uses the reference force-off,
TX/RX-status-clear, and host/IC buffer-pointer synchronization sequence. The
public status groups distinguish terminal RX events from every latch that one
receive attempt must clear; this prevents an asserted `RX_FRAME_GOOD` interrupt
from hiding the next edge. The interrupt mask enables only events owned by the
TX or RX service; automatic-acknowledgement trigger interrupts remain disabled.

Keep this copy aligned with 0.2.0 outside these documented additions, their
tests, the removed ranging protocol and state machine, this note, and removal
of the package-local profiles now supplied by the enclosing workspace. Replace
it with an upstream release once equivalent APIs are available.
