# Mission 10 patch notes

This directory is the crates.io `dw1000-rs` 0.2.0 source, upstream commit
`08d57d9969df4a385f234cf4cd6ea1cefd718455`, retained under its declared
Apache-2.0 license.

Mission 10 adds three blocking-driver entry points:

- `read_device_id`, so Linux bring-up can verify that SPI reaches a DW1000;
- `transmit_at`, which programs a previously computed absolute delayed-TX
  timestamp;
- `read_frame_raw`, which retains the hardware RX timestamp for protocols that
  perform calibration after combining timestamps from both radios.

The second entry point is required for DS-TWR. Upstream 0.2.0 computes the
timestamp inserted into a RANGE payload and then computes a later timestamp
when `transmit` writes `DX_TIME`. Even a small difference corrupts the time of
flight. The upstream helper also returns an antenna-adjusted on-air timestamp,
while `DX_TIME` requires the unadjusted deadline; `transmit_at` removes TX_ANTD
before writing the register. The ordinary relative-delay `transmit` API retains
its behavior and now delegates to `transmit_at`.

Keep this copy byte-for-byte aligned with 0.2.0 outside `src/dw1000.rs`, the
`DevId` register declaration, their driver tests, this note, and removal of the
package-local profiles now supplied by the enclosing workspace. Replace it with
an upstream release once equivalent APIs are available.
