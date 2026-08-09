# dw1000-rs

`dw1000-rs` is a `no_std` Rust driver for the Decawave DW1000 UWB transceiver.

It targets `embedded-hal` 1.0 and provides:

- a blocking driver: `Dw1000`
- an async driver: `AsyncDw1000`
- typed PHY/address configuration helpers
- an Embassy example for STM32L432 hardware

## Features

- `no_std` crate suitable for embedded targets
- blocking SPI support via `embedded-hal`
- async SPI support via `embedded-hal-async`
- optional `defmt` formatting support
- basic receive metrics and timestamp access

## Add To Your Project

```toml
[dependencies]
dw1000-rs = "0.1.0"
```

Import it as `dw1000_rs` in code.

## Examples

These snippets show the core crate APIs. They omit board-specific SPI/GPIO setup.

### Blocking Driver

```rust
use dw1000_rs::{
    DeviceIdentity, Dw1000, Error, Eui64, OperatingMode, PanId, RadioConfig, RxOptions,
    ShortAddress, TxOptions,
};
use embedded_hal::delay::DelayNs;
use embedded_hal::digital::{InputPin, OutputPin};
use embedded_hal::spi::SpiDevice;

fn bring_up_radio<SPI, IRQ, RST, PinE>(
    spi: SPI,
    irq: IRQ,
    reset: RST,
    delay: &mut impl DelayNs,
) -> Result<Dw1000<SPI, IRQ, RST>, Error<SPI::Error, PinE>>
where
    SPI: SpiDevice,
    IRQ: InputPin<Error = PinE>,
    RST: OutputPin<Error = PinE>,
{
    let identity = DeviceIdentity::new(
        PanId::new(0xDECA),
        ShortAddress::new(0x1234),
        Eui64::new([0x82, 0x17, 0x5B, 0xD5, 0xA9, 0x9A, 0xE2, 0x9C]),
    );
    let config = RadioConfig::from_mode(identity, OperatingMode::LongDataRangeAccuracy);

    let mut radio = Dw1000::new(spi, irq, reset);
    radio.init(delay, &config)?;
    radio.start_receive(RxOptions {
        delayed_time: None,
        permanent: true,
    })?;
    radio.transmit(b"ping", TxOptions::default())?;

    Ok(radio)
}
```

When an RX interrupt fires, read the frame into a user buffer:

```rust
let mut buffer = [0u8; 127];
let frame = radio.read_frame(&mut buffer)?;
let timestamps = radio.read_timestamps()?;
```

### Async Driver

`AsyncDw1000` mirrors the blocking API for `embedded-hal-async` projects:

```rust
use dw1000_rs::{AsyncDw1000, RadioConfig, RxOptions, TxOptions};

radio.init(delay, &config).await?;
radio.start_receive(RxOptions {
    delayed_time: None,
    permanent: true,
})
.await?;
radio.transmit(b"ping", TxOptions::default()).await?;
let frame = radio.read_frame(&mut buffer).await?;
```

## Example Firmware

The repository includes an Embassy example for STM32L432 in [`examples/embassy-stm32l432`](examples/embassy-stm32l432).

More detail, including wiring, flashing commands, and default addresses, is in [`examples/README.md`](examples/README.md).

## Development

```bash
cargo test
cargo lint
cargo lint-embassy
```

The root crate is covered by unit tests, and the STM32L432 Embassy example is checked through the `lint-embassy` Cargo alias from [`.cargo/config.toml`](.cargo/config.toml).

## License

Apache-2.0
