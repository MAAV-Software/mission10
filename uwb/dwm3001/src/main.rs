#![no_std]
#![no_main]

mod board;
mod host;
mod ranging;

use defmt::info;
use embassy_executor::Spawner;
use embassy_futures::join::join3;
use embassy_nrf::usb::Driver;
use embassy_nrf::usb::vbus_detect::HardwareVbusDetect;
use embassy_nrf::{bind_interrupts, pac, peripherals, spim, usb};
#[cfg(feature = "smoke")]
use embassy_time::Timer;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State as CdcState};
use embassy_usb::{Builder, Config as UsbConfig};
use {defmt_rtt as _, panic_probe as _};

use board::RadioHardware;

bind_interrupts!(pub struct Irqs {
    SPIM3 => spim::InterruptHandler<peripherals::SPI3>;
    USBD => usb::InterruptHandler<peripherals::USBD>;
    CLOCK_POWER => usb::vbus_detect::InterruptHandler;
});

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_nrf::init(Default::default());
    info!("mission10 boot");

    #[cfg(feature = "smoke")]
    loop {
        Timer::after_secs(1).await;
    }

    // Native USB requires the external 32 MHz crystal.
    pac::CLOCK.tasks_hfclkstart().write_value(1);
    while pac::CLOCK.events_hfclkstarted().read() != 1 {}

    let usb_driver = Driver::new(p.USBD, Irqs, HardwareVbusDetect::new(Irqs));
    let mut usb_config = UsbConfig::new(0x2fe3, 0x0100);
    usb_config.manufacturer = Some("MAAV");
    usb_config.product = Some("Mission 10 DWM3001 bring-up");
    usb_config.serial_number = Some("DWM3001-01");
    usb_config.max_power = 100;
    usb_config.max_packet_size_0 = 64;

    let mut config_descriptor = [0; 256];
    let mut bos_descriptor = [0; 256];
    let mut msos_descriptor = [0; 256];
    let mut control_buf = [0; 64];
    let mut cdc_state = CdcState::new();
    let mut builder = Builder::new(
        usb_driver,
        usb_config,
        &mut config_descriptor,
        &mut bos_descriptor,
        &mut msos_descriptor,
        &mut control_buf,
    );
    let cdc = CdcAcmClass::new(&mut builder, &mut cdc_state, 64);
    let mut usb_device = builder.build();

    let radio = RadioHardware {
        spi: p.SPI3,
        sck: p.P0_03,
        miso: p.P0_29,
        mosi: p.P0_08,
        cs: p.P1_06,
        reset: p.P0_25,
        irq: p.P1_02,
    };

    join3(usb_device.run(), host::run(cdc), ranging::run(radio)).await;
}
