#![allow(missing_docs)]

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::convert::Infallible;
use std::sync::Mutex;

use dw1000_rs::{
    AddressConfig, AntennaDelay, AsyncDw1000, Channel, DataRate, DeviceIdentity, DwTime, Eui64,
    PanId, PreambleLength, PulseFrequency, RadioConfig, RxError, RxOptions, SysStatus, TxOptions,
};
use embedded_hal::digital::{ErrorType as DigitalErrorType, InputPin, OutputPin};
use embedded_hal::spi::{Error as SpiErrorTrait, ErrorKind as SpiErrorKind, ErrorType, Operation};
use embedded_hal_async::{digital::Wait, spi::SpiDevice};
use futures::executor::block_on;

#[derive(Debug, Clone, PartialEq, Eq)]
struct MockSpiError;

impl SpiErrorTrait for MockSpiError {
    fn kind(&self) -> SpiErrorKind {
        SpiErrorKind::Other
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoggedTransaction {
    writes: Vec<Vec<u8>>,
    reads: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct RecordingAsyncSpi {
    log: Arc<Mutex<Vec<LoggedTransaction>>>,
    reads: VecDeque<Vec<u8>>,
}

impl RecordingAsyncSpi {
    fn new(log: Arc<Mutex<Vec<LoggedTransaction>>>, reads: VecDeque<Vec<u8>>) -> Self {
        Self { log, reads }
    }
}

impl ErrorType for RecordingAsyncSpi {
    type Error = MockSpiError;
}

impl SpiDevice<u8> for RecordingAsyncSpi {
    async fn transaction(
        &mut self,
        operations: &mut [Operation<'_, u8>],
    ) -> Result<(), Self::Error> {
        let mut transaction = LoggedTransaction {
            writes: Vec::new(),
            reads: Vec::new(),
        };
        for operation in operations {
            match operation {
                Operation::Write(data) => transaction.writes.push(data.to_vec()),
                Operation::Read(buffer) => {
                    let response = self
                        .reads
                        .pop_front()
                        .unwrap_or_else(|| vec![0; buffer.len()]);
                    assert_eq!(response.len(), buffer.len());
                    buffer.copy_from_slice(&response);
                    transaction.reads.push(response);
                }
                Operation::Transfer(read, write) => {
                    let response = self
                        .reads
                        .pop_front()
                        .unwrap_or_else(|| vec![0; read.len()]);
                    assert_eq!(response.len(), read.len());
                    read.copy_from_slice(&response);
                    transaction.writes.push(write.to_vec());
                    transaction.reads.push(response);
                }
                Operation::TransferInPlace(buffer) => {
                    let response = self
                        .reads
                        .pop_front()
                        .unwrap_or_else(|| vec![0; buffer.len()]);
                    assert_eq!(response.len(), buffer.len());
                    buffer.copy_from_slice(&response);
                    transaction.reads.push(response);
                }
                Operation::DelayNs(_) => {}
            }
        }
        self.log.lock().unwrap().push(transaction);
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
struct MockInputPin;

impl DigitalErrorType for MockInputPin {
    type Error = Infallible;
}

impl InputPin for MockInputPin {
    fn is_high(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }

    fn is_low(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

impl Wait for MockInputPin {
    async fn wait_for_high(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn wait_for_low(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn wait_for_rising_edge(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn wait_for_falling_edge(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn wait_for_any_edge(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
struct MockOutputPin;

impl DigitalErrorType for MockOutputPin {
    type Error = Infallible;
}

impl OutputPin for MockOutputPin {
    fn set_low(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn set_high(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn identity() -> DeviceIdentity {
    DeviceIdentity::new(
        PanId::new(0xDECA),
        dw1000_rs::ShortAddress::new(0x1234),
        Eui64::new([0x82, 0x17, 0x5B, 0xD5, 0xA9, 0x9A, 0xE2, 0x9C]),
    )
}

fn config() -> RadioConfig {
    RadioConfig {
        address: AddressConfig {
            identity: identity(),
        },
        phy: dw1000_rs::PhyConfig {
            data_rate: DataRate::Kbps110,
            pulse_frequency: PulseFrequency::Mhz16,
            preamble_length: PreambleLength::Symbols2048,
            channel: Channel::Channel5,
            preamble_code: None,
            smart_power: false,
        },
        antenna_delay: AntennaDelay::LEGACY_DEFAULT,
        receiver_auto_reenable: true,
        interrupt_polarity_high: true,
        frame_check: true,
    }
}

#[test]
fn async_reconfigure_programs_identity_registers() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log.clone(), VecDeque::from(vec![vec![0; 4]]));
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.reconfigure(&config())).unwrap();

    let transactions = log.lock().unwrap();
    assert!(transactions.iter().any(|transaction| {
        transaction.writes
            == vec![
                vec![0x81],
                vec![0x9C, 0xE2, 0x9A, 0xA9, 0xD5, 0x5B, 0x17, 0x82],
            ]
    }));
    assert!(transactions.iter().any(|transaction| {
        transaction.writes == vec![vec![0x83], vec![0x34, 0x12, 0xCA, 0xDE]]
    }));
}

#[test]
fn async_compute_delayed_time_aligns_timestamp_and_applies_antenna_delay() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(
        log,
        VecDeque::from(vec![vec![0x34, 0x12, 0x00, 0x00, 0x00]]),
    );
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    let delay = DwTime::from_micros(7000.0);
    let computed = block_on(driver.compute_delayed_time(delay)).unwrap();

    let mut expected = DwTime::from_bytes(&[0x34, 0x12, 0x00, 0x00, 0x00]) + delay;
    let mut bytes = expected.to_bytes();
    bytes[0] = 0;
    bytes[1] &= 0xFE;
    expected = DwTime::from_bytes(&bytes) + DwTime::from_ticks(16_456);
    assert_eq!(computed, expected);
}

#[test]
fn async_read_frame_reads_payload_and_metrics() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let reads = VecDeque::from(vec![
        vec![0; 4],
        dw1000_rs::registers::sys_status_to_bytes(SysStatus(
            dw1000_rs::registers::status::RX_FRAME_READY.0
                | dw1000_rs::registers::status::RX_FRAME_GOOD.0,
        ))
        .to_vec(),
        vec![0x07, 0x00, 0x00, 0x00],
        vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE],
        vec![0x10, 0x00, 0x00, 0x00, 0x00],
        vec![0x20, 0x00],
        vec![0x40, 0x00, 0x00, 0x00],
        vec![0x02, 0x00],
        vec![0x03, 0x00],
        vec![0x04, 0x00],
        vec![0x40, 0x00, 0x00, 0x00],
        vec![0x02, 0x00],
        vec![0x03, 0x00],
    ]);
    let spi = RecordingAsyncSpi::new(log, reads);
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.reconfigure(&config())).unwrap();
    let mut buffer = [0u8; 16];
    let frame = block_on(driver.read_frame(&mut buffer)).unwrap();

    assert_eq!(frame.bytes, &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);
    assert!(frame.metrics.receive_power_dbm.is_finite());
    assert!(frame.metrics.first_path_power_dbm.is_finite());
    assert!(frame.metrics.quality > 0.0);
}

#[test]
fn async_read_frame_rejects_non_rx_status() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let reads = VecDeque::from(vec![
        vec![0; 4],
        dw1000_rs::registers::sys_status_to_bytes(dw1000_rs::registers::status::TX_FRAME_SENT)
            .to_vec(),
    ]);
    let spi = RecordingAsyncSpi::new(log, reads);
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.reconfigure(&config())).unwrap();
    let error = block_on(driver.read_frame(&mut [0u8; 16])).unwrap_err();
    assert_eq!(error, dw1000_rs::Error::Receive(RxError::FrameNotReady));
}

#[test]
fn async_read_signal_metrics_before_config_returns_not_configured() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log, VecDeque::new());
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    let error = block_on(driver.read_signal_metrics()).unwrap_err();
    assert_eq!(error, dw1000_rs::Error::NotConfigured);
}

#[test]
fn async_clear_tx_sent_restarts_receive_when_permanent_mode_is_enabled() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log.clone(), VecDeque::new());
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.start_receive(RxOptions {
        delayed_time: None,
        permanent: true,
    }))
    .unwrap();
    block_on(driver.transmit(&[0xAA, 0xBB, 0xCC], TxOptions::default())).unwrap();
    block_on(driver.clear_events(dw1000_rs::registers::status::TX_FRAME_SENT)).unwrap();

    let transactions = log.lock().unwrap();
    let restart_count = transactions
        .iter()
        .filter(|transaction| transaction.writes == vec![vec![0x8D], vec![0x00, 0x01, 0x00, 0x00]])
        .count();
    assert_eq!(restart_count, 2);
}

#[test]
fn async_clear_rx_ready_restarts_receive_when_permanent_mode_is_enabled() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log.clone(), VecDeque::new());
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.start_receive(RxOptions {
        delayed_time: None,
        permanent: true,
    }))
    .unwrap();
    block_on(driver.clear_events(dw1000_rs::registers::status::RX_FRAME_READY)).unwrap();

    let transactions = log.lock().unwrap();
    let restart_count = transactions
        .iter()
        .filter(|transaction| transaction.writes == vec![vec![0x8D], vec![0x00, 0x01, 0x00, 0x00]])
        .count();
    assert_eq!(restart_count, 2);
}
