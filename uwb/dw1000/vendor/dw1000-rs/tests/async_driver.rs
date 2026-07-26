#![allow(missing_docs)]

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::convert::Infallible;
use std::sync::Mutex;

use dw1000_rs::{
    AddressConfig, AntennaDelay, AsyncDw1000, Channel, DataRate, DelayedTransmit, DeviceIdentity,
    DwTime, DxTimeDeadline, Eui64, OnAirTimestamp, PanId, PreambleLength, PulseFrequency,
    RadioConfig, RxError, RxOptions, SysStatus, TxOptions,
};
use embedded_hal::digital::{ErrorType as DigitalErrorType, InputPin, OutputPin};
use embedded_hal::spi::{Error as SpiErrorTrait, ErrorKind as SpiErrorKind, ErrorType, Operation};
use embedded_hal_async::{delay::DelayNs, digital::Wait, spi::SpiDevice};
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

#[derive(Debug, Default)]
struct MockDelay;

impl DelayNs for MockDelay {
    async fn delay_ns(&mut self, _ns: u32) {}
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
    assert!(transactions
        .iter()
        .any(|transaction| { transaction.writes == vec![vec![0xE7, 0x20], vec![0x01, 0x08]] }));
}

#[test]
fn async_initialization_applies_programmed_otp_tuning() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let reads = VecDeque::from(vec![
        vec![0; 4],
        vec![0; 4],
        vec![1, 0, 0, 0],
        vec![0x1A, 0, 0, 0],
        vec![0; 4],
        vec![0; 2],
        vec![0; 4],
    ]);
    let spi = RecordingAsyncSpi::new(log.clone(), reads);
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.init(&mut MockDelay, &config())).unwrap();

    let transactions = log.lock().unwrap();
    assert!(transactions
        .iter()
        .any(|transaction| transaction.writes == vec![vec![0xED, 0x12], vec![0x02]]));
    assert!(transactions
        .iter()
        .any(|transaction| transaction.writes == vec![vec![0xEB, 0x0E], vec![0x7A]]));
    assert!(transactions
        .iter()
        .any(|transaction| transaction.writes == vec![vec![0xEC, 0x0A], vec![0x00]]));
}

#[test]
fn async_delayed_transmit_separates_deadline_and_on_air_timestamp() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(
        log,
        VecDeque::from(vec![vec![0x34, 0x12, 0x00, 0x00, 0x00]]),
    );
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    let delay = DwTime::from_micros(7000.0);
    let computed = block_on(driver.schedule_delayed_transmit(delay)).unwrap();

    let expected = DwTime::from_bytes(&[0x34, 0x12, 0x00, 0x00, 0x00]) + delay;
    let mut bytes = expected.to_bytes();
    bytes[0] = 0;
    bytes[1] &= 0xFE;
    let deadline = DwTime::from_bytes(&bytes);
    let timestamp = deadline + DwTime::from_ticks(16_456);
    assert_eq!(computed.deadline().value(), deadline);
    assert_eq!(computed.timestamp().value(), timestamp);
}

#[test]
fn async_delayed_receive_programs_the_raw_deadline() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log.clone(), VecDeque::new());
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);
    let delay = DwTime::from_micros(7000.0);

    block_on(driver.start_receive(RxOptions {
        delayed_time: Some(delay),
        permanent: false,
    }))
    .unwrap();

    let mut bytes = delay.to_bytes();
    bytes[0] = 0;
    bytes[1] &= 0xfe;
    assert!(log.lock().unwrap().iter().any(|transaction| {
        transaction.writes.len() == 2
            && transaction.writes[0] == [0x8a]
            && transaction.writes[1] == bytes
    }));
}

#[test]
fn async_transmit_at_programs_the_schedule_deadline() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log.clone(), VecDeque::new());
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);
    let deadline = DwTime::from_ticks(0x12_3456_0000);
    let schedule = DelayedTransmit::new(
        DxTimeDeadline::new(deadline),
        OnAirTimestamp::new(
            deadline + DwTime::from_ticks(i64::from(AntennaDelay::LEGACY_DEFAULT.raw())),
        ),
    );

    block_on(driver.transmit_at(&[0xaa], schedule, true)).unwrap();

    let expected = deadline.to_bytes();
    assert!(log.lock().unwrap().iter().any(|transaction| {
        transaction.writes.len() == 2
            && transaction.writes[0] == [0x8a]
            && transaction.writes[1] == expected
    }));
}

#[test]
fn async_read_frame_reads_payload_and_metrics() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let reads = VecDeque::from(vec![
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
    let reads = VecDeque::from(vec![dw1000_rs::registers::sys_status_to_bytes(
        dw1000_rs::registers::status::TX_FRAME_SENT,
    )
    .to_vec()]);
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

#[test]
fn async_start_receive_synchronizes_mismatched_buffer_pointers() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log.clone(), VecDeque::from(vec![vec![0x80]]));
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.start_receive(RxOptions {
        delayed_time: None,
        permanent: true,
    }))
    .unwrap();

    let transactions = log.lock().unwrap();
    assert!(transactions
        .iter()
        .any(|transaction| { transaction.writes == vec![vec![0xCD, 0x03], vec![0x01]] }));
}

#[test]
fn async_overrides_the_raw_transmit_power() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log.clone(), VecDeque::new());
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.set_transmit_power(0xC0C0_C0C0)).unwrap();

    assert_eq!(
        log.lock().unwrap()[0].writes,
        vec![vec![0x9E], vec![0xC0, 0xC0, 0xC0, 0xC0]]
    );
}

#[test]
fn async_receiver_reset_uses_the_receiver_only_soft_reset_sequence() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingAsyncSpi::new(log.clone(), VecDeque::new());
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    block_on(driver.reset_receiver()).unwrap();

    let transactions = log.lock().unwrap();
    assert_eq!(
        transactions
            .iter()
            .map(|transaction| transaction.writes.clone())
            .collect::<Vec<_>>(),
        vec![
            vec![vec![0x8E], vec![0x00, 0x00, 0x00, 0x00]],
            vec![vec![0x8D], vec![0x40]],
            vec![vec![0x8F], vec![0xF8, 0xFF, 0x27, 0x24]],
            vec![vec![0x4F, 0x03]],
            vec![vec![0x8E], vec![0x00, 0x00, 0x00, 0x00]],
            vec![vec![0xF6, 0x03], vec![0xE0]],
            vec![vec![0xF6, 0x03], vec![0xF0]],
        ]
    );
}

#[test]
fn async_reads_a_radio_debug_state_snapshot() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let reads = VecDeque::from(vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14],
        vec![15, 16, 17, 18],
        vec![19, 20, 21, 22],
        vec![23, 24, 25, 26],
    ]);
    let spi = RecordingAsyncSpi::new(log, reads);
    let mut driver = AsyncDw1000::new(spi, MockInputPin, MockOutputPin);

    let state = block_on(driver.read_debug_state()).unwrap();

    assert_eq!(state.sys_status, SysStatus(0x05_0403_0201));
    assert_eq!(state.sys_state, 0x0A_0908_0706);
    assert_eq!(state.sys_ctrl, 0x0E0D_0C0B);
    assert_eq!(state.sys_cfg, 0x1211_100F);
    assert_eq!(state.sys_mask, 0x1615_1413);
    assert_eq!(state.pmsc_ctrl0, 0x1A19_1817);
}
