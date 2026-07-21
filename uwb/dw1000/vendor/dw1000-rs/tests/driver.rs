#![allow(missing_docs)]

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::convert::Infallible;
use std::sync::Mutex;

use dw1000_rs::{
    AddressConfig, AntennaDelay, Channel, DataRate, DeviceIdentity, Dw1000, DwTime, Eui64, PanId,
    PreambleLength, PulseFrequency, RadioConfig, RxError, RxOptions, ShortAddress, SysStatus,
    TxOptions,
};
use embedded_hal::digital::{ErrorType as DigitalErrorType, InputPin, OutputPin};
use embedded_hal::spi::{
    Error as SpiErrorTrait, ErrorKind as SpiErrorKind, ErrorType, Operation, SpiDevice,
};

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
struct RecordingSpi {
    log: Arc<Mutex<Vec<LoggedTransaction>>>,
    reads: VecDeque<Vec<u8>>,
}

impl RecordingSpi {
    fn new(log: Arc<Mutex<Vec<LoggedTransaction>>>, reads: VecDeque<Vec<u8>>) -> Self {
        Self { log, reads }
    }
}

impl ErrorType for RecordingSpi {
    type Error = MockSpiError;
}

impl SpiDevice<u8> for RecordingSpi {
    fn transaction(&mut self, operations: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
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
        ShortAddress::new(0x1234),
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
fn reconfigure_programs_identity_registers() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::from(vec![vec![0; 4]]));
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    driver.reconfigure(&config()).unwrap();

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
fn transmit_writes_buffer_control_and_start_bits() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver
        .transmit(&[0xAA, 0xBB, 0xCC], TxOptions::default())
        .unwrap();

    let transactions = log.lock().unwrap();
    assert!(transactions
        .iter()
        .any(|transaction| transaction.writes == vec![vec![0x89], vec![0xAA, 0xBB, 0xCC]]));
    assert!(transactions
        .iter()
        .any(|transaction| transaction.writes == vec![vec![0x88], vec![5, 0, 0, 0, 0]]));
    assert!(transactions
        .iter()
        .any(|transaction| transaction.writes == vec![vec![0x8D], vec![0x02, 0, 0, 0]]));
}

#[test]
fn read_sys_status_decodes_bits() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log, VecDeque::from(vec![vec![0x80, 0x40, 0, 0, 0]]));
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    let status = driver.read_sys_status().unwrap();
    assert_eq!(status, SysStatus((1 << 7) | (1 << 14)));
}

#[test]
fn read_frame_rejects_non_rx_status() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(
        log,
        VecDeque::from(vec![
            vec![0; 4],
            dw1000_rs::registers::sys_status_to_bytes(dw1000_rs::registers::status::TX_FRAME_SENT)
                .to_vec(),
        ]),
    );
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    driver.reconfigure(&config()).unwrap();

    let error = driver.read_frame(&mut [0u8; 16]).unwrap_err();
    assert_eq!(error, dw1000_rs::Error::Receive(RxError::FrameNotReady));
}

#[test]
fn read_signal_metrics_before_config_returns_not_configured() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log, VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    let error = driver.read_signal_metrics().unwrap_err();
    assert_eq!(error, dw1000_rs::Error::NotConfigured);
}

#[test]
fn clear_tx_sent_restarts_receive_when_permanent_mode_is_enabled() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver
        .start_receive(RxOptions {
            delayed_time: None,
            permanent: true,
        })
        .unwrap();
    driver
        .transmit(&[0xAA, 0xBB, 0xCC], TxOptions::default())
        .unwrap();
    driver
        .clear_events(dw1000_rs::registers::status::TX_FRAME_SENT)
        .unwrap();

    let transactions = log.lock().unwrap();
    let restart_count = transactions
        .iter()
        .filter(|transaction| transaction.writes == vec![vec![0x8D], vec![0x00, 0x01, 0x00, 0x00]])
        .count();
    assert_eq!(restart_count, 2);
}

#[test]
fn clear_rx_ready_restarts_receive_when_permanent_mode_is_enabled() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver
        .start_receive(RxOptions {
            delayed_time: None,
            permanent: true,
        })
        .unwrap();
    driver
        .clear_events(dw1000_rs::registers::status::RX_FRAME_READY)
        .unwrap();

    let transactions = log.lock().unwrap();
    let restart_count = transactions
        .iter()
        .filter(|transaction| transaction.writes == vec![vec![0x8D], vec![0x00, 0x01, 0x00, 0x00]])
        .count();
    assert_eq!(restart_count, 2);
}

#[test]
fn clearing_rx_event_does_not_cancel_pending_delayed_tx() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver
        .start_receive(RxOptions {
            delayed_time: None,
            permanent: true,
        })
        .unwrap();
    driver
        .transmit(
            &[0xAA, 0xBB, 0xCC],
            TxOptions {
                delayed_time: Some(dw1000_rs::DwTime::from_micros(7000.0)),
                wait_for_response: false,
            },
        )
        .unwrap();
    driver
        .clear_events(dw1000_rs::registers::status::RX_TIMEOUT)
        .unwrap();

    let transactions = log.lock().unwrap();
    let restart_count = transactions
        .iter()
        .filter(|transaction| transaction.writes == vec![vec![0x8D], vec![0x00, 0x01, 0x00, 0x00]])
        .count();
    assert_eq!(restart_count, 1);
}

#[test]
fn start_receive_clears_stale_rx_timeout_status() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver
        .start_receive(RxOptions {
            delayed_time: None,
            permanent: true,
        })
        .unwrap();

    let transactions = log.lock().unwrap();
    assert!(transactions.iter().any(|transaction| {
        transaction.writes == vec![vec![0x8F], vec![0x00, 0xF4, 0x07, 0x00, 0x00]]
    }));
}

#[test]
fn transmit_at_removes_antenna_delay_from_dx_time() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::from(vec![vec![0; 4]]));
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    driver.reconfigure(&config()).unwrap();
    log.lock().unwrap().clear();

    let register_ticks = 0x12_3456_0000_i64;
    let on_air_ticks = register_ticks + i64::from(AntennaDelay::LEGACY_DEFAULT.raw());
    driver
        .transmit_at(&[0xAA], DwTime::from_ticks(on_air_ticks), true)
        .unwrap();

    let expected = DwTime::from_ticks(register_ticks).to_bytes().to_vec();
    assert!(log.lock().unwrap().iter().any(|transaction| {
        transaction.writes.len() == 2
            && transaction.writes[0] == [0x8A]
            && transaction.writes[1] == expected
    }));
}

#[test]
fn reads_the_device_identifier_register() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log, VecDeque::from(vec![vec![0x30, 0x01, 0xCA, 0xDE]]));
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    assert_eq!(driver.read_device_id().unwrap(), 0xDECA_0130);
}
