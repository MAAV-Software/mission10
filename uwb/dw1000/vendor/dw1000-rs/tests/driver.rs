#![allow(missing_docs)]

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::convert::Infallible;
use std::sync::Mutex;

use dw1000_rs::{
    AddressConfig, AntennaDelay, Channel, DataRate, DelayedTransmit, DeviceIdentity, Dw1000,
    DwTime, DxTimeDeadline, Eui64, OnAirTimestamp, PanId, PreambleLength, PulseFrequency,
    RadioConfig, RxError, RxOptions, ShortAddress, SysStatus, TxOptions,
};
use embedded_hal::delay::DelayNs;
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

#[derive(Debug, Default)]
struct MockDelay;

impl DelayNs for MockDelay {
    fn delay_ns(&mut self, _ns: u32) {}
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
    assert!(transactions
        .iter()
        .any(|transaction| { transaction.writes == vec![vec![0xE7, 0x20], vec![0x01, 0x08]] }));
}

#[test]
fn initialization_applies_programmed_otp_tuning() {
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
    let spi = RecordingSpi::new(log.clone(), reads);
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver.init(&mut MockDelay, &config()).unwrap();

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
        VecDeque::from(vec![dw1000_rs::registers::sys_status_to_bytes(
            dw1000_rs::registers::status::TX_FRAME_SENT,
        )
        .to_vec()]),
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
fn start_receive_clears_all_reference_tx_rx_status() {
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
        transaction.writes == vec![vec![0x8F], vec![0xF8, 0xFF, 0x27, 0x24]]
    }));
}

#[test]
fn start_receive_synchronizes_mismatched_buffer_pointers() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::from(vec![vec![0x80]]));
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver
        .start_receive(RxOptions {
            delayed_time: None,
            permanent: true,
        })
        .unwrap();

    let transactions = log.lock().unwrap();
    assert!(transactions
        .iter()
        .any(|transaction| { transaction.writes == vec![vec![0xCD, 0x03], vec![0x01]] }));
}

#[test]
fn transmit_at_programs_the_schedule_deadline() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::from(vec![vec![0; 4]]));
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    driver.reconfigure(&config()).unwrap();
    log.lock().unwrap().clear();

    let register_ticks = 0x12_3456_0000_i64;
    let on_air_ticks = register_ticks + i64::from(AntennaDelay::LEGACY_DEFAULT.raw());
    let schedule = DelayedTransmit::new(
        DxTimeDeadline::new(DwTime::from_ticks(register_ticks)),
        OnAirTimestamp::new(DwTime::from_ticks(on_air_ticks)),
    );
    driver.transmit_at(&[0xAA], schedule, true).unwrap();

    let expected = DwTime::from_ticks(register_ticks).to_bytes().to_vec();
    assert!(log.lock().unwrap().iter().any(|transaction| {
        transaction.writes.len() == 2
            && transaction.writes[0] == [0x8A]
            && transaction.writes[1] == expected
    }));
}

#[test]
fn delayed_transmit_separates_deadline_and_on_air_timestamp() {
    let spi = RecordingSpi::new(
        Arc::new(Mutex::new(Vec::new())),
        VecDeque::from(vec![vec![0x34, 0x12, 0x00, 0x00, 0x00]]),
    );
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    let delay = DwTime::from_micros(7000.0);

    let schedule = driver.schedule_delayed_transmit(delay).unwrap();

    let expected = DwTime::from_bytes(&[0x34, 0x12, 0x00, 0x00, 0x00]) + delay;
    let mut bytes = expected.to_bytes();
    bytes[0] = 0;
    bytes[1] &= 0xfe;
    let deadline = DwTime::from_bytes(&bytes);
    assert_eq!(schedule.deadline().value(), deadline);
    assert_eq!(
        schedule.timestamp().value(),
        deadline + DwTime::from_ticks(i64::from(AntennaDelay::LEGACY_DEFAULT.raw()))
    );
}

#[test]
fn delayed_receive_programs_the_raw_deadline() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    let delay = DwTime::from_micros(7000.0);

    driver
        .start_receive(RxOptions {
            delayed_time: Some(delay),
            permanent: false,
        })
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
fn reads_the_device_identifier_register() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log, VecDeque::from(vec![vec![0x30, 0x01, 0xCA, 0xDE]]));
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);
    assert_eq!(driver.read_device_id().unwrap(), 0xDECA_0130);
}

#[test]
fn overrides_the_raw_transmit_power() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver.set_transmit_power(0xC0C0_C0C0).unwrap();

    assert_eq!(
        log.lock().unwrap()[0].writes,
        vec![vec![0x9E], vec![0xC0, 0xC0, 0xC0, 0xC0]]
    );
}

#[test]
fn receiver_reset_uses_the_receiver_only_soft_reset_sequence() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let spi = RecordingSpi::new(log.clone(), VecDeque::new());
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    driver.reset_receiver().unwrap();

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
fn reads_a_radio_debug_state_snapshot() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let reads = VecDeque::from(vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14],
        vec![15, 16, 17, 18],
        vec![19, 20, 21, 22],
        vec![23, 24, 25, 26],
    ]);
    let spi = RecordingSpi::new(log, reads);
    let mut driver = Dw1000::new(spi, MockInputPin, MockOutputPin);

    let state = driver.read_debug_state().unwrap();

    assert_eq!(state.sys_status, SysStatus(0x05_0403_0201));
    assert_eq!(state.sys_state, 0x0A_0908_0706);
    assert_eq!(state.sys_ctrl, 0x0E0D_0C0B);
    assert_eq!(state.sys_cfg, 0x1211_100F);
    assert_eq!(state.sys_mask, 0x1615_1413);
    assert_eq!(state.pmsc_ctrl0, 0x1A19_1817);
}
