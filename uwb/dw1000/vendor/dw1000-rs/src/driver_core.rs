//! Shared DW1000 driver internals used by the blocking and async frontends.

use libm::{floorf, log10f};

use crate::config::{
    ConfigError, DataRate, PacSize, PreambleCode, PreambleLength, PulseFrequency, RadioConfig,
    ValidatedPhyConfig,
};
use crate::device::{
    AntennaDelay, DelayedTransmit, DeviceIdentity, DxTimeDeadline, OnAirTimestamp, SysStatus,
};
use crate::error::RxError;
use crate::registers::status;
use crate::registers::{
    Register, DIS_DRXB_BIT, DIS_STXP_BIT, DWSFD_BIT, HIRQ_POL_BIT, LEN_RX_FINFO, LEN_SYS_MASK,
    MAFFREJ_BIT, MLDEERR_BIT, MRXDFR_BIT, MRXFCE_BIT, MRXFCG_BIT, MRXFSL_BIT, MRXOVRR_BIT,
    MRXPHE_BIT, MRXPTO_BIT, MRXRFTO_BIT, MRXSFDTO_BIT, MTXFRS_BIT, NO_SUBADDRESS, RNSSFD_BIT,
    RXAUTR_BIT, RXDLYS_BIT, RXENAB_BIT, RXM110K_BIT, SFCST_BIT, TNSSFD_BIT, TRXOFF_BIT, TXDLYS_BIT,
    TXSTRT_BIT, WAIT4RESP_BIT,
};
use crate::time::{DwTime, DISTANCE_PER_TICK_M};

pub(crate) const LEN_UWB_FRAMES: usize = 127;

const RECEIVE_RESTART_EVENTS: u64 = status::RX_TERMINAL_EVENTS.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClockMode {
    Auto,
    Xti,
}

const WRITE: u8 = 0x80;
const WRITE_SUB: u8 = 0xC0;
const READ: u8 = 0x00;
const READ_SUB: u8 = 0x40;
const RW_SUB_EXT: u8 = 0x80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DriverState {
    Idle,
    Rx,
    Tx,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DriverRuntime {
    pub(crate) frame_check: bool,
    pub(crate) permanent_receive: bool,
    pub(crate) state: DriverState,
    pub(crate) antenna_delay: AntennaDelay,
    pub(crate) phy: Option<ValidatedPhyConfig>,
    pub(crate) identity: Option<DeviceIdentity>,
    pub(crate) rx_after_tx_pending: bool,
}

impl DriverRuntime {
    pub(crate) const fn new() -> Self {
        Self {
            frame_check: true,
            permanent_receive: false,
            state: DriverState::Idle,
            antenna_delay: AntennaDelay::LEGACY_DEFAULT,
            phy: None,
            identity: None,
            rx_after_tx_pending: false,
        }
    }

    pub(crate) fn reconfigure(
        &mut self,
        config: &RadioConfig,
    ) -> Result<ValidatedPhyConfig, ConfigError> {
        let phy = config.validated_phy()?;
        self.identity = Some(config.address.identity);
        self.phy = Some(phy);
        self.antenna_delay = config.antenna_delay;
        self.frame_check = config.frame_check;
        self.permanent_receive = false;
        self.state = DriverState::Idle;
        self.rx_after_tx_pending = false;
        Ok(phy)
    }

    pub(crate) const fn identity(&self) -> Option<DeviceIdentity> {
        self.identity
    }

    pub(crate) fn validate_rx_status(&self, status: SysStatus) -> Result<(), RxError> {
        if status.contains(status::LDE_ERROR) {
            return Err(RxError::LeadingEdgeDetection);
        }
        if status.contains(status::RX_FRAME_CHECK_ERROR) {
            return Err(RxError::FrameCheck);
        }
        if status.contains(status::RX_HEADER_ERROR) {
            return Err(RxError::Header);
        }
        if status.contains(status::RX_REED_SOLOMON_ERROR) {
            return Err(RxError::ReedSolomon);
        }
        if status.contains(status::RX_TIMEOUT) {
            return Err(RxError::Timeout);
        }
        if !status.contains(status::RX_FRAME_READY) {
            return Err(RxError::FrameNotReady);
        }
        Ok(())
    }

    pub(crate) fn rx_payload_len(&self, rx_finfo: [u8; LEN_RX_FINFO]) -> usize {
        let mut len = (((rx_finfo[1] as usize) << 8) | rx_finfo[0] as usize) & 0x03FF;
        if self.frame_check && len > 2 {
            len -= 2;
        }
        len
    }

    pub(crate) fn delayed_deadline(&self, system_time: DwTime) -> DxTimeDeadline {
        let mut bytes = system_time.to_bytes();
        bytes[0] = 0;
        bytes[1] &= 0xFE;
        DxTimeDeadline::new(DwTime::from_bytes(&bytes))
    }

    pub(crate) fn delayed_transmit(&self, system_time: DwTime) -> DelayedTransmit {
        let deadline = self.delayed_deadline(system_time);
        let timestamp = deadline.value() + DwTime::from_ticks(self.antenna_delay.raw() as i64);
        DelayedTransmit::new(deadline, OnAirTimestamp::new(timestamp))
    }

    pub(crate) fn correct_receive_timestamp(
        &self,
        timestamp: DwTime,
        receive_power_dbm: f32,
    ) -> DwTime {
        let Some(phy) = self.phy else {
            return timestamp;
        };
        if !receive_power_dbm.is_finite() {
            return timestamp;
        }
        let rx_power_base = -(receive_power_dbm + 61.0) * 0.5;
        if !rx_power_base.is_finite() {
            return timestamp;
        }
        let mut low = floorf(rx_power_base) as isize;
        let mut high = low + 1;
        if low <= 0 {
            low = 0;
            high = 0;
        } else if high >= 17 {
            low = 17;
            high = 17;
        }
        let (table, zero, scale) = match (phy.channel, phy.pulse_frequency) {
            (
                crate::config::Channel::Channel4 | crate::config::Channel::Channel7,
                PulseFrequency::Mhz16,
            ) => (&BIAS_900_16, 7usize, 2.0),
            (
                crate::config::Channel::Channel4 | crate::config::Channel::Channel7,
                PulseFrequency::Mhz64,
            ) => (&BIAS_900_64, 7usize, 2.0),
            (_, PulseFrequency::Mhz16) => (&BIAS_500_16, 10usize, 1.0),
            (_, PulseFrequency::Mhz64) => (&BIAS_500_64, 8usize, 1.0),
        };
        let low_bias = signed_bias(table[low as usize], low as usize, zero) * scale;
        let high_bias = signed_bias(table[high as usize], high as usize, zero) * scale;
        let bias_mm = low_bias + (rx_power_base - low as f32) * (high_bias - low_bias);
        let adjustment = DwTime::from_ticks((bias_mm * 0.001 / DISTANCE_PER_TICK_M) as i64);
        timestamp - adjustment
    }

    pub(crate) const fn should_restart_receive(&self, event_mask: SysStatus) -> bool {
        if !self.permanent_receive {
            return false;
        }

        match self.state {
            DriverState::Tx => {
                self.rx_after_tx_pending && event_mask.contains(status::TX_FRAME_SENT)
            }
            // RXAUTR should handle this in hardware, but explicitly rearming after
            // the status is cleared keeps continuous receive reliable on all frames.
            DriverState::Rx => (event_mask.0 & RECEIVE_RESTART_EVENTS) != 0,
            DriverState::Idle => false,
        }
    }

    pub(crate) fn checked_frame_len(&self, payload_len: usize) -> Result<usize, (usize, usize)> {
        let frame_len = if self.frame_check {
            payload_len + 2
        } else {
            payload_len
        };
        if frame_len > LEN_UWB_FRAMES {
            return Err((frame_len, LEN_UWB_FRAMES));
        }
        Ok(frame_len)
    }

    pub(crate) fn begin_receive_session(&mut self, permanent: bool) {
        self.state = DriverState::Rx;
        self.permanent_receive = permanent;
        self.rx_after_tx_pending = false;
    }

    pub(crate) fn begin_transmit_session(&mut self) {
        self.state = DriverState::Tx;
        self.rx_after_tx_pending = self.permanent_receive;
    }

    pub(crate) fn complete_transmit_session(&mut self) {
        if !self.permanent_receive {
            self.state = DriverState::Idle;
        }
    }
}

pub(crate) fn compose_receive_sys_ctrl(sys_ctrl: &mut [u8], frame_check: bool, delayed: bool) {
    set_bit(sys_ctrl, SFCST_BIT, !frame_check);
    set_bit(sys_ctrl, RXENAB_BIT, true);
    if delayed {
        set_bit(sys_ctrl, RXDLYS_BIT, true);
    }
}

pub(crate) fn compose_transmit_sys_ctrl(
    sys_ctrl: &mut [u8],
    frame_check: bool,
    wait_for_response: bool,
    delayed: bool,
) {
    set_bit(sys_ctrl, SFCST_BIT, !frame_check);
    set_bit(sys_ctrl, WAIT4RESP_BIT, wait_for_response);
    if delayed {
        set_bit(sys_ctrl, TXDLYS_BIT, true);
    }
    set_bit(sys_ctrl, TXSTRT_BIT, true);
}

pub(crate) fn prepare_idle_state(runtime: &mut DriverRuntime, sys_ctrl: &mut [u8]) {
    set_bit(sys_ctrl, TRXOFF_BIT, true);
    runtime.state = DriverState::Idle;
}

pub(crate) const fn cleared_interrupt_mask() -> [u8; LEN_SYS_MASK] {
    [0; LEN_SYS_MASK]
}

pub(crate) const fn transmit_status_clear_mask() -> SysStatus {
    SysStatus(
        status::TX_FRAME_BEGIN.0
            | status::TX_PREAMBLE_SENT.0
            | status::TX_HEADER_SENT.0
            | status::TX_FRAME_SENT.0,
    )
}

pub(crate) const fn trx_status_clear_mask() -> SysStatus {
    SysStatus(
        status::AUTOMATIC_ACK_TRIGGER.0
            | transmit_status_clear_mask().0
            | status::RX_CLEAR_EVENTS.0,
    )
}

pub(crate) fn set_lde_load_preamble(pmsc_ctrl0: &mut [u8], otp_ctrl: &mut [u8]) {
    pmsc_ctrl0[0] = 0x01;
    pmsc_ctrl0[1] = 0x03;
    otp_ctrl[0] = 0x00;
    otp_ctrl[1] = 0x80;
}

pub(crate) fn set_lde_restore_preamble(pmsc_ctrl0: &mut [u8]) {
    pmsc_ctrl0[0] = 0x00;
    pmsc_ctrl0[1] &= 0x02;
}

pub(crate) fn apply_clock_mode(pmsc_ctrl0: &mut [u8], mode: ClockMode) {
    match mode {
        ClockMode::Auto => {
            pmsc_ctrl0[0] = 0x00;
            pmsc_ctrl0[1] &= 0xFE;
        }
        ClockMode::Xti => {
            pmsc_ctrl0[0] &= 0xFC;
            pmsc_ctrl0[0] |= 0x01;
        }
    }
}

pub(crate) fn compute_receive_quality(noise: u16, fp2: u16) -> f32 {
    fp2 as f32 / noise.max(1) as f32
}

pub(crate) fn compute_first_path_power(
    phy: ValidatedPhyConfig,
    fp1: u16,
    fp2: u16,
    fp3: u16,
    preamble_acc_count: u16,
) -> f32 {
    let f1 = fp1 as f32;
    let f2 = fp2 as f32;
    let f3 = fp3 as f32;
    let n = preamble_acc_count.max(1) as f32;
    let (a, corr) = match phy.pulse_frequency {
        PulseFrequency::Mhz16 => (113.77, 2.3334),
        PulseFrequency::Mhz64 => (121.74, 1.1667),
    };
    let mut power = 10.0 * log10f((f1 * f1 + f2 * f2 + f3 * f3) / (n * n)) - a;
    if power > -88.0 {
        power += (power + 88.0) * corr;
    }
    power
}

pub(crate) fn compute_receive_power(
    phy: ValidatedPhyConfig,
    cir_power: u16,
    preamble_acc_count: u16,
) -> f32 {
    let c = cir_power as f32;
    let n = preamble_acc_count.max(1) as f32;
    let (a, corr) = match phy.pulse_frequency {
        PulseFrequency::Mhz16 => (113.77, 2.3334),
        PulseFrequency::Mhz64 => (121.74, 1.1667),
    };
    let mut power = 10.0 * log10f((c * 131_072.0) / (n * n)) - a;
    if power > -88.0 {
        power += (power + 88.0) * corr;
    }
    power
}

pub(crate) fn extract_preamble_acc_count(rx_finfo: [u8; LEN_RX_FINFO]) -> u16 {
    (((rx_finfo[2] as u16) >> 4) & 0xFF) | ((rx_finfo[3] as u16) << 4)
}

pub(crate) fn compose_base_register_fields(
    sys_cfg: &mut [u8],
    sys_mask: &mut [u8],
    interrupt_polarity_high: bool,
    receiver_auto_reenable: bool,
    smart_power: bool,
) {
    set_bit(sys_cfg, DIS_DRXB_BIT, true);
    set_bit(sys_cfg, HIRQ_POL_BIT, interrupt_polarity_high);
    set_bit(sys_cfg, RXAUTR_BIT, receiver_auto_reenable);
    set_bit(sys_cfg, DIS_STXP_BIT, !smart_power);

    set_bit(sys_mask, MTXFRS_BIT, true);
    set_bit(sys_mask, MRXPHE_BIT, true);
    set_bit(sys_mask, MRXDFR_BIT, true);
    set_bit(sys_mask, MRXFCG_BIT, true);
    set_bit(sys_mask, MRXFCE_BIT, true);
    set_bit(sys_mask, MRXFSL_BIT, true);
    set_bit(sys_mask, MRXRFTO_BIT, true);
    set_bit(sys_mask, MLDEERR_BIT, true);
    set_bit(sys_mask, MRXOVRR_BIT, true);
    set_bit(sys_mask, MRXPTO_BIT, true);
    set_bit(sys_mask, MRXSFDTO_BIT, true);
    set_bit(sys_mask, MAFFREJ_BIT, true);
}

pub(crate) fn compose_phy_register_fields(
    sys_cfg: &mut [u8],
    tx_fctrl: &mut [u8],
    chan_ctrl: &mut [u8],
    phy: ValidatedPhyConfig,
) -> u8 {
    let sfd_len = compose_data_rate_fields(sys_cfg, tx_fctrl, chan_ctrl, phy.data_rate);
    compose_pulse_frequency_fields(tx_fctrl, chan_ctrl, phy.pulse_frequency);
    compose_preamble_length_fields(tx_fctrl, phy.preamble_length);
    compose_channel_fields(chan_ctrl, phy.channel as u8);
    compose_preamble_code_fields(chan_ctrl, phy.preamble_code.raw());
    sfd_len
}

fn compose_data_rate_fields(
    sys_cfg: &mut [u8],
    tx_fctrl: &mut [u8],
    chan_ctrl: &mut [u8],
    rate: DataRate,
) -> u8 {
    tx_fctrl[1] &= 0x83;
    tx_fctrl[1] |= (rate as u8) << 5;
    set_bit(sys_cfg, RXM110K_BIT, rate == DataRate::Kbps110);
    let (dwsfd, tnssfd, rnssfd, sfd_len) = match rate {
        DataRate::Mbps6800 => (false, false, false, 0x08),
        DataRate::Kbps850 => (true, true, true, 0x10),
        DataRate::Kbps110 => (true, false, false, 0x40),
    };
    set_bit(chan_ctrl, DWSFD_BIT, dwsfd);
    set_bit(chan_ctrl, TNSSFD_BIT, tnssfd);
    set_bit(chan_ctrl, RNSSFD_BIT, rnssfd);
    sfd_len
}

fn compose_pulse_frequency_fields(
    tx_fctrl: &mut [u8],
    chan_ctrl: &mut [u8],
    frequency: PulseFrequency,
) {
    tx_fctrl[2] &= 0xFC;
    tx_fctrl[2] |= frequency as u8;
    chan_ctrl[2] &= 0xF3;
    chan_ctrl[2] |= (frequency as u8) << 2;
}

fn compose_preamble_length_fields(tx_fctrl: &mut [u8], length: crate::config::PreambleLength) {
    tx_fctrl[2] &= 0xC3;
    tx_fctrl[2] |= (length as u8) << 2;
}

fn compose_channel_fields(chan_ctrl: &mut [u8], channel: u8) {
    chan_ctrl[0] = channel | (channel << 4);
}

fn compose_preamble_code_fields(chan_ctrl: &mut [u8], code: u8) {
    chan_ctrl[2] &= 0x3F;
    chan_ctrl[2] |= code << 6;
    chan_ctrl[3] = ((code >> 2) & 0x07) | (code << 3);
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TuningValues {
    pub(crate) agc_tune1: u16,
    pub(crate) drx_tune0b: u16,
    pub(crate) drx_tune1a: u16,
    pub(crate) drx_tune1b: u16,
    pub(crate) drx_tune2: u32,
    pub(crate) drx_tune4h: u16,
    pub(crate) drx_sfd_timeout: u16,
    pub(crate) rf_rxctrlh: u8,
    pub(crate) rf_txctrl: u32,
    pub(crate) tc_pgdelay: u8,
    pub(crate) fspllcfg: u32,
    pub(crate) fsplltune: u8,
    pub(crate) lde_cfg2: u16,
    pub(crate) lde_repc: u16,
    pub(crate) tx_power: u32,
}

pub(crate) fn select_tuning_values(phy: ValidatedPhyConfig) -> Result<TuningValues, ConfigError> {
    let agc_tune1 = match phy.pulse_frequency {
        PulseFrequency::Mhz16 => 0x8870u16,
        PulseFrequency::Mhz64 => 0x889Bu16,
    };

    let drx_tune0b = match phy.data_rate {
        DataRate::Kbps110 => 0x0016u16,
        DataRate::Kbps850 => 0x0006u16,
        DataRate::Mbps6800 => 0x0001u16,
    };

    let drx_tune1a = match phy.pulse_frequency {
        PulseFrequency::Mhz16 => 0x0087u16,
        PulseFrequency::Mhz64 => 0x008Du16,
    };

    let drx_tune1b: u16 = match (phy.preamble_length, phy.data_rate) {
        (
            PreambleLength::Symbols1536 | PreambleLength::Symbols2048 | PreambleLength::Symbols4096,
            DataRate::Kbps110,
        ) => 0x0064,
        (PreambleLength::Symbols64, DataRate::Mbps6800) => 0x0010,
        (_, DataRate::Kbps850 | DataRate::Mbps6800) => 0x0020,
        _ => return Err(ConfigError::UnsupportedPreambleLength),
    };

    let drx_tune2: u32 = match (phy.pac_size, phy.pulse_frequency) {
        (PacSize::Symbols8, PulseFrequency::Mhz16) => 0x311A_002D,
        (PacSize::Symbols8, PulseFrequency::Mhz64) => 0x313B_006B,
        (PacSize::Symbols16, PulseFrequency::Mhz16) => 0x331A_0052,
        (PacSize::Symbols16, PulseFrequency::Mhz64) => 0x333B_00BE,
        (PacSize::Symbols32, PulseFrequency::Mhz16) => 0x351A_009A,
        (PacSize::Symbols32, PulseFrequency::Mhz64) => 0x353B_015E,
        (PacSize::Symbols64, PulseFrequency::Mhz16) => 0x371A_011D,
        (PacSize::Symbols64, PulseFrequency::Mhz64) => 0x373B_0296,
    };

    let drx_tune4h = match phy.preamble_length {
        PreambleLength::Symbols64 => 0x0010u16,
        _ => 0x0028u16,
    };
    let drx_sfd_timeout = preamble_symbols(phy.preamble_length) + sfd_symbols(phy.data_rate) + 1
        - phy.pac_size as u16;

    let rf_rxctrlh = match phy.channel {
        crate::config::Channel::Channel4 | crate::config::Channel::Channel7 => 0xBC,
        _ => 0xD8,
    };

    let rf_txctrl: u32 = match phy.channel {
        crate::config::Channel::Channel1 => 0x0000_5C40,
        crate::config::Channel::Channel2 => 0x0004_5CA0,
        crate::config::Channel::Channel3 => 0x0008_6CC0,
        crate::config::Channel::Channel4 => 0x0004_5C80,
        crate::config::Channel::Channel5 => 0x001E_3FE0,
        crate::config::Channel::Channel7 => 0x001E_7DE0,
    };

    let tc_pgdelay = match phy.channel {
        crate::config::Channel::Channel1 => 0xC9,
        crate::config::Channel::Channel2 => 0xC2,
        crate::config::Channel::Channel3 => 0xC5,
        crate::config::Channel::Channel4 => 0x95,
        crate::config::Channel::Channel5 => 0xC0,
        crate::config::Channel::Channel7 => 0x93,
    };

    let (fspllcfg, fsplltune) = match phy.channel {
        crate::config::Channel::Channel1 => (0x0900_0407u32, 0x1E),
        crate::config::Channel::Channel2 | crate::config::Channel::Channel4 => {
            (0x0840_0508u32, 0x26)
        }
        crate::config::Channel::Channel3 => (0x0840_1009u32, 0x56),
        crate::config::Channel::Channel5 | crate::config::Channel::Channel7 => {
            (0x0800_041Du32, 0xBE)
        }
    };

    let lde_cfg2 = match phy.pulse_frequency {
        PulseFrequency::Mhz16 => 0x1607u16,
        PulseFrequency::Mhz64 => 0x0607u16,
    };

    let lde_repc = lde_repc_value(phy.preamble_code, phy.data_rate);
    let tx_power = tx_power_value(phy.channel, phy.pulse_frequency, phy.smart_power);

    Ok(TuningValues {
        agc_tune1,
        drx_tune0b,
        drx_tune1a,
        drx_tune1b,
        drx_tune2,
        drx_tune4h,
        drx_sfd_timeout,
        rf_rxctrlh,
        rf_txctrl,
        tc_pgdelay,
        fspllcfg,
        fsplltune,
        lde_cfg2,
        lde_repc,
        tx_power,
    })
}

const fn preamble_symbols(preamble: PreambleLength) -> u16 {
    match preamble {
        PreambleLength::Symbols64 => 64,
        PreambleLength::Symbols128 => 128,
        PreambleLength::Symbols256 => 256,
        PreambleLength::Symbols512 => 512,
        PreambleLength::Symbols1024 => 1024,
        PreambleLength::Symbols1536 => 1536,
        PreambleLength::Symbols2048 => 2048,
        PreambleLength::Symbols4096 => 4096,
    }
}

const fn sfd_symbols(rate: DataRate) -> u16 {
    match rate {
        DataRate::Mbps6800 => 8,
        DataRate::Kbps850 => 16,
        DataRate::Kbps110 => 64,
    }
}

pub(crate) fn build_header(register: Register, subaddress: u16, write: bool) -> [u8; 3] {
    let mut header = [0u8; 3];
    if subaddress == NO_SUBADDRESS {
        header[0] = (if write { WRITE } else { READ }) | register as u8;
        return header;
    }
    header[0] = (if write { WRITE_SUB } else { READ_SUB }) | register as u8;
    if subaddress < 128 {
        header[1] = subaddress as u8;
    } else {
        header[1] = RW_SUB_EXT | (subaddress as u8);
        header[2] = (subaddress >> 7) as u8;
    }
    header
}

pub(crate) const fn header_len(subaddress: u16) -> usize {
    if subaddress == NO_SUBADDRESS {
        1
    } else if subaddress < 128 {
        2
    } else {
        3
    }
}

pub(crate) fn compose_gpio_led_mode(gpio_mode: &mut [u8]) {
    // Keep RXOKLED/SFDLED disabled and only route RXLED/TXLED to GPIO2/GPIO3.
    set_gpio_message_mode(
        gpio_mode,
        crate::constants::MSGP0 as u16,
        crate::constants::GPIO_MODE,
    );
    set_gpio_message_mode(
        gpio_mode,
        crate::constants::MSGP1 as u16,
        crate::constants::GPIO_MODE,
    );
    set_gpio_message_mode(
        gpio_mode,
        crate::constants::MSGP2 as u16,
        crate::constants::LED_MODE,
    );
    set_gpio_message_mode(
        gpio_mode,
        crate::constants::MSGP3 as u16,
        crate::constants::LED_MODE,
    );
}

pub(crate) fn compose_led_clock_enable(pmsc_ctrl0: &mut [u8]) {
    set_bit(pmsc_ctrl0, crate::constants::GPDCE_BIT as u16, true);
    set_bit(pmsc_ctrl0, crate::constants::KHZCLKEN_BIT as u16, true);
}

pub(crate) fn compose_led_blink_enable(pmsc_ledc: &mut [u8], blink_time: u8) {
    pmsc_ledc[0] = blink_time;
    set_bit(pmsc_ledc, crate::constants::BLNKEN as u16, true);
}

pub(crate) fn set_bit(bytes: &mut [u8], bit: u16, value: bool) {
    let index = (bit / 8) as usize;
    let shift = (bit % 8) as u8;
    debug_assert!(
        index < bytes.len(),
        "bit index {} out of range for {}-byte register",
        bit,
        bytes.len()
    );
    let Some(byte) = bytes.get_mut(index) else {
        return;
    };
    if value {
        *byte |= 1 << shift;
    } else {
        *byte &= !(1 << shift);
    }
}

fn set_gpio_message_mode(gpio_mode: &mut [u8], lsb_bit: u16, mode: u8) {
    set_bit(gpio_mode, lsb_bit, (mode & 0x01) != 0);
    set_bit(gpio_mode, lsb_bit + 1, (mode & 0x02) != 0);
}

pub(crate) fn lde_repc_value(code: PreambleCode, rate: DataRate) -> u16 {
    let base = match code {
        PreambleCode::Code1 | PreambleCode::Code2 => 0x5998,
        PreambleCode::Code3 | PreambleCode::Code8 => 0x51EA,
        PreambleCode::Code4 => 0x428E,
        PreambleCode::Code5 => 0x451E,
        PreambleCode::Code6 => 0x2E14,
        PreambleCode::Code7 => 0x8000,
        PreambleCode::Code9 => 0x28F4,
        PreambleCode::Code10 | PreambleCode::Code17 => 0x3332,
        PreambleCode::Code11 => 0x3AE0,
        PreambleCode::Code12 => 0x3D70,
        PreambleCode::Code18 | PreambleCode::Code19 => 0x35C2,
        PreambleCode::Code20 => 0x47AE,
    };
    if rate == DataRate::Kbps110 {
        base >> 3
    } else {
        base
    }
}

pub(crate) fn tx_power_value(
    channel: crate::config::Channel,
    prf: PulseFrequency,
    smart_power: bool,
) -> u32 {
    match (channel, prf, smart_power) {
        (
            crate::config::Channel::Channel1 | crate::config::Channel::Channel2,
            PulseFrequency::Mhz16,
            true,
        ) => 0x1535_5575,
        (
            crate::config::Channel::Channel1 | crate::config::Channel::Channel2,
            PulseFrequency::Mhz16,
            false,
        ) => 0x7575_7575,
        (
            crate::config::Channel::Channel1 | crate::config::Channel::Channel2,
            PulseFrequency::Mhz64,
            true,
        ) => 0x0727_4767,
        (
            crate::config::Channel::Channel1 | crate::config::Channel::Channel2,
            PulseFrequency::Mhz64,
            false,
        ) => 0x6767_6767,
        (crate::config::Channel::Channel3, PulseFrequency::Mhz16, true) => 0x0F2F_4F6F,
        (crate::config::Channel::Channel3, PulseFrequency::Mhz16, false) => 0x6F6F_6F6F,
        (crate::config::Channel::Channel3, PulseFrequency::Mhz64, true) => 0x2B4B_6B8B,
        (crate::config::Channel::Channel3, PulseFrequency::Mhz64, false) => 0x8B8B_8B8B,
        (crate::config::Channel::Channel4, PulseFrequency::Mhz16, true) => 0x1F1F_3F5F,
        (crate::config::Channel::Channel4, PulseFrequency::Mhz16, false) => 0x5F5F_5F5F,
        (crate::config::Channel::Channel4, PulseFrequency::Mhz64, true) => 0x3A5A_7A9A,
        (crate::config::Channel::Channel4, PulseFrequency::Mhz64, false) => 0x9A9A_9A9A,
        (crate::config::Channel::Channel5, PulseFrequency::Mhz16, true) => 0x0E08_2848,
        (crate::config::Channel::Channel5, PulseFrequency::Mhz16, false) => 0x4848_4848,
        (crate::config::Channel::Channel5, PulseFrequency::Mhz64, true) => 0x2545_6585,
        (crate::config::Channel::Channel5, PulseFrequency::Mhz64, false) => 0x8585_8585,
        (crate::config::Channel::Channel7, PulseFrequency::Mhz16, true) => 0x3252_7292,
        (crate::config::Channel::Channel7, PulseFrequency::Mhz16, false) => 0x9292_9292,
        (crate::config::Channel::Channel7, PulseFrequency::Mhz64, true) => 0x5171_B1D1,
        (crate::config::Channel::Channel7, PulseFrequency::Mhz64, false) => 0xD1D1_D1D1,
    }
}

const BIAS_500_16: [u8; 18] = [
    198, 187, 179, 163, 143, 127, 109, 84, 59, 31, 0, 36, 65, 84, 97, 106, 110, 112,
];
const BIAS_500_64: [u8; 18] = [
    110, 105, 100, 93, 82, 69, 51, 27, 0, 21, 35, 42, 49, 62, 71, 76, 81, 86,
];
const BIAS_900_16: [u8; 18] = [
    137, 122, 105, 88, 69, 47, 25, 0, 21, 48, 79, 105, 127, 147, 160, 169, 178, 197,
];
const BIAS_900_64: [u8; 18] = [
    147, 133, 117, 99, 75, 50, 29, 0, 24, 45, 63, 76, 87, 98, 116, 122, 132, 142,
];

fn signed_bias(value: u8, index: usize, zero: usize) -> f32 {
    if index < zero {
        -(value as f32)
    } else {
        value as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{compose_base_register_fields, select_tuning_values, DriverRuntime};
    use crate::config::{OperatingMode, RadioConfig};
    use crate::device::{AntennaDelay, DeviceIdentity, Eui64, PanId, ShortAddress};
    use crate::registers::{
        status, MAAT_BIT, MAFFREJ_BIT, MLDEERR_BIT, MRXDFR_BIT, MRXFCE_BIT, MRXFCG_BIT, MRXFSL_BIT,
        MRXOVRR_BIT, MRXPHE_BIT, MRXPTO_BIT, MRXRFTO_BIT, MRXSFDTO_BIT, MTXFRS_BIT,
    };
    use crate::time::DwTime;

    fn identity() -> DeviceIdentity {
        DeviceIdentity::new(
            PanId::new(0x0D57),
            ShortAddress::new(3344),
            Eui64::new([0x82, 0x17, 0x5B, 0xD5, 0xA9, 0x9A, 0xE2, 0x9C]),
        )
    }

    #[test]
    fn every_enabled_interrupt_is_owned_by_tx_or_rx_service() {
        let mut sys_cfg = [0; 4];
        let mut sys_mask = [0; 4];
        compose_base_register_fields(&mut sys_cfg, &mut sys_mask, true, false, false);

        let enabled = u32::from_le_bytes(sys_mask);
        let owned_bits = [
            MTXFRS_BIT,
            MRXPHE_BIT,
            MRXDFR_BIT,
            MRXFCG_BIT,
            MRXFCE_BIT,
            MRXFSL_BIT,
            MRXRFTO_BIT,
            MLDEERR_BIT,
            MRXOVRR_BIT,
            MRXPTO_BIT,
            MRXSFDTO_BIT,
            MAFFREJ_BIT,
        ];
        let owned = owned_bits
            .into_iter()
            .fold(0_u32, |mask, bit| mask | (1_u32 << bit));
        assert_eq!(enabled & !owned, 0);
        assert_eq!(enabled & (1_u32 << MAAT_BIT), 0);
        assert_ne!(enabled & status::RX_TERMINAL_EVENTS.0 as u32, 0);
    }

    #[test]
    fn non_finite_receive_power_skips_timestamp_bias_correction() {
        let mut runtime = DriverRuntime::new();
        let mut config = RadioConfig::from_mode(identity(), OperatingMode::LongDataRangeAccuracy);
        config.antenna_delay = AntennaDelay::new(16_456);
        runtime.reconfigure(&config).unwrap();

        let timestamp = DwTime::from_ticks(123_456);
        assert_eq!(
            runtime.correct_receive_timestamp(timestamp, f32::NEG_INFINITY),
            timestamp
        );
        assert_eq!(
            runtime.correct_receive_timestamp(timestamp, f32::INFINITY),
            timestamp
        );
        assert_eq!(
            runtime.correct_receive_timestamp(timestamp, f32::NAN),
            timestamp
        );
    }

    #[test]
    fn compose_led_configuration_sets_required_bits() {
        let mut gpio_mode = [0xFFu8; 4];
        super::compose_gpio_led_mode(&mut gpio_mode);
        assert_eq!(gpio_mode[0] & 0b1100_0000, 0b0000_0000);
        assert_eq!(gpio_mode[1] & 0b0011_1111, 0b0001_0100);

        let mut pmsc_ctrl0 = [0u8; 4];
        super::compose_led_clock_enable(&mut pmsc_ctrl0);
        assert_eq!(pmsc_ctrl0[2] & 0b1000_0100, 0b1000_0100);

        let mut pmsc_ledc = [0u8; 4];
        super::compose_led_blink_enable(&mut pmsc_ledc, 0x20);
        assert_eq!(pmsc_ledc[0], 0x20);
        assert_eq!(pmsc_ledc[1] & 0x01, 0x01);
    }

    #[test]
    fn derives_sfd_timeout_from_preamble_sfd_and_pac() {
        let operational = RadioConfig::from_mode(identity(), OperatingMode::ShortDataFastAccuracy);
        assert_eq!(
            select_tuning_values(operational.validated_phy().unwrap())
                .unwrap()
                .drx_sfd_timeout,
            129
        );

        let robust = RadioConfig::from_mode(identity(), OperatingMode::LongDataRangeAccuracy);
        assert_eq!(
            select_tuning_values(robust.validated_phy().unwrap())
                .unwrap()
                .drx_sfd_timeout,
            2049
        );
    }
}
