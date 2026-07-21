//! Core DW1000 driver implementation.

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::{InputPin, OutputPin};
use embedded_hal::spi::{Operation, SpiDevice};

use crate::config::{RadioConfig, RxOptions, TxOptions, ValidatedPhyConfig};
use crate::constants::{GPIO_MODE_SUB, LEN_GPIO_MODE};
use crate::device::{DeviceIdentity, RxFrame, SignalMetrics, SysStatus, Timestamps};
use crate::driver_core::{
    apply_clock_mode, build_header, cleared_interrupt_mask, compose_base_register_fields,
    compose_gpio_led_mode, compose_led_blink_enable, compose_led_clock_enable,
    compose_phy_register_fields, compose_receive_sys_ctrl, compose_transmit_sys_ctrl,
    compute_first_path_power, compute_receive_power, compute_receive_quality,
    extract_preamble_acc_count, header_len, prepare_idle_state, receive_status_clear_mask,
    select_tuning_values, set_lde_load_preamble, set_lde_restore_preamble,
    transmit_status_clear_mask, ClockMode, DriverRuntime,
};
use crate::error::Error;
use crate::registers::{
    sys_status_from_bytes, sys_status_to_bytes, Register, AGC_TUNE1_SUB, AGC_TUNE2_SUB,
    AGC_TUNE3_SUB, CIR_PWR_SUB, DRX_TUNE0B_SUB, DRX_TUNE1A_SUB, DRX_TUNE1B_SUB, DRX_TUNE2_SUB,
    DRX_TUNE4H_SUB, FP_AMPL1_SUB, FP_AMPL2_SUB, FP_AMPL3_SUB, FS_PLLCFG_SUB, FS_PLLTUNE_SUB,
    FS_XTALT_SUB, LDE_CFG1_SUB, LDE_CFG2_SUB, LDE_REPC_SUB, LDE_RXANTD_SUB, LEN_CHAN_CTRL,
    LEN_CIR_PWR, LEN_FP_AMPL1, LEN_FP_AMPL2, LEN_FP_AMPL3, LEN_LDE_RXANTD, LEN_OTP_ADDR,
    LEN_OTP_CTRL, LEN_OTP_RDAT, LEN_PANADR, LEN_PMSC_CTRL0, LEN_PMSC_LEDC, LEN_RX_FINFO,
    LEN_RX_STAMP, LEN_STD_NOISE, LEN_SYS_CFG, LEN_SYS_CTRL, LEN_SYS_MASK, LEN_SYS_STATUS,
    LEN_TX_ANTD, LEN_TX_FCTRL, LEN_TX_STAMP, NO_SUBADDRESS, OTP_ADDR_SUB, OTP_CTRL_SUB,
    OTP_RDAT_SUB, PMSC_CTRL0_SUB, PMSC_LEDC_SUB, RF_RXCTRLH_SUB, RF_TXCTRL_SUB, RX_STAMP_SUB,
    SFD_LENGTH_SUB, STD_NOISE_SUB, TC_PGDELAY_SUB, TX_STAMP_SUB,
};
use crate::time::DwTime;

/// Blocking DW1000 driver.
#[derive(Debug)]
pub struct Dw1000<SPI, IRQ, RST> {
    spi: SPI,
    irq: IRQ,
    reset: RST,
    sys_cfg: [u8; LEN_SYS_CFG],
    sys_ctrl: [u8; LEN_SYS_CTRL],
    sys_mask: [u8; LEN_SYS_MASK],
    tx_fctrl: [u8; LEN_TX_FCTRL],
    chan_ctrl: [u8; LEN_CHAN_CTRL],
    panadr: [u8; LEN_PANADR],
    runtime: DriverRuntime,
}

impl<SPI, IRQ, RST, PinE> Dw1000<SPI, IRQ, RST>
where
    SPI: SpiDevice,
    IRQ: InputPin<Error = PinE>,
    RST: OutputPin<Error = PinE>,
{
    /// Creates a new driver instance.
    pub fn new(spi: SPI, irq: IRQ, reset: RST) -> Self {
        Self {
            spi,
            irq,
            reset,
            sys_cfg: [0; LEN_SYS_CFG],
            sys_ctrl: [0; LEN_SYS_CTRL],
            sys_mask: [0; LEN_SYS_MASK],
            tx_fctrl: [0; LEN_TX_FCTRL],
            chan_ctrl: [0; LEN_CHAN_CTRL],
            panadr: [0xFF; LEN_PANADR],
            runtime: DriverRuntime::new(),
        }
    }

    /// Initializes the DW1000 and applies the supplied configuration.
    pub fn init(
        &mut self,
        delay: &mut impl DelayNs,
        config: &RadioConfig,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        self.hard_reset(delay)?;
        self.enable_clock(ClockMode::Auto)?;
        delay.delay_ms(5);
        self.clear_interrupts()?;
        self.enable_clock(ClockMode::Xti)?;
        delay.delay_ms(5);
        self.manage_lde(delay)?;
        self.enable_clock(ClockMode::Auto)?;
        delay.delay_ms(5);
        self.reconfigure(config)
    }

    /// Applies a new radio configuration without performing a reset.
    pub fn reconfigure(&mut self, config: &RadioConfig) -> Result<(), Error<SPI::Error, PinE>> {
        let phy = self.runtime.reconfigure(config)?;

        self.idle()?;
        self.panadr = [0xFF; LEN_PANADR];
        self.panadr[0..2].copy_from_slice(&config.address.identity.short_address.to_le_bytes());
        self.panadr[2..4].copy_from_slice(&config.address.identity.pan_id.to_le_bytes());
        self.sys_cfg = [0; LEN_SYS_CFG];
        self.sys_mask = [0; LEN_SYS_MASK];
        self.tx_fctrl = [0; LEN_TX_FCTRL];
        self.chan_ctrl = [0; LEN_CHAN_CTRL];

        compose_base_register_fields(
            &mut self.sys_cfg,
            &mut self.sys_mask,
            config.interrupt_polarity_high,
            config.receiver_auto_reenable,
            phy.smart_power,
        );

        self.apply_phy_config(phy)?;
        let panadr = self.panadr;
        self.write_register(Register::PanAdr, NO_SUBADDRESS, &panadr)?;
        self.write_register(
            Register::Eui,
            NO_SUBADDRESS,
            &config.address.identity.eui.to_register_bytes(),
        )?;
        let sys_cfg = self.sys_cfg;
        let sys_mask = self.sys_mask;
        let chan_ctrl = self.chan_ctrl;
        let tx_fctrl = self.tx_fctrl;
        self.write_register(Register::SysCfg, NO_SUBADDRESS, &sys_cfg)?;
        self.write_register(Register::SysMask, NO_SUBADDRESS, &sys_mask)?;
        self.write_register(Register::ChanCtrl, NO_SUBADDRESS, &chan_ctrl)?;
        self.write_register(Register::TxFctrl, NO_SUBADDRESS, &tx_fctrl)?;
        self.apply_tuning(phy)?;
        let antenna = self.runtime.antenna_delay.to_time_bytes();
        self.write_register(Register::TxAntd, NO_SUBADDRESS, &antenna[..LEN_TX_ANTD])?;
        self.write_register(Register::LdeIf, LDE_RXANTD_SUB, &antenna[..LEN_LDE_RXANTD])?;
        Ok(())
    }

    /// Starts a receive session.
    pub fn start_receive(&mut self, options: RxOptions) -> Result<(), Error<SPI::Error, PinE>> {
        self.idle()?;
        self.clear_receive_status()?;
        self.sys_ctrl = [0; LEN_SYS_CTRL];
        self.runtime.begin_receive_session(options.permanent);
        if let Some(delay) = options.delayed_time {
            let future = self.compute_delayed_time(delay)?;
            self.write_register(Register::DxTime, NO_SUBADDRESS, &future.to_bytes())?;
        }
        compose_receive_sys_ctrl(
            &mut self.sys_ctrl,
            self.runtime.frame_check,
            options.delayed_time.is_some(),
        );
        let sys_ctrl = self.sys_ctrl;
        self.write_register(Register::SysCtrl, NO_SUBADDRESS, &sys_ctrl)?;
        Ok(())
    }

    /// Transmits a frame.
    pub fn transmit(
        &mut self,
        frame: &[u8],
        options: TxOptions,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let delayed_time = match options.delayed_time {
            Some(delay) => Some(self.compute_delayed_time(delay)?),
            None => None,
        };
        self.transmit_inner(frame, delayed_time, options.wait_for_response)
    }

    /// Transmits a frame at an already-computed absolute DW1000 timestamp.
    ///
    /// Ranging payloads must contain the on-air timestamp produced by the
    /// programmed `DX_TIME` deadline. This entry point prevents a second
    /// system-time read from moving that deadline after the payload is encoded.
    pub fn transmit_at(
        &mut self,
        frame: &[u8],
        delayed_time: DwTime,
        wait_for_response: bool,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        self.transmit_inner(frame, Some(delayed_time), wait_for_response)
    }

    fn transmit_inner(
        &mut self,
        frame: &[u8],
        delayed_time: Option<DwTime>,
        wait_for_response: bool,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let frame_len = self
            .runtime
            .checked_frame_len(frame.len())
            .map_err(|(len, max)| Error::FrameTooLong { len, max })?;

        self.idle()?;
        self.clear_transmit_status()?;
        self.write_register(Register::TxBuffer, NO_SUBADDRESS, frame)?;
        self.tx_fctrl[0] = (frame_len & 0xFF) as u8;
        self.tx_fctrl[1] &= 0xFC;
        self.tx_fctrl[1] |= ((frame_len >> 8) & 0x03) as u8;
        let tx_fctrl = self.tx_fctrl;
        self.write_register(Register::TxFctrl, NO_SUBADDRESS, &tx_fctrl)?;

        self.sys_ctrl = [0; LEN_SYS_CTRL];
        self.runtime.begin_transmit_session();
        if let Some(future) = delayed_time {
            // `future` is the antenna-adjusted TX timestamp placed in ranging
            // payloads. DX_TIME takes the unadjusted radio deadline; the
            // hardware adds TX_ANTD to the timestamp it reports after sending.
            let register_time =
                future - DwTime::from_ticks(self.runtime.antenna_delay.raw() as i64);
            self.write_register(Register::DxTime, NO_SUBADDRESS, &register_time.to_bytes())?;
        }
        compose_transmit_sys_ctrl(
            &mut self.sys_ctrl,
            self.runtime.frame_check,
            wait_for_response,
            delayed_time.is_some(),
        );
        let sys_ctrl = self.sys_ctrl;
        self.write_register(Register::SysCtrl, NO_SUBADDRESS, &sys_ctrl)?;
        self.runtime.complete_transmit_session();
        Ok(())
    }

    /// Reads the silicon device identifier register.
    pub fn read_device_id(&mut self) -> Result<u32, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; 4];
        self.read_register(Register::DevId, NO_SUBADDRESS, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Computes the delayed absolute transmit or receive timestamp used by the DW1000.
    pub fn compute_delayed_time(
        &mut self,
        delay: DwTime,
    ) -> Result<DwTime, Error<SPI::Error, PinE>> {
        let now = self.read_system_timestamp()? + delay;
        Ok(self.runtime.compute_delayed_time(now))
    }

    /// Reads a received frame into `buffer`.
    pub fn read_frame<'a>(
        &mut self,
        buffer: &'a mut [u8],
    ) -> Result<RxFrame<'a>, Error<SPI::Error, PinE>> {
        self.read_frame_inner(buffer, true)
    }

    /// Reads a received frame while preserving the raw hardware timestamp.
    ///
    /// This supports protocols that apply antenna and signal-level calibration
    /// after collecting all timestamps from both radios.
    pub fn read_frame_raw<'a>(
        &mut self,
        buffer: &'a mut [u8],
    ) -> Result<RxFrame<'a>, Error<SPI::Error, PinE>> {
        self.read_frame_inner(buffer, false)
    }

    fn read_frame_inner<'a>(
        &mut self,
        buffer: &'a mut [u8],
        correct_timestamp: bool,
    ) -> Result<RxFrame<'a>, Error<SPI::Error, PinE>> {
        let status = self.read_sys_status()?;
        self.runtime
            .validate_rx_status(status)
            .map_err(Error::Receive)?;

        let needed = self.read_received_length()?;
        if buffer.len() < needed {
            return Err(Error::BufferTooSmall {
                len: buffer.len(),
                needed,
            });
        }
        self.read_register(Register::RxBuffer, NO_SUBADDRESS, &mut buffer[..needed])?;
        let raw_timestamp = self.read_receive_timestamp()?;
        let metrics = self.read_signal_metrics()?;
        let timestamp = if correct_timestamp {
            self.correct_receive_timestamp(raw_timestamp, metrics.receive_power_dbm)
        } else {
            raw_timestamp
        };
        Ok(RxFrame {
            bytes: &buffer[..needed],
            timestamp,
            metrics,
            status,
        })
    }

    /// Reads the latest timestamps.
    pub fn read_timestamps(&mut self) -> Result<Timestamps, Error<SPI::Error, PinE>> {
        Ok(Timestamps {
            tx: self.read_transmit_timestamp()?,
            rx: self.read_receive_timestamp()?,
            system: self.read_system_timestamp()?,
        })
    }

    /// Reads the current receive metrics.
    pub fn read_signal_metrics(&mut self) -> Result<SignalMetrics, Error<SPI::Error, PinE>> {
        Ok(SignalMetrics {
            receive_power_dbm: self.read_receive_power()?,
            first_path_power_dbm: self.read_first_path_power()?,
            quality: self.read_receive_quality()?,
        })
    }

    /// Reads `SYS_STATUS`.
    pub fn read_sys_status(&mut self) -> Result<SysStatus, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; LEN_SYS_STATUS];
        self.read_register(Register::SysStatus, NO_SUBADDRESS, &mut bytes)?;
        Ok(sys_status_from_bytes(&bytes))
    }

    /// Clears the selected `SYS_STATUS` bits and re-arms permanent receive after RX/TX completion.
    pub fn clear_events(&mut self, event_mask: SysStatus) -> Result<(), Error<SPI::Error, PinE>> {
        self.write_register(
            Register::SysStatus,
            NO_SUBADDRESS,
            &sys_status_to_bytes(event_mask),
        )?;
        let should_restart_receive = self.runtime.should_restart_receive(event_mask);
        if should_restart_receive {
            self.start_receive(RxOptions {
                delayed_time: None,
                permanent: true,
            })?;
        }
        Ok(())
    }

    /// Returns the currently configured identity, if the radio has been configured.
    pub const fn identity(&self) -> Option<DeviceIdentity> {
        self.runtime.identity()
    }

    /// Returns `true` if the IRQ pin is asserted.
    pub fn irq_asserted(&mut self) -> Result<bool, Error<SPI::Error, PinE>> {
        self.irq.is_high().map_err(Error::Pin)
    }

    /// Enables RX and TX LED indicators on GPIO pins.
    /// GPIO2 will show RX activity and GPIO3 will show TX activity.
    pub fn enable_leds(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        let mut pmsc_ctrl0 = [0u8; LEN_PMSC_CTRL0];
        self.read_register(Register::Pmsc, PMSC_CTRL0_SUB, &mut pmsc_ctrl0)?;
        compose_led_clock_enable(&mut pmsc_ctrl0);
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB, &pmsc_ctrl0)?;

        let mut pmsc_ledc = [0u8; LEN_PMSC_LEDC];
        self.read_register(Register::Pmsc, PMSC_LEDC_SUB, &mut pmsc_ledc)?;
        compose_led_blink_enable(&mut pmsc_ledc, 0x20);
        self.write_register(Register::Pmsc, PMSC_LEDC_SUB, &pmsc_ledc)?;

        let mut gpio_mode = [0u8; LEN_GPIO_MODE];
        self.read_register(Register::GpioCtrl, GPIO_MODE_SUB as u16, &mut gpio_mode)?;
        compose_gpio_led_mode(&mut gpio_mode);
        self.write_register(Register::GpioCtrl, GPIO_MODE_SUB as u16, &gpio_mode)?;
        Ok(())
    }

    fn hard_reset(&mut self, delay: &mut impl DelayNs) -> Result<(), Error<SPI::Error, PinE>> {
        self.reset.set_low().map_err(Error::Pin)?;
        delay.delay_ms(2);
        self.reset.set_high().map_err(Error::Pin)?;
        delay.delay_ms(10);
        Ok(())
    }

    fn apply_phy_config(&mut self, phy: ValidatedPhyConfig) -> Result<(), Error<SPI::Error, PinE>> {
        let sfd_len = compose_phy_register_fields(
            &mut self.sys_cfg,
            &mut self.tx_fctrl,
            &mut self.chan_ctrl,
            phy,
        );
        self.write_register(Register::UsrSfd, SFD_LENGTH_SUB, &[sfd_len])?;
        Ok(())
    }

    fn apply_tuning(&mut self, phy: ValidatedPhyConfig) -> Result<(), Error<SPI::Error, PinE>> {
        let tuning = select_tuning_values(phy).map_err(Error::InvalidConfig)?;

        self.write_register(
            Register::AgcTune,
            AGC_TUNE1_SUB,
            &tuning.agc_tune1.to_le_bytes(),
        )?;
        self.write_register(
            Register::AgcTune,
            AGC_TUNE2_SUB,
            &0x2502_A907u32.to_le_bytes(),
        )?;
        self.write_register(Register::AgcTune, AGC_TUNE3_SUB, &0x0035u16.to_le_bytes())?;

        self.write_register(
            Register::DrxTune,
            DRX_TUNE0B_SUB,
            &tuning.drx_tune0b.to_le_bytes(),
        )?;
        self.write_register(
            Register::DrxTune,
            DRX_TUNE1A_SUB,
            &tuning.drx_tune1a.to_le_bytes(),
        )?;
        self.write_register(
            Register::DrxTune,
            DRX_TUNE1B_SUB,
            &tuning.drx_tune1b.to_le_bytes(),
        )?;
        self.write_register(
            Register::DrxTune,
            DRX_TUNE2_SUB,
            &tuning.drx_tune2.to_le_bytes(),
        )?;
        self.write_register(
            Register::DrxTune,
            DRX_TUNE4H_SUB,
            &tuning.drx_tune4h.to_le_bytes(),
        )?;

        self.write_register(Register::RfConf, RF_RXCTRLH_SUB, &[tuning.rf_rxctrlh])?;
        self.write_register(
            Register::RfConf,
            RF_TXCTRL_SUB,
            &tuning.rf_txctrl.to_le_bytes(),
        )?;
        self.write_register(Register::TxCal, TC_PGDELAY_SUB, &[tuning.tc_pgdelay])?;

        self.write_register(
            Register::FsCtrl,
            FS_PLLCFG_SUB,
            &tuning.fspllcfg.to_le_bytes(),
        )?;
        self.write_register(Register::FsCtrl, FS_PLLTUNE_SUB, &[tuning.fsplltune])?;

        self.write_register(Register::LdeIf, LDE_CFG1_SUB, &[0x0D])?;
        self.write_register(
            Register::LdeIf,
            LDE_CFG2_SUB,
            &tuning.lde_cfg2.to_le_bytes(),
        )?;
        self.write_register(
            Register::LdeIf,
            LDE_REPC_SUB,
            &tuning.lde_repc.to_le_bytes(),
        )?;
        self.write_register(
            Register::TxPower,
            NO_SUBADDRESS,
            &tuning.tx_power.to_le_bytes(),
        )?;

        let xtal_trim = self.read_otp(0x01E)?[0];
        let fs_xtalt = if xtal_trim == 0 {
            0x70
        } else {
            (xtal_trim & 0x1F) | 0x60
        };
        self.write_register(Register::FsCtrl, FS_XTALT_SUB, &[fs_xtalt])?;
        Ok(())
    }

    fn manage_lde(&mut self, delay: &mut impl DelayNs) -> Result<(), Error<SPI::Error, PinE>> {
        let mut pmsc_ctrl0 = [0u8; LEN_PMSC_CTRL0];
        self.read_register(Register::Pmsc, PMSC_CTRL0_SUB, &mut pmsc_ctrl0)?;
        let mut otp_ctrl = [0u8; LEN_OTP_CTRL];
        self.read_register(Register::OtpIf, OTP_CTRL_SUB, &mut otp_ctrl)?;
        set_lde_load_preamble(&mut pmsc_ctrl0, &mut otp_ctrl);
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB, &pmsc_ctrl0[..2])?;
        self.write_register(Register::OtpIf, OTP_CTRL_SUB, &otp_ctrl)?;
        delay.delay_ms(5);
        set_lde_restore_preamble(&mut pmsc_ctrl0);
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB, &pmsc_ctrl0[..2])?;
        Ok(())
    }

    fn enable_clock(&mut self, mode: ClockMode) -> Result<(), Error<SPI::Error, PinE>> {
        let mut pmsc_ctrl0 = [0u8; LEN_PMSC_CTRL0];
        self.read_register(Register::Pmsc, PMSC_CTRL0_SUB, &mut pmsc_ctrl0)?;
        apply_clock_mode(&mut pmsc_ctrl0, mode);
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB, &pmsc_ctrl0[..2])
    }

    fn read_otp(&mut self, address: u16) -> Result<[u8; LEN_OTP_RDAT], Error<SPI::Error, PinE>> {
        let mut address_bytes = [0u8; LEN_OTP_ADDR];
        address_bytes.copy_from_slice(&address.to_le_bytes());
        self.write_register(Register::OtpIf, OTP_ADDR_SUB, &address_bytes)?;
        self.write_register(Register::OtpIf, OTP_CTRL_SUB, &[0x03])?;
        self.write_register(Register::OtpIf, OTP_CTRL_SUB, &[0x01])?;
        let mut data = [0u8; LEN_OTP_RDAT];
        self.read_register(Register::OtpIf, OTP_RDAT_SUB, &mut data)?;
        self.write_register(Register::OtpIf, OTP_CTRL_SUB, &[0x00])?;
        Ok(data)
    }

    fn idle(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.sys_ctrl = [0; LEN_SYS_CTRL];
        prepare_idle_state(&mut self.runtime, &mut self.sys_ctrl);
        let sys_ctrl = self.sys_ctrl;
        self.write_register(Register::SysCtrl, NO_SUBADDRESS, &sys_ctrl)
    }

    fn clear_interrupts(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.sys_mask = cleared_interrupt_mask();
        let sys_mask = self.sys_mask;
        self.write_register(Register::SysMask, NO_SUBADDRESS, &sys_mask)
    }

    fn clear_receive_status(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.clear_events(receive_status_clear_mask())
    }

    fn clear_transmit_status(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.clear_events(transmit_status_clear_mask())
    }

    fn read_received_length(&mut self) -> Result<usize, Error<SPI::Error, PinE>> {
        let mut info = [0u8; LEN_RX_FINFO];
        self.read_register(Register::RxFinfo, NO_SUBADDRESS, &mut info)?;
        Ok(self.runtime.rx_payload_len(info))
    }

    fn read_transmit_timestamp(&mut self) -> Result<DwTime, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; LEN_TX_STAMP];
        self.read_register(Register::TxTime, TX_STAMP_SUB, &mut bytes)?;
        Ok(DwTime::from_bytes(&bytes))
    }

    fn read_receive_timestamp(&mut self) -> Result<DwTime, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; LEN_RX_STAMP];
        self.read_register(Register::RxTime, RX_STAMP_SUB, &mut bytes)?;
        Ok(DwTime::from_bytes(&bytes))
    }

    fn read_system_timestamp(&mut self) -> Result<DwTime, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; LEN_RX_STAMP];
        self.read_register(Register::SysTime, NO_SUBADDRESS, &mut bytes)?;
        Ok(DwTime::from_bytes(&bytes))
    }

    fn read_receive_quality(&mut self) -> Result<f32, Error<SPI::Error, PinE>> {
        let mut noise = [0u8; LEN_STD_NOISE];
        let mut fp2 = [0u8; LEN_FP_AMPL2];
        self.read_register(Register::RxFqual, STD_NOISE_SUB, &mut noise)?;
        self.read_register(Register::RxFqual, FP_AMPL2_SUB, &mut fp2)?;
        Ok(compute_receive_quality(
            u16::from_le_bytes(noise),
            u16::from_le_bytes(fp2),
        ))
    }

    fn read_first_path_power(&mut self) -> Result<f32, Error<SPI::Error, PinE>> {
        let phy = self.runtime.phy.ok_or(Error::NotConfigured)?;
        let mut fp1 = [0u8; LEN_FP_AMPL1];
        let mut fp2 = [0u8; LEN_FP_AMPL2];
        let mut fp3 = [0u8; LEN_FP_AMPL3];
        let mut info = [0u8; LEN_RX_FINFO];
        self.read_register(Register::RxTime, FP_AMPL1_SUB, &mut fp1)?;
        self.read_register(Register::RxFqual, FP_AMPL2_SUB, &mut fp2)?;
        self.read_register(Register::RxFqual, FP_AMPL3_SUB, &mut fp3)?;
        self.read_register(Register::RxFinfo, NO_SUBADDRESS, &mut info)?;
        Ok(compute_first_path_power(
            phy,
            u16::from_le_bytes(fp1),
            u16::from_le_bytes(fp2),
            u16::from_le_bytes(fp3),
            extract_preamble_acc_count(info),
        ))
    }

    fn read_receive_power(&mut self) -> Result<f32, Error<SPI::Error, PinE>> {
        let phy = self.runtime.phy.ok_or(Error::NotConfigured)?;
        let mut cir = [0u8; LEN_CIR_PWR];
        let mut info = [0u8; LEN_RX_FINFO];
        self.read_register(Register::RxFqual, CIR_PWR_SUB, &mut cir)?;
        self.read_register(Register::RxFinfo, NO_SUBADDRESS, &mut info)?;
        Ok(compute_receive_power(
            phy,
            u16::from_le_bytes(cir),
            extract_preamble_acc_count(info),
        ))
    }

    fn correct_receive_timestamp(&self, timestamp: DwTime, receive_power_dbm: f32) -> DwTime {
        self.runtime
            .correct_receive_timestamp(timestamp, receive_power_dbm)
    }

    fn read_register(
        &mut self,
        register: Register,
        subaddress: u16,
        buffer: &mut [u8],
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let header = build_header(register, subaddress, false);
        let header_len = header_len(subaddress);
        let mut operations = [
            Operation::Write(&header[..header_len]),
            Operation::Read(buffer),
        ];
        self.spi.transaction(&mut operations).map_err(Error::Spi)
    }

    fn write_register(
        &mut self,
        register: Register,
        subaddress: u16,
        data: &[u8],
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let header = build_header(register, subaddress, true);
        let header_len = header_len(subaddress);
        let mut operations = [
            Operation::Write(&header[..header_len]),
            Operation::Write(data),
        ];
        self.spi.transaction(&mut operations).map_err(Error::Spi)
    }
}
