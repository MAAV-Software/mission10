//! Async DW1000 driver implementation.

use embedded_hal::digital::{InputPin, OutputPin};
use embedded_hal::spi::Operation;
use embedded_hal_async::{delay::DelayNs, digital::Wait, spi::SpiDevice};

use crate::config::{RadioConfig, RxOptions, TxOptions, ValidatedPhyConfig};
use crate::constants::{GPIO_MODE_SUB, LEN_GPIO_MODE};
use crate::device::{
    DelayedTransmit, DeviceIdentity, DxTimeDeadline, RadioDebugState, RxFrame, SignalMetrics,
    SysStatus, Timestamps,
};
use crate::driver_core::{
    apply_clock_mode, build_header, cleared_interrupt_mask, compose_base_register_fields,
    compose_gpio_led_mode, compose_led_blink_enable, compose_led_clock_enable,
    compose_phy_register_fields, compose_receive_sys_ctrl, compose_transmit_sys_ctrl,
    compute_first_path_power, compute_receive_power, compute_receive_quality,
    extract_preamble_acc_count, header_len, prepare_idle_state, select_tuning_values,
    set_lde_load_preamble, set_lde_restore_preamble, transmit_status_clear_mask,
    trx_status_clear_mask, ClockMode, DriverRuntime,
};
use crate::error::Error;
use crate::registers::{
    sys_status_from_bytes, sys_status_to_bytes, Register, AGC_TUNE1_SUB, AGC_TUNE2_SUB,
    AGC_TUNE3_SUB, AON_CFG1_SUB, CIR_PWR_SUB, DRX_SFDTOC_SUB, DRX_TUNE0B_SUB, DRX_TUNE1A_SUB,
    DRX_TUNE1B_SUB, DRX_TUNE2_SUB, DRX_TUNE4H_SUB, EXT_SYNC_CTRL_SUB, FP_AMPL1_SUB, FP_AMPL2_SUB,
    FP_AMPL3_SUB, FS_PLLCFG_SUB, FS_PLLTUNE_SUB, FS_XTALT_SUB, HSRBP_BYTE_BIT, ICRBP_BYTE_BIT,
    LDE_CFG1_SUB, LDE_CFG2_SUB, LDE_REPC_SUB, LDE_RXANTD_SUB, LEN_CHAN_CTRL, LEN_CIR_PWR,
    LEN_FP_AMPL1, LEN_FP_AMPL2, LEN_FP_AMPL3, LEN_LDE_RXANTD, LEN_OTP_ADDR, LEN_OTP_CTRL,
    LEN_OTP_RDAT, LEN_PANADR, LEN_PMSC_CTRL0, LEN_PMSC_LEDC, LEN_RX_FINFO, LEN_RX_STAMP,
    LEN_STD_NOISE, LEN_SYS_CFG, LEN_SYS_CTRL, LEN_SYS_MASK, LEN_SYS_STATE, LEN_SYS_STATUS,
    LEN_TX_ANTD, LEN_TX_FCTRL, LEN_TX_STAMP, NO_SUBADDRESS, OTP_ADDR_SUB, OTP_CTRL_SUB,
    OTP_RDAT_SUB, OTP_SF_SUB, PMSC_CTRL0_SUB, PMSC_LEDC_SUB, RF_RXCTRLH_SUB, RF_TXCTRL_SUB,
    RX_STAMP_SUB, SFD_LENGTH_SUB, STD_NOISE_SUB, SYS_CTRL_HRBT_SUB, SYS_STATUS_BUFFER_POINTER_SUB,
    TC_PGDELAY_SUB, TX_STAMP_SUB,
};
use crate::time::DwTime;

/// Async DW1000 driver.
#[derive(Debug)]
pub struct AsyncDw1000<SPI, IRQ, RST> {
    spi: SPI,
    irq: IRQ,
    reset: RST,
    sys_cfg: [u8; LEN_SYS_CFG],
    sys_ctrl: [u8; LEN_SYS_CTRL],
    sys_mask: [u8; LEN_SYS_MASK],
    tx_fctrl: [u8; LEN_TX_FCTRL],
    chan_ctrl: [u8; LEN_CHAN_CTRL],
    panadr: [u8; LEN_PANADR],
    xtal_trim: u8,
    runtime: DriverRuntime,
}

impl<SPI, IRQ, RST, PinE> AsyncDw1000<SPI, IRQ, RST>
where
    SPI: SpiDevice,
    IRQ: InputPin<Error = PinE> + Wait<Error = PinE>,
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
            xtal_trim: 0x70,
            runtime: DriverRuntime::new(),
        }
    }

    /// Initializes the DW1000 and applies the supplied configuration.
    pub async fn init(
        &mut self,
        delay: &mut impl DelayNs,
        config: &RadioConfig,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        self.hard_reset(delay).await?;
        self.enable_clock(ClockMode::Auto).await?;
        delay.delay_ms(5).await;
        self.clear_interrupts().await?;
        self.enable_clock(ClockMode::Xti).await?;
        delay.delay_ms(5).await;
        self.load_otp_tuning().await?;
        self.manage_lde(delay).await?;
        self.enable_clock(ClockMode::Auto).await?;
        delay.delay_ms(5).await;
        self.write_register(Register::Aon, AON_CFG1_SUB, &[0x00])
            .await?;
        self.reconfigure(config).await
    }

    /// Applies a new radio configuration without performing a reset.
    pub async fn reconfigure(
        &mut self,
        config: &RadioConfig,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let phy = self.runtime.reconfigure(config)?;

        self.idle().await?;
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

        self.apply_phy_config(phy).await?;
        let panadr = self.panadr;
        self.write_register(Register::PanAdr, NO_SUBADDRESS, &panadr)
            .await?;
        self.write_register(
            Register::Eui,
            NO_SUBADDRESS,
            &config.address.identity.eui.to_register_bytes(),
        )
        .await?;
        let sys_cfg = self.sys_cfg;
        let sys_mask = self.sys_mask;
        let chan_ctrl = self.chan_ctrl;
        let tx_fctrl = self.tx_fctrl;
        self.write_register(Register::SysCfg, NO_SUBADDRESS, &sys_cfg)
            .await?;
        self.write_register(Register::SysMask, NO_SUBADDRESS, &sys_mask)
            .await?;
        self.write_register(Register::ChanCtrl, NO_SUBADDRESS, &chan_ctrl)
            .await?;
        self.write_register(Register::TxFctrl, NO_SUBADDRESS, &tx_fctrl)
            .await?;
        self.apply_tuning(phy).await?;
        let antenna = self.runtime.antenna_delay.to_time_bytes();
        self.write_register(Register::TxAntd, NO_SUBADDRESS, &antenna[..LEN_TX_ANTD])
            .await?;
        self.write_register(Register::LdeIf, LDE_RXANTD_SUB, &antenna[..LEN_LDE_RXANTD])
            .await?;
        Ok(())
    }

    /// Starts a receive session.
    pub async fn start_receive(
        &mut self,
        options: RxOptions,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        self.force_idle_and_sync_rx().await?;
        self.sys_ctrl = [0; LEN_SYS_CTRL];
        self.runtime.begin_receive_session(options.permanent);
        if let Some(delay) = options.delayed_time {
            let deadline = self.schedule_delayed_receive(delay).await?;
            self.write_register(
                Register::DxTime,
                NO_SUBADDRESS,
                &deadline.value().to_bytes(),
            )
            .await?;
        }
        compose_receive_sys_ctrl(
            &mut self.sys_ctrl,
            self.runtime.frame_check,
            options.delayed_time.is_some(),
        );
        let sys_ctrl = self.sys_ctrl;
        self.write_register(Register::SysCtrl, NO_SUBADDRESS, &sys_ctrl)
            .await?;
        Ok(())
    }

    /// Transmits a frame.
    pub async fn transmit(
        &mut self,
        frame: &[u8],
        options: TxOptions,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let schedule = match options.delayed_time {
            Some(delay) => Some(self.schedule_delayed_transmit(delay).await?),
            None => None,
        };
        self.transmit_inner(frame, schedule, options.wait_for_response)
            .await
    }

    /// Transmits a frame with an already-computed delayed-transmit schedule.
    pub async fn transmit_at(
        &mut self,
        frame: &[u8],
        schedule: DelayedTransmit,
        wait_for_response: bool,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        self.transmit_inner(frame, Some(schedule), wait_for_response)
            .await
    }

    async fn transmit_inner(
        &mut self,
        frame: &[u8],
        schedule: Option<DelayedTransmit>,
        wait_for_response: bool,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let frame_len = self
            .runtime
            .checked_frame_len(frame.len())
            .map_err(|(len, max)| Error::FrameTooLong { len, max })?;

        self.idle().await?;
        self.clear_transmit_status().await?;
        self.write_register(Register::TxBuffer, NO_SUBADDRESS, frame)
            .await?;
        self.tx_fctrl[0] = (frame_len & 0xFF) as u8;
        self.tx_fctrl[1] &= 0xFC;
        self.tx_fctrl[1] |= ((frame_len >> 8) & 0x03) as u8;
        let tx_fctrl = self.tx_fctrl;
        self.write_register(Register::TxFctrl, NO_SUBADDRESS, &tx_fctrl)
            .await?;

        self.sys_ctrl = [0; LEN_SYS_CTRL];
        self.runtime.begin_transmit_session();
        if let Some(schedule) = schedule {
            self.write_register(
                Register::DxTime,
                NO_SUBADDRESS,
                &schedule.deadline().value().to_bytes(),
            )
            .await?;
        }
        compose_transmit_sys_ctrl(
            &mut self.sys_ctrl,
            self.runtime.frame_check,
            wait_for_response,
            schedule.is_some(),
        );
        let sys_ctrl = self.sys_ctrl;
        self.write_register(Register::SysCtrl, NO_SUBADDRESS, &sys_ctrl)
            .await?;
        self.runtime.complete_transmit_session();
        Ok(())
    }

    /// Computes a delayed transmit deadline and its antenna-adjusted timestamp.
    pub async fn schedule_delayed_transmit(
        &mut self,
        delay: DwTime,
    ) -> Result<DelayedTransmit, Error<SPI::Error, PinE>> {
        let now = self.read_system_timestamp().await? + delay;
        Ok(self.runtime.delayed_transmit(now))
    }

    /// Computes the raw system deadline at which delayed receive starts.
    pub async fn schedule_delayed_receive(
        &mut self,
        delay: DwTime,
    ) -> Result<DxTimeDeadline, Error<SPI::Error, PinE>> {
        let now = self.read_system_timestamp().await? + delay;
        Ok(self.runtime.delayed_deadline(now))
    }

    /// Reads a received frame into `buffer`.
    pub async fn read_frame<'a>(
        &mut self,
        buffer: &'a mut [u8],
    ) -> Result<RxFrame<'a>, Error<SPI::Error, PinE>> {
        self.read_frame_inner(buffer, true).await
    }

    /// Reads a received frame while preserving the raw hardware timestamp.
    pub async fn read_frame_raw<'a>(
        &mut self,
        buffer: &'a mut [u8],
    ) -> Result<RxFrame<'a>, Error<SPI::Error, PinE>> {
        self.read_frame_inner(buffer, false).await
    }

    async fn read_frame_inner<'a>(
        &mut self,
        buffer: &'a mut [u8],
        correct_timestamp: bool,
    ) -> Result<RxFrame<'a>, Error<SPI::Error, PinE>> {
        let status = self.read_sys_status().await?;
        self.runtime
            .validate_rx_status(status)
            .map_err(Error::Receive)?;

        let needed = self.read_received_length().await?;
        if buffer.len() < needed {
            return Err(Error::BufferTooSmall {
                len: buffer.len(),
                needed,
            });
        }
        self.read_register(Register::RxBuffer, NO_SUBADDRESS, &mut buffer[..needed])
            .await?;
        let timestamp = self.read_receive_timestamp().await?;
        let metrics = self.read_signal_metrics().await?;
        Ok(RxFrame {
            bytes: &buffer[..needed],
            timestamp: if correct_timestamp {
                self.correct_receive_timestamp(timestamp, metrics.receive_power_dbm)
            } else {
                timestamp
            },
            metrics,
            status,
        })
    }

    /// Reads the latest timestamps.
    pub async fn read_timestamps(&mut self) -> Result<Timestamps, Error<SPI::Error, PinE>> {
        Ok(Timestamps {
            tx: self.read_transmit_timestamp().await?,
            rx: self.read_receive_timestamp().await?,
            system: self.read_system_timestamp().await?,
        })
    }

    /// Reads the current receive metrics.
    pub async fn read_signal_metrics(&mut self) -> Result<SignalMetrics, Error<SPI::Error, PinE>> {
        Ok(SignalMetrics {
            receive_power_dbm: self.read_receive_power().await?,
            first_path_power_dbm: self.read_first_path_power().await?,
            quality: self.read_receive_quality().await?,
        })
    }

    /// Reads `SYS_STATUS`.
    pub async fn read_sys_status(&mut self) -> Result<SysStatus, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; LEN_SYS_STATUS];
        self.read_register(Register::SysStatus, NO_SUBADDRESS, &mut bytes)
            .await?;
        Ok(sys_status_from_bytes(&bytes))
    }

    /// Overrides the raw `TX_POWER` register value.
    pub async fn set_transmit_power(
        &mut self,
        tx_power: u32,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        self.write_register(Register::TxPower, NO_SUBADDRESS, &tx_power.to_le_bytes())
            .await
    }

    /// Resets only the receiver state machine, preserving radio configuration.
    pub async fn reset_receiver(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.force_idle_and_sync_rx().await?;
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB + 3, &[0xE0])
            .await?;
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB + 3, &[0xF0])
            .await
    }

    /// Reads a bounded snapshot of registers relevant to receiver diagnosis.
    pub async fn read_debug_state(&mut self) -> Result<RadioDebugState, Error<SPI::Error, PinE>> {
        let mut sys_status = [0u8; LEN_SYS_STATUS];
        let mut sys_state = [0u8; LEN_SYS_STATE];
        let mut sys_ctrl = [0u8; LEN_SYS_CTRL];
        let mut sys_cfg = [0u8; LEN_SYS_CFG];
        let mut sys_mask = [0u8; LEN_SYS_MASK];
        let mut pmsc_ctrl0 = [0u8; LEN_PMSC_CTRL0];
        self.read_register(Register::SysStatus, NO_SUBADDRESS, &mut sys_status)
            .await?;
        self.read_register(Register::SysState, NO_SUBADDRESS, &mut sys_state)
            .await?;
        self.read_register(Register::SysCtrl, NO_SUBADDRESS, &mut sys_ctrl)
            .await?;
        self.read_register(Register::SysCfg, NO_SUBADDRESS, &mut sys_cfg)
            .await?;
        self.read_register(Register::SysMask, NO_SUBADDRESS, &mut sys_mask)
            .await?;
        self.read_register(Register::Pmsc, PMSC_CTRL0_SUB, &mut pmsc_ctrl0)
            .await?;
        Ok(RadioDebugState {
            sys_status: sys_status_from_bytes(&sys_status),
            sys_state: u40_from_le_bytes(sys_state),
            sys_ctrl: u32::from_le_bytes(sys_ctrl),
            sys_cfg: u32::from_le_bytes(sys_cfg),
            sys_mask: u32::from_le_bytes(sys_mask),
            pmsc_ctrl0: u32::from_le_bytes(pmsc_ctrl0),
        })
    }

    /// Clears the selected `SYS_STATUS` bits and re-arms permanent receive after RX/TX completion.
    pub async fn clear_events(
        &mut self,
        event_mask: SysStatus,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        self.write_register(
            Register::SysStatus,
            NO_SUBADDRESS,
            &sys_status_to_bytes(event_mask),
        )
        .await?;
        let should_restart_receive = self.runtime.should_restart_receive(event_mask);
        if should_restart_receive {
            self.start_receive(RxOptions {
                delayed_time: None,
                permanent: true,
            })
            .await?;
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

    /// Waits until the DW1000 IRQ line is asserted.
    pub async fn wait_for_irq(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.irq.wait_for_high().await.map_err(Error::Pin)
    }

    /// Enables RX and TX LED indicators on GPIO pins.
    /// GPIO2 will show RX activity and GPIO3 will show TX activity.
    pub async fn enable_leds(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        let mut pmsc_ctrl0 = [0u8; LEN_PMSC_CTRL0];
        self.read_register(Register::Pmsc, PMSC_CTRL0_SUB, &mut pmsc_ctrl0)
            .await?;
        compose_led_clock_enable(&mut pmsc_ctrl0);
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB, &pmsc_ctrl0)
            .await?;

        let mut pmsc_ledc = [0u8; LEN_PMSC_LEDC];
        self.read_register(Register::Pmsc, PMSC_LEDC_SUB, &mut pmsc_ledc)
            .await?;
        compose_led_blink_enable(&mut pmsc_ledc, 0x20);
        self.write_register(Register::Pmsc, PMSC_LEDC_SUB, &pmsc_ledc)
            .await?;

        let mut gpio_mode = [0u8; LEN_GPIO_MODE];
        self.read_register(Register::GpioCtrl, GPIO_MODE_SUB as u16, &mut gpio_mode)
            .await?;
        compose_gpio_led_mode(&mut gpio_mode);
        self.write_register(Register::GpioCtrl, GPIO_MODE_SUB as u16, &gpio_mode)
            .await?;
        Ok(())
    }

    async fn hard_reset(
        &mut self,
        delay: &mut impl DelayNs,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        self.reset.set_low().map_err(Error::Pin)?;
        delay.delay_ms(2).await;
        self.reset.set_high().map_err(Error::Pin)?;
        delay.delay_ms(10).await;
        Ok(())
    }

    async fn apply_phy_config(
        &mut self,
        phy: ValidatedPhyConfig,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let sfd_len = compose_phy_register_fields(
            &mut self.sys_cfg,
            &mut self.tx_fctrl,
            &mut self.chan_ctrl,
            phy,
        );
        self.write_register(Register::UsrSfd, SFD_LENGTH_SUB, &[sfd_len])
            .await?;
        Ok(())
    }

    async fn apply_tuning(
        &mut self,
        phy: ValidatedPhyConfig,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let tuning = select_tuning_values(phy).map_err(Error::InvalidConfig)?;

        self.write_register(
            Register::AgcTune,
            AGC_TUNE1_SUB,
            &tuning.agc_tune1.to_le_bytes(),
        )
        .await?;
        self.write_register(
            Register::AgcTune,
            AGC_TUNE2_SUB,
            &0x2502_A907u32.to_le_bytes(),
        )
        .await?;
        self.write_register(Register::AgcTune, AGC_TUNE3_SUB, &0x0035u16.to_le_bytes())
            .await?;

        self.write_register(
            Register::DrxTune,
            DRX_TUNE0B_SUB,
            &tuning.drx_tune0b.to_le_bytes(),
        )
        .await?;
        self.write_register(
            Register::DrxTune,
            DRX_TUNE1A_SUB,
            &tuning.drx_tune1a.to_le_bytes(),
        )
        .await?;
        self.write_register(
            Register::DrxTune,
            DRX_TUNE1B_SUB,
            &tuning.drx_tune1b.to_le_bytes(),
        )
        .await?;
        self.write_register(
            Register::DrxTune,
            DRX_TUNE2_SUB,
            &tuning.drx_tune2.to_le_bytes(),
        )
        .await?;
        self.write_register(
            Register::DrxTune,
            DRX_TUNE4H_SUB,
            &tuning.drx_tune4h.to_le_bytes(),
        )
        .await?;
        self.write_register(
            Register::DrxTune,
            DRX_SFDTOC_SUB,
            &tuning.drx_sfd_timeout.to_le_bytes(),
        )
        .await?;

        self.write_register(Register::RfConf, RF_RXCTRLH_SUB, &[tuning.rf_rxctrlh])
            .await?;
        self.write_register(
            Register::RfConf,
            RF_TXCTRL_SUB,
            &tuning.rf_txctrl.to_le_bytes(),
        )
        .await?;
        self.write_register(Register::TxCal, TC_PGDELAY_SUB, &[tuning.tc_pgdelay])
            .await?;

        self.write_register(
            Register::FsCtrl,
            FS_PLLCFG_SUB,
            &tuning.fspllcfg.to_le_bytes(),
        )
        .await?;
        self.write_register(Register::FsCtrl, FS_PLLTUNE_SUB, &[tuning.fsplltune])
            .await?;

        self.write_register(Register::LdeIf, LDE_CFG1_SUB, &[0x0D])
            .await?;
        self.write_register(
            Register::LdeIf,
            LDE_CFG2_SUB,
            &tuning.lde_cfg2.to_le_bytes(),
        )
        .await?;
        self.write_register(
            Register::LdeIf,
            LDE_REPC_SUB,
            &tuning.lde_repc.to_le_bytes(),
        )
        .await?;
        self.write_register(
            Register::TxPower,
            NO_SUBADDRESS,
            &tuning.tx_power.to_le_bytes(),
        )
        .await?;

        self.write_register(Register::FsCtrl, FS_XTALT_SUB, &[self.xtal_trim])
            .await?;
        Ok(())
    }

    async fn load_otp_tuning(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.write_register(Register::ExtSync, EXT_SYNC_CTRL_SUB, &[0x04])
            .await?;
        if self.read_otp(0x004).await?[0] != 0 {
            self.write_register(Register::OtpIf, OTP_SF_SUB, &[0x02])
                .await?;
        }
        let xtal_trim = self.read_otp(0x01E).await?[0];
        self.xtal_trim = if xtal_trim == 0 {
            0x70
        } else {
            (xtal_trim & 0x1F) | 0x60
        };
        Ok(())
    }

    async fn manage_lde(
        &mut self,
        delay: &mut impl DelayNs,
    ) -> Result<(), Error<SPI::Error, PinE>> {
        let mut pmsc_ctrl0 = [0u8; LEN_PMSC_CTRL0];
        self.read_register(Register::Pmsc, PMSC_CTRL0_SUB, &mut pmsc_ctrl0)
            .await?;
        let mut otp_ctrl = [0u8; LEN_OTP_CTRL];
        self.read_register(Register::OtpIf, OTP_CTRL_SUB, &mut otp_ctrl)
            .await?;
        set_lde_load_preamble(&mut pmsc_ctrl0, &mut otp_ctrl);
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB, &pmsc_ctrl0[..2])
            .await?;
        self.write_register(Register::OtpIf, OTP_CTRL_SUB, &otp_ctrl)
            .await?;
        delay.delay_ms(5).await;
        set_lde_restore_preamble(&mut pmsc_ctrl0);
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB, &pmsc_ctrl0[..2])
            .await?;
        Ok(())
    }

    async fn enable_clock(&mut self, mode: ClockMode) -> Result<(), Error<SPI::Error, PinE>> {
        let mut pmsc_ctrl0 = [0u8; LEN_PMSC_CTRL0];
        self.read_register(Register::Pmsc, PMSC_CTRL0_SUB, &mut pmsc_ctrl0)
            .await?;
        apply_clock_mode(&mut pmsc_ctrl0, mode);
        self.write_register(Register::Pmsc, PMSC_CTRL0_SUB, &pmsc_ctrl0[..2])
            .await
    }

    async fn read_otp(
        &mut self,
        address: u16,
    ) -> Result<[u8; LEN_OTP_RDAT], Error<SPI::Error, PinE>> {
        let mut address_bytes = [0u8; LEN_OTP_ADDR];
        address_bytes.copy_from_slice(&address.to_le_bytes());
        self.write_register(Register::OtpIf, OTP_ADDR_SUB, &address_bytes)
            .await?;
        self.write_register(Register::OtpIf, OTP_CTRL_SUB, &[0x03])
            .await?;
        self.write_register(Register::OtpIf, OTP_CTRL_SUB, &[0x01])
            .await?;
        let mut data = [0u8; LEN_OTP_RDAT];
        self.read_register(Register::OtpIf, OTP_RDAT_SUB, &mut data)
            .await?;
        self.write_register(Register::OtpIf, OTP_CTRL_SUB, &[0x00])
            .await?;
        Ok(data)
    }

    async fn idle(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.sys_ctrl = [0; LEN_SYS_CTRL];
        prepare_idle_state(&mut self.runtime, &mut self.sys_ctrl);
        let sys_ctrl = self.sys_ctrl;
        self.write_register(Register::SysCtrl, NO_SUBADDRESS, &sys_ctrl)
            .await
    }

    async fn force_idle_and_sync_rx(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        let disabled_mask = cleared_interrupt_mask();
        self.write_register(Register::SysMask, NO_SUBADDRESS, &disabled_mask)
            .await?;

        self.sys_ctrl = [0; LEN_SYS_CTRL];
        prepare_idle_state(&mut self.runtime, &mut self.sys_ctrl);
        let trxoff = [self.sys_ctrl[0]];
        self.write_register(Register::SysCtrl, NO_SUBADDRESS, &trxoff)
            .await?;

        let clear = sys_status_to_bytes(trx_status_clear_mask());
        self.write_register(Register::SysStatus, NO_SUBADDRESS, &clear[..4])
            .await?;

        let mut pointers = [0u8; 1];
        self.read_register(
            Register::SysStatus,
            SYS_STATUS_BUFFER_POINTER_SUB,
            &mut pointers,
        )
        .await?;
        let host = pointers[0] & (1 << HSRBP_BYTE_BIT) != 0;
        let radio = pointers[0] & (1 << ICRBP_BYTE_BIT) != 0;
        if host != radio {
            self.write_register(Register::SysCtrl, SYS_CTRL_HRBT_SUB, &[0x01])
                .await?;
        }

        let sys_mask = self.sys_mask;
        self.write_register(Register::SysMask, NO_SUBADDRESS, &sys_mask)
            .await
    }

    async fn clear_interrupts(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.sys_mask = cleared_interrupt_mask();
        let sys_mask = self.sys_mask;
        self.write_register(Register::SysMask, NO_SUBADDRESS, &sys_mask)
            .await
    }

    async fn clear_transmit_status(&mut self) -> Result<(), Error<SPI::Error, PinE>> {
        self.write_register(
            Register::SysStatus,
            NO_SUBADDRESS,
            &sys_status_to_bytes(transmit_status_clear_mask()),
        )
        .await
    }

    async fn read_received_length(&mut self) -> Result<usize, Error<SPI::Error, PinE>> {
        let mut info = [0u8; LEN_RX_FINFO];
        self.read_register(Register::RxFinfo, NO_SUBADDRESS, &mut info)
            .await?;
        Ok(self.runtime.rx_payload_len(info))
    }

    async fn read_transmit_timestamp(&mut self) -> Result<DwTime, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; LEN_TX_STAMP];
        self.read_register(Register::TxTime, TX_STAMP_SUB, &mut bytes)
            .await?;
        Ok(DwTime::from_bytes(&bytes))
    }

    async fn read_receive_timestamp(&mut self) -> Result<DwTime, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; LEN_RX_STAMP];
        self.read_register(Register::RxTime, RX_STAMP_SUB, &mut bytes)
            .await?;
        Ok(DwTime::from_bytes(&bytes))
    }

    async fn read_system_timestamp(&mut self) -> Result<DwTime, Error<SPI::Error, PinE>> {
        let mut bytes = [0u8; LEN_RX_STAMP];
        self.read_register(Register::SysTime, NO_SUBADDRESS, &mut bytes)
            .await?;
        Ok(DwTime::from_bytes(&bytes))
    }

    async fn read_receive_quality(&mut self) -> Result<f32, Error<SPI::Error, PinE>> {
        let mut noise = [0u8; LEN_STD_NOISE];
        let mut fp2 = [0u8; LEN_FP_AMPL2];
        self.read_register(Register::RxFqual, STD_NOISE_SUB, &mut noise)
            .await?;
        self.read_register(Register::RxFqual, FP_AMPL2_SUB, &mut fp2)
            .await?;
        Ok(compute_receive_quality(
            u16::from_le_bytes(noise),
            u16::from_le_bytes(fp2),
        ))
    }

    async fn read_first_path_power(&mut self) -> Result<f32, Error<SPI::Error, PinE>> {
        let phy = self.runtime.phy.ok_or(Error::NotConfigured)?;
        let mut fp1 = [0u8; LEN_FP_AMPL1];
        let mut fp2 = [0u8; LEN_FP_AMPL2];
        let mut fp3 = [0u8; LEN_FP_AMPL3];
        let mut info = [0u8; LEN_RX_FINFO];
        self.read_register(Register::RxTime, FP_AMPL1_SUB, &mut fp1)
            .await?;
        self.read_register(Register::RxFqual, FP_AMPL2_SUB, &mut fp2)
            .await?;
        self.read_register(Register::RxFqual, FP_AMPL3_SUB, &mut fp3)
            .await?;
        self.read_register(Register::RxFinfo, NO_SUBADDRESS, &mut info)
            .await?;
        Ok(compute_first_path_power(
            phy,
            u16::from_le_bytes(fp1),
            u16::from_le_bytes(fp2),
            u16::from_le_bytes(fp3),
            extract_preamble_acc_count(info),
        ))
    }

    async fn read_receive_power(&mut self) -> Result<f32, Error<SPI::Error, PinE>> {
        let phy = self.runtime.phy.ok_or(Error::NotConfigured)?;
        let mut cir = [0u8; LEN_CIR_PWR];
        let mut info = [0u8; LEN_RX_FINFO];
        self.read_register(Register::RxFqual, CIR_PWR_SUB, &mut cir)
            .await?;
        self.read_register(Register::RxFinfo, NO_SUBADDRESS, &mut info)
            .await?;
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

    async fn read_register(
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
        self.spi
            .transaction(&mut operations)
            .await
            .map_err(Error::Spi)
    }

    async fn write_register(
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
        self.spi
            .transaction(&mut operations)
            .await
            .map_err(Error::Spi)
    }
}

fn u40_from_le_bytes(bytes: [u8; 5]) -> u64 {
    u64::from(bytes[0])
        | (u64::from(bytes[1]) << 8)
        | (u64::from(bytes[2]) << 16)
        | (u64::from(bytes[3]) << 24)
        | (u64::from(bytes[4]) << 32)
}
