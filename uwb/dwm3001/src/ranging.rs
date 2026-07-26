use defmt::warn;
use dw3000_ng::DW3000;
use dw3000_ng::hl::SendTime;
use dw3000_ng::time::Instant as RadioInstant;
use embassy_embedded_hal::SetConfig;
use embassy_nrf::gpio::{Level, Output, OutputDrive};
use embassy_nrf::spim;
use embassy_time::{Delay, Duration, Instant, Timer};
use embedded_hal_bus::spi::ExclusiveDevice;
use mission10_uwb_protocol::air::{
    self, AIR_FRAME_MAX_NO_FCS, AirEnvelope, AirMessage, DecodedAirFrame, PAN_ID,
    REPORT_STATUS_CLOSE,
};
use mission10_uwb_protocol::clock::{
    ClockCorrelator, CorrelationSample, MAPPING_MAX_ERROR_US, q32_rate,
};
use mission10_uwb_protocol::host::{
    CompletedExchange, Diagnostic, HealthCounters, MissionEventTime, RadioConfiguration,
    RadioToHost,
};
use mission10_uwb_protocol::scheduler::{
    ExchangeId, FlightRoster, MAX_PAIRS, Pair, PairMacState, classify_range, due_pair, next_due_us,
    pair_index,
};
use mission10_uwb_protocol::{
    DTU_PER_US, Destination, NodeAddress, REPORT_TURNAROUND_US, ResponderTimestamps,
    TIMESTAMP_MASK, delayed_tx_time, distance_metres, scheduled_tx_matches, wrapping_delta,
};

use crate::Irqs;
use crate::board::{
    FALLBACK_ANTENNA_DELAY, REPLY_DELAY_US, RESPONSE_TIMEOUT_US, RadioHardware, RadioWait,
    diagnostic, finish_receiving, finish_sending, prepare_rx, prepare_tx, radio_config,
    recoverable_receive_error, reset_after_radio_failure, wait_receive, wait_send,
};
use crate::host::{
    clear_measurements, host_connected, latest_ego_state, publish, transport_counters,
    try_clock_probe_sent, try_clock_reply, try_configuration, update_radio_counters,
};

const PHY_FCS_LEN: usize = 2;
const CONFIGURATION_POLL_MS: u64 = 10;
const POLL_PROGRAM_LEAD_US: u64 = 1_000;
const RX_RELEASE_LEAD_US: u64 = 1_000;
const CLOCK_PROBE_INTERVAL_MS: u64 = 250;
const MAX_IDLE_LISTEN_US: u64 = 100_000;

fn publish_error(diagnostic: Diagnostic) {
    warn!("radio diagnostic={=u8}", diagnostic as u8);
    publish(RadioToHost::Error { diagnostic });
}

const RETAINED_RESET_REPORT_MS: u64 = 3_000;
const RETAINED_RESET_REPORT_INTERVAL_MS: u64 = 750;

fn publish_reset_diagnostics(
    previous_radio_failure: Option<Diagnostic>,
    previous_watchdog_reset: bool,
) {
    if let Some(diagnostic) = previous_radio_failure {
        publish_error(diagnostic);
        publish_error(Diagnostic::RadioReset);
    }
    if previous_watchdog_reset {
        publish_error(Diagnostic::WatchdogReset);
    }
}

async fn report_previous_reset(
    wait: &mut RadioWait<'_>,
    previous_radio_failure: Option<Diagnostic>,
    previous_watchdog_reset: bool,
) {
    let reset_was_retained = previous_radio_failure.is_some() || previous_watchdog_reset;
    if !reset_was_retained {
        return;
    }

    // Repeat through the stable USB window: CDC may become configured and
    // consume a one-shot event before the host process has opened the tty.
    // Four radio-failure pairs exactly fit the diagnostic queue if no host is
    // attached, so this reporting path cannot create queue drops by itself.
    for _ in 0..RETAINED_RESET_REPORT_MS / RETAINED_RESET_REPORT_INTERVAL_MS {
        publish_reset_diagnostics(previous_radio_failure, previous_watchdog_reset);
        wait.pet_watchdog();
        Timer::after_millis(RETAINED_RESET_REPORT_INTERVAL_MS).await;
    }
    wait.pet_watchdog();
}

pub async fn run(hw: RadioHardware) {
    let mut wait = RadioWait::new(hw.irq, hw.watchdog);
    report_previous_reset(
        &mut wait,
        hw.previous_radio_failure,
        hw.previous_watchdog_reset,
    )
    .await;

    // DW_RST is open-drain. Driving high disconnects the nRF output and lets
    // the module pull-up release reset.
    let mut reset = Output::new(hw.reset, Level::High, OutputDrive::Standard0Disconnect1);
    reset.set_low();
    Timer::after_millis(2).await;
    reset.set_high();
    Timer::after_millis(5).await;

    let mut spi_config = spim::Config::default();
    spi_config.frequency = spim::Frequency::M4;
    let spi = spim::Spim::new(hw.spi, Irqs, hw.sck, hw.miso, hw.mosi, spi_config);
    let cs = Output::new(hw.cs, Level::High, OutputDrive::Standard);
    let spi_device = match ExclusiveDevice::new(spi, cs, Delay) {
        Ok(device) => device,
        Err(_) => reset_after_radio_failure(Diagnostic::Unknown),
    };
    let mut radio = DW3000::new(spi_device);
    let id = match radio.ll().dev_id().read().await {
        Ok(id) => id,
        Err(_) => reset_after_radio_failure(Diagnostic::Spi),
    };
    publish(RadioToHost::RadioId {
        ridtag: id.ridtag(),
        model: id.model(),
        version: id.ver(),
        revision: id.rev(),
    });

    let tx_power = radio
        .read_otp(0x011)
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    let antenna = radio
        .read_otp(0x01a)
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    let xtal = radio
        .read_otp(0x01e)
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    let otp_revision = radio
        .read_otp(0x01f)
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    publish(RadioToHost::Otp {
        tx_power,
        antenna,
        xtal,
        revision: otp_revision,
    });

    let radio = match radio.init().await {
        Ok(radio) => radio,
        Err(error) => reset_after_radio_failure(diagnostic(&error)),
    };
    let mut radio = match radio.config(radio_config(), Delay).await {
        Ok(radio) => radio,
        Err(error) => reset_after_radio_failure(diagnostic(&error)),
    };

    // The DW3000 must be initialized on a slow SPI clock. Once it is in its
    // configured IDLE_PLL state, both it and nRF52833 SPIM3 support 32 MHz.
    let mut fast_spi_config = spim::Config::default();
    fast_spi_config.frequency = spim::Frequency::M32;
    radio
        .ll()
        .bus()
        .bus_mut()
        .set_config(&fast_spi_config)
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Unknown));

    let otp_tx_delay = antenna as u16;
    let otp_rx_delay = (antenna >> 16) as u16;
    let use_fallback = cfg!(feature = "engineering-sample")
        || otp_tx_delay == 0
        || otp_rx_delay == 0
        || antenna == u32::MAX;
    let (rx_delay, tx_delay) = if use_fallback {
        (FALLBACK_ANTENNA_DELAY, FALLBACK_ANTENNA_DELAY)
    } else {
        (otp_rx_delay, otp_tx_delay)
    };
    if radio.set_antenna_delay(rx_delay, tx_delay).await.is_err() {
        reset_after_radio_failure(Diagnostic::Spi);
    }
    if tx_power != 0
        && tx_power != u32::MAX
        && radio
            .ll()
            .tx_power()
            .write(|w| w.value(tx_power))
            .await
            .is_err()
    {
        reset_after_radio_failure(Diagnostic::Spi);
    }
    publish(RadioToHost::Ready { rx_delay, tx_delay });
    let mut initial_counters = wait.counters().health_counters();
    if hw.previous_radio_failure.is_some() {
        initial_counters.radio_reinitializations = 1;
    }
    update_radio_counters(initial_counters);
    publish(RadioToHost::Health {
        request_id: 0,
        counters: initial_counters,
    });

    let mut runtime = Runtime::default();
    runtime.health.radio_reinitializations = initial_counters.radio_reinitializations;
    let configuration = wait_for_configuration(&mut wait).await;
    apply_configuration(&mut radio, configuration).await;
    clear_measurements();
    publish(RadioToHost::Configured { configuration });
    runtime.reconfigure(configuration);

    loop {
        wait.pet_watchdog();
        if let Some(configuration) = try_configuration() {
            apply_configuration(&mut radio, configuration).await;
            clear_measurements();
            publish(RadioToHost::Configured { configuration });
            runtime.reconfigure(configuration);
        }
        refresh_clock(&mut runtime);
        let now_us = Instant::now().as_micros();
        for state in &mut runtime.pair_states {
            state.expire(now_us);
        }
        if let Some(pair) = due_pair(
            runtime.roster(),
            runtime.local,
            now_us,
            &runtime.pair_states,
        ) {
            let exchange_id = runtime.next_exchange_id();
            let index = pair_index(runtime.roster(), pair)
                .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidConfiguration));
            runtime.pair_states[index].begin_attempt(now_us);
            runtime.active_pair = Some(pair);
            let (returned, outcome) = initiate_exchange(
                radio,
                &mut wait,
                &mut runtime,
                pair.responder,
                exchange_id,
                tx_delay,
            )
            .await;
            radio = returned;
            runtime.finish_attempt(pair, outcome);
        } else {
            let next_due = next_due_us(runtime.roster(), runtime.local, &runtime.pair_states);
            let until_due = next_due.map_or(MAX_IDLE_LISTEN_US, |due| due.saturating_sub(now_us));
            if until_due <= RX_RELEASE_LEAD_US {
                if let Some(due) = next_due {
                    Timer::at(Instant::from_micros(due)).await;
                }
                continue;
            }
            let listen_us = until_due
                .saturating_sub(RX_RELEASE_LEAD_US)
                .clamp(1, MAX_IDLE_LISTEN_US);
            let (returned, outcome) = respond_exchange(
                radio,
                &mut wait,
                &mut runtime,
                u32::try_from(listen_us).unwrap_or(u32::MAX),
                tx_delay,
            )
            .await;
            radio = returned;
            if let Some(outcome) = outcome {
                let pair = Pair::new(runtime.local, outcome.exchange.peer)
                    .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidConfiguration));
                runtime.finish_attempt(pair, Some(outcome));
            }
        }
        runtime.active_pair = None;
        synchronize_health(&wait, runtime.health);
    }
}

struct ExchangeOutcome {
    exchange: CompletedExchange,
    close: bool,
}

struct Runtime {
    configuration: RadioConfiguration,
    local: NodeAddress,
    pair_states: [PairMacState; MAX_PAIRS],
    clock: ClockCorrelator<32>,
    next_clock_probe: Instant,
    clock_request_id: u16,
    clock_generation: Option<u32>,
    pending_clock_probe: Option<crate::host::ClockProbeSent>,
    mapping_was_unavailable: bool,
    active_pair: Option<Pair>,
    next_exchange_id: u16,
    last_received_exchange: [Option<(ExchangeId, u64)>; mission10_uwb_protocol::host::MAX_PEERS],
    mac_sequence: u8,
    peer_missed_cycles: [u8; mission10_uwb_protocol::host::MAX_PEERS],
    peer_stale: [bool; mission10_uwb_protocol::host::MAX_PEERS],
    spi_errors_since_range: u8,
    health: HealthCounters,
    rx_buf: [u8; AIR_FRAME_MAX_NO_FCS + PHY_FCS_LEN],
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            configuration: RadioConfiguration::default(),
            local: NodeAddress::default(),
            pair_states: [PairMacState::new(0); MAX_PAIRS],
            clock: ClockCorrelator::new(q32_rate(1)),
            next_clock_probe: Instant::from_micros(0),
            clock_request_id: 0,
            clock_generation: None,
            pending_clock_probe: None,
            mapping_was_unavailable: true,
            active_pair: None,
            next_exchange_id: 0,
            last_received_exchange: [None; mission10_uwb_protocol::host::MAX_PEERS],
            mac_sequence: 0,
            peer_missed_cycles: [0; mission10_uwb_protocol::host::MAX_PEERS],
            peer_stale: [false; mission10_uwb_protocol::host::MAX_PEERS],
            spi_errors_since_range: 0,
            health: HealthCounters::default(),
            rx_buf: [0; AIR_FRAME_MAX_NO_FCS + PHY_FCS_LEN],
        }
    }
}

impl Runtime {
    fn reconfigure(&mut self, configuration: RadioConfiguration) {
        self.configuration = configuration;
        self.local = configuration.node_address;
        let boot_nonce = Instant::now().as_ticks() as u32 ^ u32::from(self.local.get());
        self.pair_states = [PairMacState::new(boot_nonce); MAX_PAIRS];
        let now_us = Instant::now().as_micros();
        for (index, pair) in self.roster().pairs().enumerate() {
            self.pair_states[index].initialize(now_us, pair);
        }
        self.clock.clear();
        self.next_clock_probe = Instant::from_micros(0);
        self.clock_generation = None;
        self.pending_clock_probe = None;
        self.mapping_was_unavailable = true;
        self.active_pair = None;
        self.next_exchange_id = 0;
        self.last_received_exchange = [None; mission10_uwb_protocol::host::MAX_PEERS];
        self.peer_missed_cycles = [0; mission10_uwb_protocol::host::MAX_PEERS];
        self.peer_stale = [false; mission10_uwb_protocol::host::MAX_PEERS];
        self.spi_errors_since_range = 0;
    }

    fn roster(&self) -> FlightRoster {
        self.configuration
            .roster()
            .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidConfiguration))
    }

    fn next_exchange_id(&mut self) -> ExchangeId {
        let exchange_id = ExchangeId::new(self.next_exchange_id);
        self.next_exchange_id = self.next_exchange_id.wrapping_add(1);
        exchange_id
    }

    fn finish_attempt(&mut self, pair: Pair, outcome: Option<ExchangeOutcome>) {
        let index = pair_index(self.roster(), pair)
            .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidConfiguration));
        let peer = if pair.initiator == self.local {
            pair.responder
        } else {
            pair.initiator
        };
        let peer_index = self
            .configuration
            .peers()
            .and_then(|peers| peers.iter().position(|candidate| *candidate == peer))
            .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidConfiguration));
        let now_us = Instant::now().as_micros();
        if let Some(outcome) = outcome {
            self.peer_missed_cycles[peer_index] = 0;
            self.peer_stale[peer_index] = false;
            self.pair_states[index].record_success(now_us, outcome.close);
            self.health.exchange_completions = self.health.exchange_completions.wrapping_add(1);
            publish(RadioToHost::CompletedExchange {
                exchange: outcome.exchange,
            });
        } else {
            self.health.exchange_timeouts = self.health.exchange_timeouts.wrapping_add(1);
            self.health.contention_backoffs = self.health.contention_backoffs.wrapping_add(1);
            self.pair_states[index].record_failure(now_us);
            self.peer_missed_cycles[peer_index] =
                self.peer_missed_cycles[peer_index].saturating_add(1);
            if self.peer_missed_cycles[peer_index] >= 3 && !self.peer_stale[peer_index] {
                self.peer_stale[peer_index] = true;
                self.health.peer_stale_transitions =
                    self.health.peer_stale_transitions.wrapping_add(1);
            }
        }
    }
}

fn refresh_clock(runtime: &mut Runtime) {
    if !host_connected() {
        return;
    }
    let now = Instant::now();
    while let Some(sent) = try_clock_probe_sent() {
        runtime.pending_clock_probe = Some(sent);
    }
    while let Some(reply) = try_clock_reply() {
        let local_rx_us = reply.local_rx_us;
        let Some(sent) = runtime
            .pending_clock_probe
            .filter(|sent| sent.request_id == reply.request_id)
        else {
            runtime.health.clock_samples_rejected =
                runtime.health.clock_samples_rejected.wrapping_add(1);
            continue;
        };
        runtime.pending_clock_probe = None;
        let Some(host_elapsed) = reply.mission_tx_us.checked_sub(reply.mission_rx_us) else {
            runtime.health.clock_samples_rejected =
                runtime.health.clock_samples_rejected.wrapping_add(1);
            continue;
        };
        let local_elapsed = local_rx_us.saturating_sub(sent.local_tx_us);
        let Some(residual) = local_elapsed.checked_sub(host_elapsed) else {
            runtime.health.clock_samples_rejected =
                runtime.health.clock_samples_rejected.wrapping_add(1);
            continue;
        };
        let error_us = reply
            .source_error_us
            .saturating_add(u32::try_from(residual / 2).unwrap_or(u32::MAX));
        if runtime.clock_generation != Some(reply.mission_generation) {
            if runtime.clock_generation.is_some() {
                runtime.health.clock_generation_changes =
                    runtime.health.clock_generation_changes.wrapping_add(1);
            }
            runtime.clock_generation = Some(reply.mission_generation);
        }
        let sample = CorrelationSample {
            source: sent.local_tx_us + local_elapsed / 2,
            target: reply.mission_rx_us + host_elapsed / 2,
            error_us,
            generation: reply.mission_generation,
        };
        if runtime.clock.push(sample).is_ok() {
            runtime.health.clock_samples_accepted =
                runtime.health.clock_samples_accepted.wrapping_add(1);
        } else {
            runtime.health.clock_samples_rejected =
                runtime.health.clock_samples_rejected.wrapping_add(1);
        }
        let model = runtime.clock.model();
        publish(RadioToHost::ClockStatus {
            generation: reply.mission_generation,
            error_us: model.map_or(error_us, |value| value.error_at(local_rx_us)),
            age_us: model.map_or(0, |value| {
                u32::try_from(local_rx_us.saturating_sub(value.anchor_source())).unwrap_or(u32::MAX)
            }),
            mapped: model.is_some_and(|value| value.maps_at_source(local_rx_us)),
        });
    }
    if now >= runtime.next_clock_probe {
        runtime.next_clock_probe = now + Duration::from_millis(CLOCK_PROBE_INTERVAL_MS);
        let request_id = runtime.clock_request_id;
        runtime.clock_request_id = runtime.clock_request_id.wrapping_add(1);
        publish(RadioToHost::ClockProbe { request_id });
    }
}

async fn wait_for_configuration(wait: &mut RadioWait<'_>) -> RadioConfiguration {
    loop {
        if let Some(configuration) = try_configuration() {
            return configuration;
        }
        wait.pet_watchdog();
        Timer::after_millis(CONFIGURATION_POLL_MS).await;
    }
}

async fn apply_configuration<SPI>(
    radio: &mut DW3000<SPI, dw3000_ng::Ready>,
    configuration: RadioConfiguration,
) where
    SPI: embedded_hal_async::spi::SpiDevice<u8>,
{
    if configuration.roster().is_none() {
        reset_after_radio_failure(Diagnostic::InvalidConfiguration);
    }
    if radio
        .ll()
        .panadr()
        .write(|w| {
            w.pan_id(PAN_ID)
                .short_addr(configuration.node_address.get())
        })
        .await
        .is_err()
    {
        reset_after_radio_failure(Diagnostic::Spi);
    }
}

async fn initiate_exchange<SPI>(
    mut radio: DW3000<SPI, dw3000_ng::Ready>,
    wait: &mut RadioWait<'_>,
    runtime: &mut Runtime,
    peer: NodeAddress,
    exchange_id: ExchangeId,
    tx_delay: u16,
) -> (DW3000<SPI, dw3000_ng::Ready>, Option<ExchangeOutcome>)
where
    SPI: embedded_hal_async::spi::SpiDevice<u8>,
{
    if let Err(error) = prepare_tx(&mut radio).await {
        recover_exchange(error, wait, runtime);
        return (radio, None);
    }
    let poll = encode_air(
        runtime.local,
        &mut runtime.mac_sequence,
        Destination::Node(peer),
        AirEnvelope::new(
            exchange_id,
            AirMessage::Poll {
                state: latest_ego_state(),
            },
        ),
    )
    .unwrap_or_else(|error| reset_after_radio_failure(error));
    let now = radio
        .sys_time()
        .await
        .unwrap_or_else(|_| reset_after_radio_failure(Diagnostic::Spi));
    let poll_at_value = delayed_tx_time(u64::from(now) << 8, POLL_PROGRAM_LEAD_US as u32);
    let predicted_poll_tx = poll_at_value.wrapping_add(u64::from(tx_delay)) & TIMESTAMP_MASK;
    let poll_at = RadioInstant::new(poll_at_value)
        .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidTimestamp));
    let mut sending = radio
        .send_raw(poll.bytes(), SendTime::Delayed(poll_at), &radio_config())
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(diagnostic(&error)));
    let poll_tx = wait_send(&mut sending, wait)
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(error));
    radio = finish_sending(sending, wait).await;
    if !scheduled_tx_matches(predicted_poll_tx, poll_tx.value()) {
        recover_exchange(Diagnostic::DelayedSendTooLate, wait, runtime);
        return (radio, None);
    }
    runtime.health.polls_sent = runtime.health.polls_sent.wrapping_add(1);

    if let Err(error) = prepare_rx(&mut radio, Some(RESPONSE_TIMEOUT_US)).await {
        recover_exchange(error, wait, runtime);
        return (radio, None);
    }
    let mut receiving = radio
        .receive(radio_config())
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(diagnostic(&error)));
    let response = wait_receive(&mut receiving, &mut runtime.rx_buf, wait).await;
    radio = finish_receiving(receiving, wait).await;
    let (length, response_rx) = match response {
        Ok(response) => response,
        Err(error) if recoverable_receive_error(error) => {
            recover_exchange(error, wait, runtime);
            return (radio, None);
        }
        Err(error) => reset_after_radio_failure(error),
    };
    let response = match decode_air(&runtime.rx_buf, length, runtime.local) {
        Ok(frame) => frame,
        Err(error) => {
            recover_exchange(error, wait, runtime);
            runtime.health.malformed_air_frames =
                runtime.health.malformed_air_frames.wrapping_add(1);
            return (radio, None);
        }
    };
    let (poll_rx, response_tx, peer_state) = match response.envelope.message {
        AirMessage::Response {
            poll_rx,
            response_tx,
            state,
        } if response.source == peer && response.envelope.exchange_id == exchange_id => {
            runtime.health.responses_received = runtime.health.responses_received.wrapping_add(1);
            (poll_rx, response_tx, state)
        }
        _ => {
            unexpected(runtime);
            return (radio, None);
        }
    };

    let final_at_value = delayed_tx_time(response_rx.value(), REPLY_DELAY_US);
    let predicted_final_tx = final_at_value.wrapping_add(u64::from(tx_delay)) & TIMESTAMP_MASK;
    let final_frame = encode_air(
        runtime.local,
        &mut runtime.mac_sequence,
        Destination::Node(peer),
        AirEnvelope::new(
            exchange_id,
            AirMessage::Final {
                poll_tx: poll_tx.value(),
                response_rx: response_rx.value(),
                final_tx: predicted_final_tx,
            },
        ),
    )
    .unwrap_or_else(|error| reset_after_radio_failure(error));
    if let Err(error) = prepare_tx(&mut radio).await {
        recover_exchange(error, wait, runtime);
        return (radio, None);
    }
    let final_at = RadioInstant::new(final_at_value)
        .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidTimestamp));
    let mut sending = radio
        .send_raw(
            final_frame.bytes(),
            SendTime::Delayed(final_at),
            &radio_config(),
        )
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(diagnostic(&error)));
    let actual_final_tx = match wait_send(&mut sending, wait).await {
        Ok(timestamp) => timestamp,
        Err(error) => {
            recover_exchange(error, wait, runtime);
            let radio = finish_sending(sending, wait).await;
            return (radio, None);
        }
    };
    radio = finish_sending(sending, wait).await;
    if !scheduled_tx_matches(predicted_final_tx, actual_final_tx.value()) {
        recover_exchange(Diagnostic::DelayedSendTooLate, wait, runtime);
        return (radio, None);
    }
    runtime.health.finals_sent = runtime.health.finals_sent.wrapping_add(1);

    if let Err(error) = prepare_rx(&mut radio, Some(RESPONSE_TIMEOUT_US)).await {
        recover_exchange(error, wait, runtime);
        return (radio, None);
    }
    let mut receiving = radio
        .receive(radio_config())
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(diagnostic(&error)));
    let report = wait_receive(&mut receiving, &mut runtime.rx_buf, wait).await;
    radio = finish_receiving(receiving, wait).await;
    let (length, _) = match report {
        Ok(report) => report,
        Err(error) if recoverable_receive_error(error) => {
            recover_exchange(error, wait, runtime);
            return (radio, None);
        }
        Err(error) => reset_after_radio_failure(error),
    };
    let report = match decode_air(&runtime.rx_buf, length, runtime.local) {
        Ok(frame) => frame,
        Err(error) => {
            recover_exchange(error, wait, runtime);
            runtime.health.malformed_air_frames =
                runtime.health.malformed_air_frames.wrapping_add(1);
            return (radio, None);
        }
    };
    let (final_rx, close) = match report.envelope.message {
        AirMessage::Report { final_rx, status }
            if report.source == peer && report.envelope.exchange_id == exchange_id =>
        {
            runtime.health.reports_received = runtime.health.reports_received.wrapping_add(1);
            (final_rx, status & REPORT_STATUS_CLOSE != 0)
        }
        _ => {
            unexpected(runtime);
            return (radio, None);
        }
    };
    let Some(distance) = distance_metres(
        [poll_tx.value(), response_rx.value(), predicted_final_tx],
        ResponderTimestamps {
            poll_rx,
            response_tx,
            final_rx,
        },
    ) else {
        recover_exchange(Diagnostic::InvalidDistance, wait, runtime);
        return (radio, None);
    };
    let mission_event_time = map_event_time(&mut radio, runtime, actual_final_tx.value()).await;
    runtime.spi_errors_since_range = 0;
    (
        radio,
        Some(ExchangeOutcome {
            exchange: CompletedExchange {
                peer,
                exchange_id,
                range_event_time_dtu: actual_final_tx.value(),
                mission_event_time,
                millimetres: millimetres(distance),
                rssi_cdbm: i16::MIN,
                quality_flags: 0,
                state: peer_state,
            },
            close,
        }),
    )
}

async fn respond_exchange<SPI>(
    mut radio: DW3000<SPI, dw3000_ng::Ready>,
    wait: &mut RadioWait<'_>,
    runtime: &mut Runtime,
    listen_timeout_us: u32,
    tx_delay: u16,
) -> (DW3000<SPI, dw3000_ng::Ready>, Option<ExchangeOutcome>)
where
    SPI: embedded_hal_async::spi::SpiDevice<u8>,
{
    if let Err(error) = prepare_rx(&mut radio, Some(listen_timeout_us)).await {
        recover_exchange(error, wait, runtime);
        return (radio, None);
    }
    let mut receiving = radio
        .receive(radio_config())
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(diagnostic(&error)));
    let poll = wait_receive(&mut receiving, &mut runtime.rx_buf, wait).await;
    radio = finish_receiving(receiving, wait).await;
    let (length, poll_rx) = match poll {
        Ok(poll) => poll,
        Err(error) if recoverable_receive_error(error) => {
            if error == Diagnostic::RxFrameWaitTimeout {
                wait.completed();
            } else {
                recover_exchange(error, wait, runtime);
            }
            return (radio, None);
        }
        Err(error) => reset_after_radio_failure(error),
    };
    let poll = match decode_air(&runtime.rx_buf, length, runtime.local) {
        Ok(frame) => frame,
        Err(error) => {
            recover_exchange(error, wait, runtime);
            runtime.health.malformed_air_frames =
                runtime.health.malformed_air_frames.wrapping_add(1);
            return (radio, None);
        }
    };
    let peer = poll.source;
    let exchange_id = poll.envelope.exchange_id;
    let pair = Pair::new(runtime.local, peer)
        .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidConfiguration));
    let peer_index = runtime
        .configuration
        .peers()
        .and_then(|peers| peers.iter().position(|candidate| *candidate == peer));
    let received_now_us = Instant::now().as_micros();
    let duplicate = peer_index.is_some_and(|index| {
        runtime.last_received_exchange[index].is_some_and(|(previous, received_at)| {
            previous == exchange_id && received_now_us.saturating_sub(received_at) < 20_000
        })
    });
    let allowed = pair.initiator == peer && runtime.roster().contains(peer) && !duplicate;
    let peer_state = match poll.envelope.message {
        AirMessage::Poll { state } if allowed => {
            if let Some(index) = peer_index {
                runtime.last_received_exchange[index] = Some((exchange_id, received_now_us));
            }
            runtime.health.polls_received = runtime.health.polls_received.wrapping_add(1);
            runtime.active_pair = Some(pair);
            state
        }
        AirMessage::Poll { .. } => {
            unexpected(runtime);
            return (radio, None);
        }
        _ => {
            unexpected(runtime);
            return (radio, None);
        }
    };
    let response_at_value = delayed_tx_time(poll_rx.value(), REPLY_DELAY_US);
    let predicted_response_tx =
        response_at_value.wrapping_add(u64::from(tx_delay)) & TIMESTAMP_MASK;
    if let Err(error) = prepare_tx(&mut radio).await {
        recover_exchange(error, wait, runtime);
        return (radio, None);
    }
    let response = encode_air(
        runtime.local,
        &mut runtime.mac_sequence,
        Destination::Node(peer),
        AirEnvelope::new(
            exchange_id,
            AirMessage::Response {
                poll_rx: poll_rx.value(),
                response_tx: predicted_response_tx,
                state: latest_ego_state(),
            },
        ),
    )
    .unwrap_or_else(|error| reset_after_radio_failure(error));
    let response_at = RadioInstant::new(response_at_value)
        .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidTimestamp));
    let mut sending = radio
        .send_raw(
            response.bytes(),
            SendTime::Delayed(response_at),
            &radio_config(),
        )
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(diagnostic(&error)));
    let actual_response_tx = match wait_send(&mut sending, wait).await {
        Ok(timestamp) => timestamp,
        Err(error) => {
            recover_exchange(error, wait, runtime);
            let radio = finish_sending(sending, wait).await;
            return (radio, None);
        }
    };
    radio = finish_sending(sending, wait).await;
    if !scheduled_tx_matches(predicted_response_tx, actual_response_tx.value()) {
        recover_exchange(Diagnostic::DelayedSendTooLate, wait, runtime);
        return (radio, None);
    }
    runtime.health.responses_sent = runtime.health.responses_sent.wrapping_add(1);

    if let Err(error) = prepare_rx(&mut radio, Some(RESPONSE_TIMEOUT_US)).await {
        recover_exchange(error, wait, runtime);
        return (radio, None);
    }
    let mut receiving = radio
        .receive(radio_config())
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(diagnostic(&error)));
    let final_message = wait_receive(&mut receiving, &mut runtime.rx_buf, wait).await;
    radio = finish_receiving(receiving, wait).await;
    let (length, final_rx) = match final_message {
        Ok(final_message) => final_message,
        Err(error) if recoverable_receive_error(error) => {
            recover_exchange(error, wait, runtime);
            return (radio, None);
        }
        Err(error) => reset_after_radio_failure(error),
    };
    let final_frame = match decode_air(&runtime.rx_buf, length, runtime.local) {
        Ok(frame) => frame,
        Err(error) => {
            recover_exchange(error, wait, runtime);
            runtime.health.malformed_air_frames =
                runtime.health.malformed_air_frames.wrapping_add(1);
            return (radio, None);
        }
    };
    let (poll_tx, response_rx, final_tx) = match final_frame.envelope.message {
        AirMessage::Final {
            poll_tx,
            response_rx,
            final_tx,
        } if final_frame.source == peer && final_frame.envelope.exchange_id == exchange_id => {
            runtime.health.finals_received = runtime.health.finals_received.wrapping_add(1);
            (poll_tx, response_rx, final_tx)
        }
        _ => {
            unexpected(runtime);
            return (radio, None);
        }
    };
    let Some(distance) = distance_metres(
        [poll_tx, response_rx, final_tx],
        ResponderTimestamps {
            poll_rx: poll_rx.value(),
            response_tx: predicted_response_tx,
            final_rx: final_rx.value(),
        },
    ) else {
        recover_exchange(Diagnostic::InvalidDistance, wait, runtime);
        return (radio, None);
    };
    let millimetres = millimetres(distance);
    let index = pair_index(runtime.roster(), pair)
        .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidConfiguration));
    let close = classify_range(
        runtime.pair_states[index].is_close(Instant::now().as_micros()),
        millimetres,
    );
    if let Err(error) = prepare_tx(&mut radio).await {
        recover_exchange(error, wait, runtime);
        return (radio, None);
    }
    let report = encode_air(
        runtime.local,
        &mut runtime.mac_sequence,
        Destination::Node(peer),
        AirEnvelope::new(
            exchange_id,
            AirMessage::Report {
                final_rx: final_rx.value(),
                status: if close { REPORT_STATUS_CLOSE } else { 0 },
            },
        ),
    )
    .unwrap_or_else(|error| reset_after_radio_failure(error));
    let report_at_value = delayed_tx_time(final_rx.value(), REPORT_TURNAROUND_US);
    let report_at = RadioInstant::new(report_at_value)
        .unwrap_or_else(|| reset_after_radio_failure(Diagnostic::InvalidTimestamp));
    let mut sending = radio
        .send_raw(
            report.bytes(),
            SendTime::Delayed(report_at),
            &radio_config(),
        )
        .await
        .unwrap_or_else(|error| reset_after_radio_failure(diagnostic(&error)));
    if let Err(error) = wait_send(&mut sending, wait).await {
        recover_exchange(error, wait, runtime);
        let radio = finish_sending(sending, wait).await;
        return (radio, None);
    }
    radio = finish_sending(sending, wait).await;
    runtime.health.reports_sent = runtime.health.reports_sent.wrapping_add(1);
    let mission_event_time = map_event_time(&mut radio, runtime, final_rx.value()).await;
    runtime.spi_errors_since_range = 0;
    (
        radio,
        Some(ExchangeOutcome {
            exchange: CompletedExchange {
                peer,
                exchange_id,
                range_event_time_dtu: final_rx.value(),
                mission_event_time,
                millimetres,
                rssi_cdbm: i16::MIN,
                quality_flags: 0,
                state: peer_state,
            },
            close,
        }),
    )
}

fn encode_air(
    local: NodeAddress,
    mac_sequence: &mut u8,
    destination: Destination,
    envelope: AirEnvelope,
) -> Result<air::EncodedAirFrame, Diagnostic> {
    let sequence = *mac_sequence;
    *mac_sequence = (*mac_sequence).wrapping_add(1);
    air::encode(local, destination, sequence, &envelope).map_err(|_| Diagnostic::MalformedAir)
}

async fn map_event_time<SPI>(
    radio: &mut DW3000<SPI, dw3000_ng::Ready>,
    runtime: &mut Runtime,
    event_dtu: u64,
) -> MissionEventTime
where
    SPI: embedded_hal_async::spi::SpiDevice<u8>,
{
    let local_before_us = Instant::now().as_micros();
    let Ok(radio_now) = radio.sys_time().await else {
        runtime.health.clock_samples_rejected =
            runtime.health.clock_samples_rejected.wrapping_add(1);
        return MissionEventTime::Unavailable;
    };
    let local_after_us = Instant::now().as_micros();
    let local_now_us = local_before_us + local_after_us.saturating_sub(local_before_us) / 2;
    let radio_now_dtu = (u64::from(radio_now) << 8) & TIMESTAMP_MASK;
    let age_us = wrapping_delta(radio_now_dtu, event_dtu).div_ceil(DTU_PER_US);
    let local_event_us = local_now_us.saturating_sub(age_us);
    let bracket_error_us =
        u32::try_from(local_after_us.saturating_sub(local_before_us).div_ceil(2))
            .unwrap_or(u32::MAX);
    let Some(model) = runtime.clock.model() else {
        mark_mapping_unavailable(runtime);
        return MissionEventTime::Unavailable;
    };
    let error_us = model
        .error_at(local_event_us)
        .saturating_add(bracket_error_us);
    let Some(mission_time_us) = model.target_at(local_event_us) else {
        mark_mapping_unavailable(runtime);
        return MissionEventTime::Unavailable;
    };
    if error_us > MAPPING_MAX_ERROR_US {
        mark_mapping_unavailable(runtime);
        return MissionEventTime::Unavailable;
    }
    runtime.mapping_was_unavailable = false;
    MissionEventTime::Mapped {
        mission_time_us,
        generation: model.generation(),
        error_us,
    }
}

fn mark_mapping_unavailable(runtime: &mut Runtime) {
    if !runtime.mapping_was_unavailable {
        runtime.health.time_mapping_unavailable =
            runtime.health.time_mapping_unavailable.wrapping_add(1);
    }
    runtime.mapping_was_unavailable = true;
}

fn decode_air(
    buffer: &[u8],
    received_length: usize,
    local: NodeAddress,
) -> Result<DecodedAirFrame, Diagnostic> {
    let bytes = air_bytes(buffer, received_length)?;
    air::decode(bytes, local).map_err(|_| Diagnostic::MalformedAir)
}

fn air_bytes(buffer: &[u8], received_length: usize) -> Result<&[u8], Diagnostic> {
    let Some(frame_length) = received_length.checked_sub(PHY_FCS_LEN) else {
        return Err(Diagnostic::MalformedAir);
    };
    if frame_length > AIR_FRAME_MAX_NO_FCS || frame_length > buffer.len() {
        return Err(Diagnostic::MalformedAir);
    }
    Ok(&buffer[..frame_length])
}

fn millimetres(distance: f64) -> u32 {
    (distance * 1_000.0 + 0.5).clamp(0.0, u32::MAX as f64) as u32
}

fn unexpected(runtime: &mut Runtime) {
    runtime.health.unexpected_air_frames = runtime.health.unexpected_air_frames.wrapping_add(1);
    publish_error(Diagnostic::UnexpectedAir);
}

fn recover_exchange(diagnostic: Diagnostic, wait: &mut RadioWait<'_>, runtime: &mut Runtime) {
    // A missing peer normally expires the receive window. Pair timeout and
    // peer-stale counters own that expected fleet condition.
    if diagnostic != Diagnostic::RxFrameWaitTimeout {
        publish_error(diagnostic);
    }
    if matches!(
        diagnostic,
        Diagnostic::DelayedSendTooLate | Diagnostic::DelayedSendPowerUpWarning
    ) {
        runtime.health.late_delayed_transmits =
            runtime.health.late_delayed_transmits.wrapping_add(1);
    }
    wait.recovered();
    if diagnostic == Diagnostic::Spi {
        runtime.spi_errors_since_range = runtime.spi_errors_since_range.saturating_add(1);
        if runtime.spi_errors_since_range >= 3 {
            reset_after_radio_failure(diagnostic);
        }
    } else {
        runtime.spi_errors_since_range = 0;
    }
}

fn synchronize_health(wait: &RadioWait<'_>, runtime: HealthCounters) {
    let mut counters = runtime;
    let waits = wait.counters().health_counters();
    counters.irq_wakes = waits.irq_wakes;
    counters.spurious_irq_wakes = waits.spurious_irq_wakes;
    counters.wait_timeouts = waits.wait_timeouts;
    counters.recoveries = waits.recoveries;
    counters.merge_transport(transport_counters());
    update_radio_counters(counters);
}
