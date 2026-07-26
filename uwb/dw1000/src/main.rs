use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use mission10_dw1000::NodeAddress;
use mission10_dw1000::board::{
    DEFAULT_GPIO_CHIP, DEFAULT_IRQ_GPIO, DEFAULT_RESET_GPIO, DEFAULT_RUN_SPI_HZ, DEFAULT_SPI_PATH,
    HardwareConfig, PhyProfile, open_and_initialize,
};
use mission10_dw1000::oracle::run_rx_oracle;
use mission10_dw1000::ranging::{Ranger, RangingConfig, RangingStats};

#[derive(Debug, Parser)]
#[command(about = "Mission 10 DW1000 Linux bring-up and DS-TWR tool")]
struct Cli {
    #[command(flatten)]
    hardware: HardwareArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Args)]
struct HardwareArgs {
    #[arg(long, default_value = DEFAULT_SPI_PATH)]
    spi: PathBuf,
    #[arg(long, default_value = DEFAULT_GPIO_CHIP)]
    gpio_chip: PathBuf,
    #[arg(long, default_value_t = DEFAULT_IRQ_GPIO)]
    irq_gpio: u32,
    #[arg(long, default_value_t = DEFAULT_RESET_GPIO)]
    reset_gpio: u32,
    #[arg(long, default_value_t = DEFAULT_RUN_SPI_HZ)]
    spi_hz: u32,
}

impl From<HardwareArgs> for HardwareConfig {
    fn from(args: HardwareArgs) -> Self {
        Self {
            spi_path: args.spi,
            gpio_chip: args.gpio_chip,
            irq_gpio: args.irq_gpio,
            reset_gpio: args.reset_gpio,
            run_spi_hz: args.spi_hz,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Reset, initialize, and identify the connected DW1000.
    Probe {
        #[arg(long, value_parser = parse_node_address, default_value = "0")]
        address: NodeAddress,
    },
    /// Observe raw receive-pipeline events without running the ranging FSM.
    RxOracle(RxOracleArgs),
    /// Run the native addressed DW1000/DWM3001 DS-TWR exchange.
    Range(RangeArgs),
}

#[derive(Debug, Args)]
struct RxOracleArgs {
    #[arg(long, value_parser = parse_node_address)]
    address: NodeAddress,
    /// Stop after this many seconds.
    #[arg(long, default_value_t = 5.0)]
    duration: f64,
}

#[derive(Debug, Args)]
struct RangeArgs {
    #[arg(long, value_parser = parse_node_address)]
    address: NodeAddress,
    #[arg(long = "peer", value_parser = parse_node_address, required = true)]
    peers: Vec<NodeAddress>,
    #[arg(long, default_value_t = 2_000)]
    reply_delay_us: u32,
    #[arg(long, default_value_t = 3.5)]
    timeout_ms: f64,
    /// Reduce TX gain by 8.5 dB to diagnose very close-range receiver saturation.
    #[arg(long)]
    low_tx_power: bool,
    /// Stop after this many seconds; omit to run until Ctrl-C.
    #[arg(long)]
    duration: Option<f64>,
    #[arg(long)]
    quiet: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let hardware: HardwareConfig = cli.hardware.into();
    match cli.command {
        Command::Probe { address } => probe(&hardware, address),
        Command::RxOracle(args) => rx_oracle(&hardware, args),
        Command::Range(args) => range(&hardware, args),
    }
}

fn probe(hardware: &HardwareConfig, address: NodeAddress) -> Result<()> {
    let run_spi_hz = hardware.run_spi_hz;
    let radio = open_and_initialize(hardware, address, PhyProfile::Operational)?;
    let device_id = radio.device_id();
    let identity = radio.identity();
    println!(
        "DW1000 ready: device_id=0x{device_id:08x} address={address} short=0x{:04x} pan=0x{:04x} spi_hz={}",
        identity.short_address.raw(),
        identity.pan_id.raw(),
        run_spi_hz,
    );
    Ok(())
}

fn rx_oracle(hardware: &HardwareConfig, args: RxOracleArgs) -> Result<()> {
    ensure!(
        args.duration.is_finite() && args.duration > 0.0,
        "duration must be positive"
    );
    let duration = Duration::from_secs_f64(args.duration);
    let mut radio = open_and_initialize(hardware, args.address, PhyProfile::Operational)?;
    println!(
        "DW1000 RX oracle ready: device_id=0x{:08x} address={} duration={:.3}s",
        radio.device_id(),
        args.address,
        args.duration,
    );

    let stop = Arc::new(AtomicBool::new(false));
    let signal_stop = stop.clone();
    ctrlc::set_handler(move || signal_stop.store(true, Ordering::Relaxed))
        .context("install Ctrl-C handler")?;
    let stats = run_rx_oracle(&mut radio, &stop, duration)?;
    println!(
        "RX_ORACLE reads={} arms={} preambles={} sfds={} headers={} frames={} fcs_good={} header_errors={} fcs_errors={} rs_errors={} frame_timeouts={} preamble_timeouts={} sfd_timeouts={} lde_errors={} filter_rejections={} overruns={} pointer_mismatches={} rf_pll_loss={} clock_pll_loss={} preamble_rejection={} final_status=0x{:010x} final_state=0x{:010x}",
        stats.status_reads,
        stats.rx_arms,
        stats.preambles,
        stats.sfds,
        stats.headers,
        stats.frames_ready,
        stats.fcs_good,
        stats.header_errors,
        stats.fcs_errors,
        stats.reed_solomon_errors,
        stats.frame_timeouts,
        stats.preamble_timeouts,
        stats.sfd_timeouts,
        stats.lde_errors,
        stats.frame_filter_rejections,
        stats.overruns,
        stats.buffer_pointer_mismatches,
        stats.rf_pll_loss_observed,
        stats.clock_pll_loss_observed,
        stats.preamble_rejection_observed,
        stats.final_status,
        stats.final_state,
    );
    Ok(())
}

fn range(hardware: &HardwareConfig, args: RangeArgs) -> Result<()> {
    ensure!(
        args.timeout_ms.is_finite() && args.timeout_ms > 0.0,
        "response timeout must be positive"
    );
    ensure!(
        args.duration
            .is_none_or(|duration| duration.is_finite() && duration > 0.0),
        "duration must be positive"
    );
    let config = RangingConfig::new(
        args.address,
        args.peers.clone(),
        args.reply_delay_us,
        Duration::from_secs_f64(args.timeout_ms / 1_000.0),
    )?;
    let duration = args.duration.map(Duration::from_secs_f64);
    let quiet = args.quiet;
    report_runtime_scheduling();
    let profile = if args.low_tx_power {
        PhyProfile::CloseRangeDiagnostic
    } else {
        PhyProfile::Operational
    };
    let stop = Arc::new(AtomicBool::new(false));
    let signal_stop = stop.clone();
    ctrlc::set_handler(move || signal_stop.store(true, Ordering::Relaxed))
        .context("install Ctrl-C handler")?;
    let started = Instant::now();
    let mut stats = RangingStats::default();
    let mut restart_delay = Duration::from_millis(100);
    while !stop.load(Ordering::Relaxed)
        && duration.is_none_or(|duration| started.elapsed() < duration)
    {
        let radio = match open_and_initialize(hardware, args.address, profile) {
            Ok(radio) => radio,
            Err(error) => {
                stats.radio_reinitializations += 1;
                eprintln!(
                    "DW1000 initialization failed; retrying in {:.3}s: {error:#}",
                    restart_delay.as_secs_f64()
                );
                std::thread::sleep(restart_delay);
                restart_delay = (restart_delay * 2).min(Duration::from_secs(2));
                continue;
            }
        };
        let mut ranger = Ranger::new(radio, config.clone());
        let peers: Vec<_> = ranger.peers().iter().map(|peer| peer.get()).collect();
        println!(
            "DW1000 ranger ready: device_id=0x{device_id:08x} address={} peers={:?} profile={profile:?}",
            args.address,
            peers,
            device_id = ranger.device_id(),
        );
        let remaining = duration.map(|duration| duration.saturating_sub(started.elapsed()));
        match ranger.run(&stop, remaining, |measurement| {
            if !quiet {
                println!(
                    "range recv={} src={} metres={:.3} seq={} range_event_time_dtu={} mission_event_time_us={:?} mission_generation={:?} mission_time_error_us={}",
                    measurement.receiver,
                    measurement.source,
                    measurement.distance_metres,
                    measurement.sequence,
                    measurement.range_event_time_dtu,
                    measurement.mission_event_time_us,
                    measurement.mission_generation,
                    measurement.mission_time_error_us,
                );
            }
        }) {
            Ok(run_stats) => {
                stats.merge(run_stats);
                break;
            }
            Err(error) => {
                stats.merge(ranger.stats());
                stats.radio_reinitializations += 1;
                eprintln!(
                    "DW1000 service failed; reinitializing in {:.3}s: {error:#}",
                    restart_delay.as_secs_f64()
                );
                std::thread::sleep(restart_delay);
                restart_delay = (restart_delay * 2).min(Duration::from_secs(2));
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
    println!(
        "STATS elapsed={elapsed:.3}s rate={:.1}Hz reinitializations={} polls_tx={} polls_rx={} responses_tx={} responses_rx={} finals_tx={} finals_rx={} reports_tx={} reports_rx={} completions={} backoffs={} ranges={} timeouts={} tx_completion_timeouts={} rx_completion_timeouts={} rx_errors={} invalid={} wrong_peer={} unexpected_frames={} unexpected_tx_events={} scheduled_tx_misses={} explicit_rx_arms={} receiver_resets={}",
        stats.ranges as f64 / elapsed,
        stats.radio_reinitializations,
        stats.polls_sent,
        stats.polls_received,
        stats.responses_sent,
        stats.responses_received,
        stats.finals_sent,
        stats.finals_received,
        stats.reports_sent,
        stats.reports_received,
        stats.exchange_completions,
        stats.contention_backoffs,
        stats.ranges,
        stats.timeouts,
        stats.tx_completion_timeouts,
        stats.rx_completion_timeouts,
        stats.rx_errors,
        stats.invalid_frames,
        stats.wrong_peer_frames,
        stats.unexpected_frames,
        stats.unexpected_tx_events,
        stats.scheduled_tx_misses,
        stats.explicit_rx_arms,
        stats.receiver_resets,
    );
    Ok(())
}

fn report_runtime_scheduling() {
    let mut parameters = libc::sched_param { sched_priority: 0 };
    // These queries do not change process state.
    let policy = unsafe { libc::sched_getscheduler(0) };
    let parameter_result = unsafe { libc::sched_getparam(0, &mut parameters) };
    let cpu = unsafe { libc::sched_getcpu() };
    let policy_name = match policy {
        libc::SCHED_FIFO => "fifo",
        libc::SCHED_RR => "round-robin",
        libc::SCHED_OTHER => "other",
        _ => "unknown",
    };
    if parameter_result == 0 {
        println!(
            "runtime scheduler={policy_name} priority={} cpu={cpu}",
            parameters.sched_priority
        );
    }
    if policy != libc::SCHED_FIFO {
        eprintln!(
            "warning: direct DW1000 service is not running with SCHED_FIFO; use chrt and taskset for qualification"
        );
    }
}

fn parse_node_address(value: &str) -> Result<NodeAddress, String> {
    let raw = if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u16::from_str_radix(hex, 16)
    } else {
        value.parse()
    }
    .map_err(|_| format!("invalid 16-bit node address {value:?}"))?;
    NodeAddress::new(raw).ok_or_else(|| {
        format!("node address {value:?} is outside aircraft 0..3 and development 0x8000..0x80ff")
    })
}

#[cfg(test)]
mod tests {
    use super::parse_node_address;

    #[test]
    fn cli_address_parser_matches_the_mission_namespace() {
        for raw in 0_u32..=u32::from(u16::MAX) {
            let expected = raw <= 3 || (0x8000..=0x80ff).contains(&raw);
            assert_eq!(
                parse_node_address(&raw.to_string()).is_ok(),
                expected,
                "{raw:#06x}"
            );
        }
        assert_eq!(parse_node_address("0x8000").unwrap().get(), 0x8000);
    }
}
