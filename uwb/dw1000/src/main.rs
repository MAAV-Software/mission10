use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand};
use mission10_dw1000::NodeIndex;
use mission10_dw1000::board::{
    DEFAULT_GPIO_CHIP, DEFAULT_IRQ_GPIO, DEFAULT_RESET_GPIO, DEFAULT_RUN_SPI_HZ, DEFAULT_SPI_PATH,
    HardwareConfig, open_and_initialize,
};
use mission10_dw1000::ranging::{Ranger, RangingConfig};

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
        #[arg(long, default_value_t = 0)]
        index: u8,
    },
    /// Run the symmetric DW1000/DWM3001 DS-TWR bench exchange.
    Range(RangeArgs),
}

#[derive(Debug, Args)]
struct RangeArgs {
    #[arg(long)]
    index: u8,
    #[arg(long = "peer", required = true)]
    peers: Vec<u8>,
    #[arg(long, default_value_t = 10.0)]
    poll_period_ms: f64,
    #[arg(long, default_value_t = 2_000)]
    reply_delay_us: u32,
    #[arg(long, default_value_t = 10.0)]
    timeout_ms: f64,
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
        Command::Probe { index } => probe(&hardware, index),
        Command::Range(args) => range(&hardware, args),
    }
}

fn probe(hardware: &HardwareConfig, index: u8) -> Result<()> {
    let index = node_index(index)?;
    let run_spi_hz = hardware.run_spi_hz;
    let radio = open_and_initialize(hardware, index)?;
    let device_id = radio.device_id();
    let identity = radio.identity();
    println!(
        "DW1000 ready: device_id=0x{device_id:08x} index={index} short=0x{:04x} pan=0x{:04x} spi_hz={}",
        identity.short_address.raw(),
        identity.pan_id.raw(),
        run_spi_hz,
    );
    Ok(())
}

fn range(hardware: &HardwareConfig, args: RangeArgs) -> Result<()> {
    ensure!(
        args.poll_period_ms.is_finite() && args.poll_period_ms > 0.0,
        "poll period must be positive"
    );
    ensure!(
        args.timeout_ms.is_finite() && args.timeout_ms > 0.0,
        "response timeout must be positive"
    );
    ensure!(
        args.duration
            .is_none_or(|duration| duration.is_finite() && duration > 0.0),
        "duration must be positive"
    );
    let index = node_index(args.index)?;
    let peers = args
        .peers
        .iter()
        .copied()
        .map(node_index)
        .collect::<Result<Vec<_>>>()?;
    let config = RangingConfig::new(
        index,
        peers,
        Duration::from_secs_f64(args.poll_period_ms / 1_000.0),
        args.reply_delay_us,
        Duration::from_secs_f64(args.timeout_ms / 1_000.0),
    )?;
    let duration = args.duration.map(Duration::from_secs_f64);
    let quiet = args.quiet;
    let radio = open_and_initialize(hardware, index)?;
    let mut ranger = Ranger::new(radio, config);
    let peers: Vec<_> = ranger.peers().iter().map(|peer| peer.get()).collect();
    println!(
        "DW1000 ranger ready: device_id=0x{device_id:08x} index={} peers={:?}",
        index,
        peers,
        device_id = ranger.device_id(),
    );

    let stop = Arc::new(AtomicBool::new(false));
    let signal_stop = stop.clone();
    ctrlc::set_handler(move || signal_stop.store(true, Ordering::Relaxed))
        .context("install Ctrl-C handler")?;
    let started = Instant::now();
    let stats = ranger.run(&stop, duration, |measurement| {
        if !quiet {
            println!(
                "range recv={} src={} metres={:.3} seq={} completed_at_dtu={}",
                measurement.receiver,
                measurement.source,
                measurement.distance_metres,
                measurement.sequence,
                measurement.completed_at_dtu
            );
        }
    })?;
    let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
    println!(
        "STATS elapsed={elapsed:.3}s rate={:.1}Hz polls={} ranges={} timeouts={} rx_errors={} invalid={} wrong_peer={} unexpected_frames={} unexpected_tx_events={} scheduled_tx_misses={}",
        stats.ranges as f64 / elapsed,
        stats.polls_sent,
        stats.ranges,
        stats.timeouts,
        stats.rx_errors,
        stats.invalid_frames,
        stats.wrong_peer_frames,
        stats.unexpected_frames,
        stats.unexpected_tx_events,
        stats.scheduled_tx_misses,
    );
    Ok(())
}

fn node_index(value: u8) -> Result<NodeIndex> {
    let Some(index) = NodeIndex::new(value) else {
        bail!("node index {value} is outside the bench address range 0..=15")
    };
    Ok(index)
}
