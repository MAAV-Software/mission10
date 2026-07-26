use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use dw1000_rs::registers::status;
use dw1000_rs::{RxOptions, SysStatus};

use crate::board::RadioHardware;

const STATUS_POLL_PERIOD: Duration = Duration::from_millis(1);
const STICKY_DIAGNOSTIC_BITS: u64 =
    status::RF_PLL_LOSS.0 | status::CLOCK_PLL_LOSS.0 | status::RX_PREAMBLE_REJECTION.0;

#[derive(Debug, Default)]
pub struct RxOracleStats {
    pub status_reads: u64,
    pub rx_arms: u64,
    pub preambles: u64,
    pub sfds: u64,
    pub headers: u64,
    pub frames_ready: u64,
    pub fcs_good: u64,
    pub header_errors: u64,
    pub fcs_errors: u64,
    pub reed_solomon_errors: u64,
    pub frame_timeouts: u64,
    pub preamble_timeouts: u64,
    pub sfd_timeouts: u64,
    pub lde_errors: u64,
    pub frame_filter_rejections: u64,
    pub overruns: u64,
    pub buffer_pointer_mismatches: u64,
    pub rf_pll_loss_observed: bool,
    pub clock_pll_loss_observed: bool,
    pub preamble_rejection_observed: bool,
    pub final_status: u64,
    pub final_state: u64,
}

pub fn run_rx_oracle(
    hardware: &mut RadioHardware,
    stop: &AtomicBool,
    duration: Duration,
) -> Result<RxOracleStats> {
    let mut stats = RxOracleStats::default();
    arm_receiver(hardware, &mut stats)?;

    let started = Instant::now();
    let mut seen = SysStatus::EMPTY;
    let mut pointer_mismatch_seen = false;
    while !stop.load(Ordering::Relaxed) && started.elapsed() < duration {
        let current = hardware
            .radio
            .read_sys_status()
            .map_err(|error| anyhow::anyhow!("read DW1000 oracle status: {error:?}"))?;
        stats.status_reads += 1;
        let terminal = observe_status(current, &mut seen, &mut stats);

        let host_pointer = current.0 & (1 << 30) != 0;
        let radio_pointer = current.0 & (1 << 31) != 0;
        if host_pointer != radio_pointer && !pointer_mismatch_seen {
            stats.buffer_pointer_mismatches += 1;
            pointer_mismatch_seen = true;
        }

        if terminal {
            arm_receiver(hardware, &mut stats)?;
            seen = SysStatus(seen.0 & STICKY_DIAGNOSTIC_BITS);
            pointer_mismatch_seen = false;
        } else {
            std::thread::sleep(STATUS_POLL_PERIOD);
        }
    }

    let final_state = hardware
        .radio
        .read_debug_state()
        .map_err(|error| anyhow::anyhow!("read final DW1000 oracle state: {error:?}"))?;
    stats.final_status = final_state.sys_status.0;
    stats.final_state = final_state.sys_state;
    Ok(stats)
}

fn arm_receiver(hardware: &mut RadioHardware, stats: &mut RxOracleStats) -> Result<()> {
    hardware
        .radio
        .start_receive(RxOptions {
            delayed_time: None,
            permanent: false,
        })
        .map_err(|error| anyhow::anyhow!("arm DW1000 oracle receiver: {error:?}"))?;
    stats.rx_arms += 1;
    Ok(())
}

fn observe_status(current: SysStatus, seen: &mut SysStatus, stats: &mut RxOracleStats) -> bool {
    let new = current.0 & !seen.0;
    stats.preambles += u64::from(new & status::RX_PREAMBLE_DETECTED.0 != 0);
    stats.sfds += u64::from(new & status::RX_SFD_DETECTED.0 != 0);
    stats.headers += u64::from(new & status::RX_HEADER_DETECTED.0 != 0);
    stats.frames_ready += u64::from(new & status::RX_FRAME_READY.0 != 0);
    stats.fcs_good += u64::from(new & status::RX_FRAME_GOOD.0 != 0);
    stats.header_errors += u64::from(new & status::RX_HEADER_ERROR.0 != 0);
    stats.fcs_errors += u64::from(new & status::RX_FRAME_CHECK_ERROR.0 != 0);
    stats.reed_solomon_errors += u64::from(new & status::RX_REED_SOLOMON_ERROR.0 != 0);
    stats.frame_timeouts += u64::from(new & status::RX_TIMEOUT.0 != 0);
    stats.preamble_timeouts += u64::from(new & status::RX_PREAMBLE_TIMEOUT.0 != 0);
    stats.sfd_timeouts += u64::from(new & status::RX_SFD_TIMEOUT.0 != 0);
    stats.lde_errors += u64::from(new & status::LDE_ERROR.0 != 0);
    stats.frame_filter_rejections += u64::from(new & status::FRAME_FILTER_REJECTION.0 != 0);
    stats.overruns += u64::from(new & status::RX_OVERRUN.0 != 0);
    stats.rf_pll_loss_observed |= new & status::RF_PLL_LOSS.0 != 0;
    stats.clock_pll_loss_observed |= new & status::CLOCK_PLL_LOSS.0 != 0;
    stats.preamble_rejection_observed |= new & status::RX_PREAMBLE_REJECTION.0 != 0;
    *seen |= current;
    current.0 & status::RX_TERMINAL_EVENTS.0 != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_each_latched_stage_once_per_receive_arm() {
        let mut stats = RxOracleStats::default();
        let mut seen = SysStatus::EMPTY;
        let good = status::RX_PREAMBLE_DETECTED
            | status::RX_SFD_DETECTED
            | status::RX_HEADER_DETECTED
            | status::RX_FRAME_READY
            | status::RX_FRAME_GOOD;

        assert!(observe_status(good, &mut seen, &mut stats));
        assert!(observe_status(good, &mut seen, &mut stats));
        assert_eq!(stats.preambles, 1);
        assert_eq!(stats.sfds, 1);
        assert_eq!(stats.headers, 1);
        assert_eq!(stats.frames_ready, 1);
        assert_eq!(stats.fcs_good, 1);
    }

    #[test]
    fn reports_nonterminal_pll_and_preamble_rejection_state() {
        let mut stats = RxOracleStats::default();
        let mut seen = SysStatus::EMPTY;
        let diagnostic =
            status::RF_PLL_LOSS | status::CLOCK_PLL_LOSS | status::RX_PREAMBLE_REJECTION;

        assert!(!observe_status(diagnostic, &mut seen, &mut stats));
        assert!(stats.rf_pll_loss_observed);
        assert!(stats.clock_pll_loss_observed);
        assert!(stats.preamble_rejection_observed);
    }
}
