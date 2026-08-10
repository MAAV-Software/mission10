use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use mission10_uwb_protocol::host::{
    CompletedExchange, HOST_FRAME_MAX_SIZE, HOST_RAW_MAX_SIZE, HostToRadio, MissionEventTime,
    RadioToHost, RadioToHostEnvelope, decode_host_to_radio, encode_radio_to_host,
};

use crate::ranging::{RadioCommand, RadioEvent};

const QUEUE_DEPTH: usize = 64;

pub struct HostService {
    pub commands: Receiver<RadioCommand>,
    pub events: SyncSender<RadioEvent>,
}

pub fn start(path: &Path) -> Result<HostService> {
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("remove stale socket {}", path.display()))?;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create socket directory {}", parent.display()))?;
    }
    let listener =
        UnixListener::bind(path).with_context(|| format!("bind host socket {}", path.display()))?;
    let (command_tx, command_rx) = sync_channel(QUEUE_DEPTH);
    let (event_tx, event_rx) = sync_channel(QUEUE_DEPTH);
    thread::Builder::new()
        .name("uwb-host".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => run_connection(stream, &command_tx, &event_rx),
                    Err(error) => eprintln!("uwb-host: accept failed: {error}"),
                }
            }
        })
        .context("spawn UWB host thread")?;
    Ok(HostService {
        commands: command_rx,
        events: event_tx,
    })
}

fn run_connection(
    mut stream: UnixStream,
    commands: &SyncSender<RadioCommand>,
    events: &Receiver<RadioEvent>,
) {
    if let Err(error) = stream.set_read_timeout(Some(Duration::from_millis(10))) {
        eprintln!("uwb-host: set read timeout: {error}");
        return;
    }
    // A client that stops reading must not wedge this thread: it also owns
    // command forwarding, so an unbounded write_all would stall SetEgoState
    // until every peer marks this node HOST_STALE.
    if let Err(error) = stream.set_write_timeout(Some(Duration::from_millis(200))) {
        eprintln!("uwb-host: set write timeout: {error}");
        return;
    }
    let mut input = [0_u8; 256];
    let mut frame = Vec::with_capacity(HOST_FRAME_MAX_SIZE);
    let mut raw = [0_u8; HOST_RAW_MAX_SIZE];
    let mut output = [0_u8; HOST_FRAME_MAX_SIZE];
    let mut sequence = 0_u32;
    loop {
        match stream.read(&mut input) {
            Ok(0) => return,
            Ok(length) => {
                for byte in &input[..length] {
                    frame.push(*byte);
                    if *byte == 0 {
                        let envelope = match decode_host_to_radio(&frame, &mut raw) {
                            Ok(envelope) => envelope,
                            Err(error) => {
                                eprintln!("uwb-host: bad client frame, disconnecting: {error:?}");
                                return;
                            }
                        };
                        frame.clear();
                        let command = match envelope.message {
                            HostToRadio::SetEgoState { state } => RadioCommand::SetEgoState(state),
                            HostToRadio::BroadcastFleetMode { mode } => {
                                RadioCommand::BroadcastFleetMode(mode)
                            }
                            _ => {
                                eprintln!("uwb-host: unsupported command, disconnecting");
                                return;
                            }
                        };
                        match commands.try_send(command) {
                            Ok(()) => {}
                            Err(TrySendError::Full(_)) => {
                                eprintln!("uwb-host: command queue full, dropping command");
                            }
                            Err(TrySendError::Disconnected(_)) => return,
                        }
                    }
                }
            }
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => return,
        }
        loop {
            let event = match events.try_recv() {
                Ok(event) => event,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            };
            let message = radio_message(event);
            let envelope = RadioToHostEnvelope::new(sequence, message);
            sequence = sequence.wrapping_add(1);
            let length = match encode_radio_to_host(&envelope, &mut output) {
                Ok(length) => length,
                Err(error) => {
                    eprintln!("uwb-host: encode event failed: {error:?}");
                    continue;
                }
            };
            // A write timeout leaves the client stream mid-frame, so any
            // write error ends the connection rather than corrupt framing.
            if stream.write_all(&output[..length]).is_err() {
                return;
            }
        }
    }
}

fn radio_message(event: RadioEvent) -> RadioToHost {
    match event {
        RadioEvent::FleetMode { source, mode } => RadioToHost::FleetModeReceived { source, mode },
        RadioEvent::Range(measurement) => RadioToHost::CompletedExchange {
            exchange: CompletedExchange {
                peer: measurement.source,
                exchange_id: measurement.exchange_id,
                range_event_time_dtu: measurement.range_event_time_dtu,
                mission_event_time: match (
                    measurement.mission_event_time_us,
                    measurement.mission_generation,
                ) {
                    (Some(mission_time_us), Some(generation)) => MissionEventTime::Mapped {
                        mission_time_us,
                        generation,
                        error_us: measurement.mission_time_error_us,
                    },
                    _ => MissionEventTime::Unavailable,
                },
                millimetres: (measurement.distance_metres * 1_000.0).round() as u32,
                rssi_cdbm: i16::MIN,
                quality_flags: 0,
                state: measurement.peer_state,
            },
        },
    }
}
