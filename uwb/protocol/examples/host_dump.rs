use std::fs::File;
use std::io::{self, Read};

use mission10_uwb_protocol::{HOST_FRAME_MAX_SIZE, HOST_RAW_MAX_SIZE, decode_host_frame};

fn main() -> io::Result<()> {
    // Put a serial/CDC device in raw mode before using this tiny generic reader,
    // for example: `stty -F /dev/ttyACM0 raw -echo`.
    let mut input: Box<dyn Read> = match std::env::args_os().nth(1) {
        Some(path) => Box::new(File::open(path)?),
        None => Box::new(io::stdin()),
    };
    let mut frame = Vec::with_capacity(HOST_FRAME_MAX_SIZE);
    let mut raw = [0_u8; HOST_RAW_MAX_SIZE];
    let mut chunk = [0_u8; 64];

    loop {
        let count = input.read(&mut chunk)?;
        if count == 0 {
            return Ok(());
        }
        for &byte in &chunk[..count] {
            if frame.len() == HOST_FRAME_MAX_SIZE {
                eprintln!("discarding oversized host frame");
                frame.clear();
            }
            frame.push(byte);
            if byte == 0 {
                match decode_host_frame(&frame, &mut raw) {
                    Ok(envelope) => println!("{envelope:?}"),
                    Err(error) => eprintln!("discarding invalid host frame: {error:?}"),
                }
                frame.clear();
            }
        }
    }
}
