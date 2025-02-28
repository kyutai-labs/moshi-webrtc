use std::sync::Arc;

use anyhow::Result;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use tokio::sync::{mpsc, Mutex};

use lazy_static::lazy_static;

lazy_static! {
    static ref SDP_CHAN_TX_MUTEX: Arc<Mutex<Option<mpsc::Sender<String>>>> =
        Arc::new(Mutex::new(None));
}

/// decode decodes the input from base64
/// It can optionally unzip the input after decoding
pub fn decode(s: &str) -> Result<String> {
    let b = BASE64_STANDARD.decode(s)?;
    let s = String::from_utf8(b)?;
    Ok(s)
}
