// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::utils::OPUS_ENCODER_FRAME_SIZE;
pub(crate) const SAMPLE_RATE: u32 = 48_000;
pub(crate) const CHANNELS: u16 = 1;

use std::str::FromStr;
use std::sync::Arc;

use anyhow::Result;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp::packetizer::{Depacketizer, Payloader};

type Receiver = tokio::sync::broadcast::Receiver<Arc<webrtc::media::Sample>>;

pub(crate) async fn run(args: crate::DemoArgs) -> Result<()> {
    let pcm_rx = {
        let (pcm, sample_rate) = kaudio::pcm_decode(&args.audio)?;
        let pcm = kaudio::resample(
            &pcm[(sample_rate as usize * 4)..],
            sample_rate as usize,
            SAMPLE_RATE as usize,
        )?;
        let pcm =
            if CHANNELS == 2 { (0..pcm.len() * 2).map(|i| pcm[i / 2]).collect() } else { pcm };
        send_loop(pcm)?
    };

    let app = axum::Router::new()
        .route("/api/connect", axum::routing::post(connect))
        .route("/api/broadcast", axum::routing::post(broadcast))
        .fallback_service(
            tower_http::services::ServeDir::new(".").append_index_html_on_directories(true),
        )
        .layer(tower::ServiceBuilder::new().layer(tower_http::trace::TraceLayer::new_for_http()))
        .with_state(Arc::new(pcm_rx));
    let sock_addr = std::net::SocketAddr::from((
        std::net::IpAddr::from_str(args.addr.as_str())
            .unwrap_or(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
        args.port,
    ));
    tracing::info!("listening on http://{}", sock_addr);
    let listener = tokio::net::TcpListener::bind(sock_addr).await?;
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .await?;
    Ok(())
}

fn send_loop(pcm: Vec<f32>) -> Result<Receiver> {
    use webrtc::rtp::codecs::opus::OpusPayloader;
    let channels = match CHANNELS {
        1 => opus::Channels::Mono,
        2 => opus::Channels::Stereo,
        c => anyhow::bail!("unsupported channels {c}"),
    };
    let mut encoder = opus::Encoder::new(SAMPLE_RATE, channels, opus::Application::Voip)?;

    let (tx, rx) = tokio::sync::broadcast::channel(10);
    tokio::task::spawn(async move {
        let mut out_pcm_index = 0;
        let mut out_pcm_buf = vec![0u8; SAMPLE_RATE as usize];
        let mut start_time = tokio::time::Instant::now();
        loop {
            let (start_i, end_i) =
                (out_pcm_index, out_pcm_index + OPUS_ENCODER_FRAME_SIZE * CHANNELS as usize);
            if end_i <= pcm.len() {
                let size = encoder.encode_float(&pcm[start_i..end_i], &mut out_pcm_buf)?;
                let duration = (end_i - start_i) as f32 / SAMPLE_RATE as f32 / CHANNELS as f32;
                let bytes = hyper::body::Bytes::copy_from_slice(&out_pcm_buf[..size]);
                let play_time = start_time
                    + tokio::time::Duration::from_secs_f32(
                        end_i as f32 / SAMPLE_RATE as f32 / CHANNELS as f32,
                    );
                tokio::time::sleep_until(play_time).await;
                for payload in OpusPayloader.payload(1500, &bytes)? {
                    let sample = webrtc::media::Sample {
                        data: payload,
                        duration: std::time::Duration::from_secs_f32(duration),
                        ..Default::default()
                    };
                    // Do not fail on error as this happens when there is no receiver.
                    let _ = tx.send(Arc::new(sample));
                }
                out_pcm_index += OPUS_ENCODER_FRAME_SIZE * CHANNELS as usize
            } else {
                start_time = tokio::time::Instant::now();
                out_pcm_index = 0;
            }
        }
        #[allow(unreachable_code)]
        Ok::<_, anyhow::Error>(())
    });
    Ok(rx)
}

async fn connect(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<Arc<Receiver>>,
    axum::extract::Json(req): axum::extract::Json<RTCSessionDescription>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    match connect_(state.resubscribe(), req).await {
        Err(err) => {
            tracing::error!("{err:?}");
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("{err}")).into_response()
        }
        Ok(descr) => axum::Json(descr).into_response(),
    }
}

async fn broadcast(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<Arc<Receiver>>,
    axum::extract::Json(req): axum::extract::Json<RTCSessionDescription>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    match broadcast_(state.resubscribe(), req).await {
        Err(err) => {
            tracing::error!("{err:?}");
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("{err}")).into_response()
        }
        Ok(descr) => axum::Json(descr).into_response(),
    }
}

async fn connect_(
    mut pcm_rx: Receiver,
    client_offer: RTCSessionDescription,
) -> Result<RTCSessionDescription> {
    let peer_connection = crate::utils::peer_connection(SAMPLE_RATE, CHANNELS).await?;
    let output_track = crate::utils::output_track(&peer_connection, SAMPLE_RATE, CHANNELS).await?;
    // Set the remote SessionDescription
    peer_connection.set_remote_description(client_offer).await?;

    // Set a handler for when a new remote track starts, this handler copies inbound RTP packets,
    // replaces the SSRC and sends them back
    peer_connection.on_track(Box::new(move |track, _, _| {
        tracing::info!("connection established {track:?}");

        tokio::spawn(async move {
            if let Err(err) = handle_rtp(track).await {
                tracing::error!("{err:?}")
            }
        });
        Box::pin(async {})
    }));

    let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);
    let (start_tx, mut start_rx) = tokio::sync::mpsc::channel::<()>(1);

    // Set the handler for Peer connection state
    // This will notify you when the peer has connected/disconnected
    peer_connection.on_peer_connection_state_change(Box::new(move |s: RTCPeerConnectionState| {
        tracing::info!("Peer Connection State has changed: {s}");

        if s == RTCPeerConnectionState::Connected {
            let _ = start_tx.try_send(());
        }
        if s == RTCPeerConnectionState::Failed {
            // Wait until PeerConnection has had no network activity for 30 seconds or another failure. It may be reconnected using an ICE Restart.
            // Use webrtc.PeerConnectionStateDisconnected if you are interested in detecting faster timeout.
            // Note that the PeerConnection may come back from PeerConnectionStateDisconnected.
            tracing::error!("Peer Connection has gone to failed exiting");
            let _ = done_tx.try_send(());
        }

        Box::pin(async {})
    }));

    // Create an answer
    let answer = peer_connection.create_answer(None).await?;

    // Create channel that is blocked until ICE Gathering is complete
    let mut gather_complete = peer_connection.gathering_complete_promise().await;

    // Sets the LocalDescription, and starts our UDP listeners
    peer_connection.set_local_description(answer).await?;

    // Block until ICE Gathering is complete, disabling trickle ICE
    // we do this because we only can exchange one signaling message
    // in a production application you should exchange ICE Candidates via OnICECandidate
    let _ = gather_complete.recv().await;

    // Output the answer in base64 so we can paste it in browser
    let local_desc = match peer_connection.local_description().await {
        Some(desc) => desc,
        None => anyhow::bail!("generate local_description failed!"),
    };
    tracing::info!("desc {local_desc:?}");

    tokio::task::spawn(async move {
        tracing::info!("waiting for a connection");
        let _ = start_rx.recv().await;
        tracing::info!("starting send loop");
        loop {
            let sample = match pcm_rx.recv().await {
                Ok(sample) => sample,
                Err(err) => {
                    tracing::error!(?err, "no sample");
                    continue;
                }
            };
            if let Err(err) = output_track.write_sample(&sample).await {
                tracing::error!("output track write_rtp got error: {err}");
                break;
            }
            if done_rx.try_recv().is_ok() {
                tracing::info!("exiting the send loop");
                break;
            }
        }
        tracing::info!("closing peer connection");
        peer_connection.close().await?;
        Ok::<_, anyhow::Error>(())
    });

    Ok(local_desc)
}

async fn handle_rtp(track: Arc<webrtc::track::track_remote::TrackRemote>) -> Result<()> {
    use webrtc::rtp::codecs::opus::OpusPacket;
    let channels = match CHANNELS {
        1 => opus::Channels::Mono,
        2 => opus::Channels::Stereo,
        c => anyhow::bail!("unsupported channels {c}"),
    };
    let mut decoder = opus::Decoder::new(SAMPLE_RATE, channels)?;

    tracing::info!(
        "Track has started, of type {}: {}",
        track.payload_type(),
        track.codec().capability.mime_type
    );
    let mut pcm_buf = vec![0i16; SAMPLE_RATE as usize];
    let mut all_pcm: Vec<i16> = Vec::with_capacity(10_000_000);
    let mut payloads = vec![];
    // Read RTP packets being sent to webrtc-rs
    while let Ok((rtp, _)) = track.read_rtp().await {
        if !rtp.payload.is_empty() {
            let payload = OpusPacket.depacketize(&rtp.payload)?;
            payloads.push(payload.clone());
            let size_read = decoder.decode(&payload, &mut pcm_buf, /* fec? */ false)?;
            if CHANNELS == 2 {
                for i in 0..size_read {
                    let v = (pcm_buf[2 * i] as f32 + pcm_buf[2 * i + 1] as f32) / 2.0;
                    all_pcm.push(v as i16)
                }
            } else {
                all_pcm.extend_from_slice(&pcm_buf[..size_read]);
            }
        }
        if all_pcm.len() > SAMPLE_RATE as usize * 20 {
            let mut file = std::fs::File::create("thisisatest.wav")?;
            kaudio::wav::write_pcm_as_wav(&mut file, &all_pcm, SAMPLE_RATE, 1)?;
            std::fs::write("thisisatest.opus", payloads.concat())?;
            break;
        }
    }
    tracing::info!(
        "on_track finished, of type {}: {}",
        track.payload_type(),
        track.codec().capability.mime_type
    );
    Ok(())
}

async fn broadcast_(
    mut pcm_rx: Receiver,
    client_offer: RTCSessionDescription,
) -> Result<RTCSessionDescription> {
    let peer_connection = crate::utils::peer_connection(SAMPLE_RATE, CHANNELS).await?;
    let output_track = crate::utils::output_track(&peer_connection, SAMPLE_RATE, CHANNELS).await?;
    peer_connection.set_remote_description(client_offer).await?;

    let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);
    let (start_tx, mut start_rx) = tokio::sync::mpsc::channel::<()>(1);
    peer_connection.on_peer_connection_state_change(Box::new(move |s: RTCPeerConnectionState| {
        tracing::info!("Peer Connection State has changed: {s}");
        if s == RTCPeerConnectionState::Connected {
            let _ = start_tx.try_send(());
        }
        if s == RTCPeerConnectionState::Failed {
            tracing::error!("Peer Connection has gone to failed exiting");
            let _ = done_tx.try_send(());
        }
        Box::pin(async {})
    }));

    let answer = peer_connection.create_answer(None).await?;
    let mut gather_complete = peer_connection.gathering_complete_promise().await;
    peer_connection.set_local_description(answer).await?;
    let _ = gather_complete.recv().await;
    let local_desc = match peer_connection.local_description().await {
        Some(desc) => desc,
        None => anyhow::bail!("generate local_description failed!"),
    };
    tokio::task::spawn(async move {
        tracing::info!("waiting for a connection");
        let _ = start_rx.recv().await;
        tracing::info!("starting send loop");
        loop {
            let sample = match pcm_rx.recv().await {
                Ok(sample) => sample,
                Err(err) => {
                    tracing::error!(?err, "no sample");
                    continue;
                }
            };
            if let Err(err) = output_track.write_sample(&sample).await {
                tracing::error!("output track write_rtp got error: {err}");
                break;
            }
            if done_rx.try_recv().is_ok() {
                tracing::info!("exiting the send loop");
                break;
            }
        }
        tracing::info!("closing peer connection");
        peer_connection.close().await?;
        Ok::<_, anyhow::Error>(())
    });
    Ok(local_desc)
}
