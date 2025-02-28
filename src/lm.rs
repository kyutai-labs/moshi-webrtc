// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;

use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp::packetizer::{Depacketizer, Payloader};

enum BroadcastEvent {
    Sample(webrtc::media::Sample),
    Text(String),
}

type Sender = tokio::sync::broadcast::Sender<Arc<BroadcastEvent>>;
type Receiver = tokio::sync::broadcast::Receiver<Arc<BroadcastEvent>>;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub static_dir: String,
    pub lm_model_file: String,
    pub text_tokenizer_file: String,
    pub audio_tokenizer_file: String,
    pub model: moshi::lm::Config,
    pub gen: moshi::lm_generate_multistream::Config,
}

impl Config {
    fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        use crate::utils::resolve_or_download as rod;
        let cfg = std::fs::read_to_string(p.as_ref())?;
        let mut cfg: Self = toml::from_str(&cfg)?;
        cfg.static_dir = rod(&cfg.static_dir)?;
        cfg.lm_model_file = rod(&cfg.lm_model_file)?;
        cfg.text_tokenizer_file = rod(&cfg.text_tokenizer_file)?;
        cfg.audio_tokenizer_file = rod(&cfg.audio_tokenizer_file)?;
        Ok(cfg)
    }
}

#[allow(unused)]
enum LogEvent {
    Text(String),
    TextToken(u32),
    AudioTokens(Vec<u32>),
    Pcm(Vec<f32>),
}

struct TextDecoder {
    gen_config: moshi::lm_generate_multistream::Config,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
}

impl TextDecoder {
    fn text(&self, prev_text_token: u32, text_token: u32) -> Option<String> {
        let config = &self.gen_config;
        if text_token != config.text_start_token
            && text_token != config.text_pad_token
            && text_token != config.text_eop_token
        {
            if prev_text_token == config.text_start_token {
                self.text_tokenizer.decode_piece_ids(&[text_token]).ok()
            } else {
                let prev_ids = self.text_tokenizer.decode_piece_ids(&[prev_text_token]).ok();
                let ids = self.text_tokenizer.decode_piece_ids(&[prev_text_token, text_token]).ok();
                prev_ids.and_then(|prev_ids| {
                    ids.map(|ids| {
                        if ids.len() > prev_ids.len() {
                            ids[prev_ids.len()..].to_string()
                        } else {
                            String::new()
                        }
                    })
                })
            }
        } else {
            None
        }
    }
}

struct Lm {
    dev: Device,
    gen_config: moshi::lm_generate_multistream::Config,
    lm: moshi::lm::LmModel,
    audio_tokenizer: moshi::mimi::Mimi,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
}

impl Lm {
    fn new(lm: &Config, dev: &Device) -> Result<Self> {
        let dtype = dev.bf16_default_to_f32();
        let model_config = &lm.model;
        let gen_config = lm.gen.clone();
        let audio_tokenizer = moshi::mimi::load(&lm.audio_tokenizer_file, Some(8), dev)?;
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&lm.text_tokenizer_file)
            .with_context(|| lm.text_tokenizer_file.clone())?;
        let text_tokenizer = Arc::new(text_tokenizer);
        let vb_lm =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&lm.lm_model_file], dtype, dev)? };
        let lm = moshi::lm::LmModel::new(
            model_config,
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
        )?;
        Ok(Self { audio_tokenizer, text_tokenizer, lm, gen_config, dev: dev.clone() })
    }

    fn run(
        &self,
        pcm_in: std::sync::mpsc::Receiver<Vec<f32>>,
        tx: std::sync::mpsc::Sender<LogEvent>,
    ) -> Result<()> {
        let dev = self.dev.clone();
        let mut audio_tokenizer = self.audio_tokenizer.clone();
        audio_tokenizer.reset_state();
        let text_decoder = TextDecoder {
            gen_config: self.gen_config.clone(),
            text_tokenizer: self.text_tokenizer.clone(),
        };
        let text_lp = LogitsProcessor::from_sampling(
            299792458,
            candle_transformers::generation::Sampling::TopK { k: 25, temperature: 0.8 },
        );
        let audio_lp = LogitsProcessor::from_sampling(
            299792458,
            candle_transformers::generation::Sampling::TopK { k: 250, temperature: 0.8 },
        );
        let conditions = match self.lm.condition_provider() {
            None => None,
            Some(cp) => {
                let conditions = cp.condition_lut("description", "very_good")?;
                tracing::info!(?conditions, "generated conditions");
                Some(conditions)
            }
        };
        let mut state = moshi::lm_generate_multistream::State::new(
            self.lm.clone(),
            /* max_steps = */ 4096,
            audio_lp,
            text_lp,
            None,
            None,
            None,
            self.gen_config.clone(),
        );
        let mut prev_text_token = state.config().text_start_token;

        while let Ok(pcm) = pcm_in.recv() {
            let pcm = Tensor::new(pcm, &dev)?.reshape((1, 1, ()))?;
            let audio_tokens = audio_tokenizer.encode_step(&pcm.into())?;
            let audio_tokens = match audio_tokens.as_option() {
                None => continue,
                Some(audio_tokens) => audio_tokens,
            };
            let (_one, _codebooks, steps) = audio_tokens.dims3()?;

            for step in 0..steps {
                let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                let text_token =
                    state.step_(Some(prev_text_token), &codes, None, None, conditions.as_ref())?;

                if let Some(text) = text_decoder.text(prev_text_token, text_token) {
                    tx.send(LogEvent::Text(text))?
                }
                tx.send(LogEvent::TextToken(text_token))?;
                tracing::info!(text_token, "sampled text token");
                if let Some(audio_tokens) = state.last_audio_tokens() {
                    let audio_tokens_t = {
                        let cb = state.config().generated_audio_codebooks;
                        Tensor::from_slice(&audio_tokens[..cb], (1, cb, 1), &dev)?
                    };
                    tx.send(LogEvent::AudioTokens(audio_tokens))?;
                    let pcm = audio_tokenizer.decode_step(&audio_tokens_t.into())?;
                    if let Some(pcm) = pcm.as_option() {
                        let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
                        tx.send(LogEvent::Pcm(pcm))?;
                    }
                }
                prev_text_token = text_token
            }
        }

        Ok(())
    }

    fn warmup(&self) -> Result<()> {
        let mut lm_model = self.lm.clone();
        let in_codebooks = self.gen_config.input_audio_codebooks;
        let (_v, ys) = lm_model.forward(None, vec![None; in_codebooks])?;
        let mut lp = candle_transformers::generation::LogitsProcessor::new(123, None, None);
        let _ = lm_model.depformer_sample(&ys, None, &[], &mut lp)?;

        let config = self.audio_tokenizer.config();
        let frame_length = (config.sample_rate / config.frame_rate).ceil() as usize;
        let fake_pcm = candle::Tensor::zeros((1, 1, frame_length), candle::DType::F32, &self.dev)?;
        let mut audio_tokenizer = self.audio_tokenizer.clone();
        let codes = audio_tokenizer.encode_step(&fake_pcm.into())?;
        let ys = audio_tokenizer.decode_step(&codes)?;
        if ys.as_option().is_none() {
            anyhow::bail!("Expected Mimi to output some stuff, but nothing came out.");
        }
        self.dev.synchronize()?;
        Ok(())
    }
}

struct State {
    lm: Lm,
    out_tx: Sender,
    out_rx: Receiver,
}

pub(crate) async fn run(args: crate::LmArgs) -> Result<()> {
    let cfg = Config::load(&args.config)?;
    let dev = crate::utils::device(args.cpu)?;
    let lm = Lm::new(&cfg, &dev)?;

    tracing::info!("warming up the model");
    lm.warmup()?;
    tracing::info!("model is ready to roll!");
    let (out_tx, out_rx) = tokio::sync::broadcast::channel(10);
    let state = Arc::new(State { lm, out_tx, out_rx });

    let app = axum::Router::new()
        .route("/api/connect", axum::routing::post(connect))
        .route("/api/broadcast", axum::routing::post(broadcast))
        .fallback_service(
            tower_http::services::ServeDir::new(&cfg.static_dir)
                .append_index_html_on_directories(true),
        )
        .layer(tower::ServiceBuilder::new().layer(tower_http::trace::TraceLayer::new_for_http()))
        .with_state(state);
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

async fn broadcast(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    axum::extract::State(state): axum::extract::State<Arc<State>>,
    axum::extract::Json(req): axum::extract::Json<RTCSessionDescription>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    match broadcast_(state.out_rx.resubscribe(), req).await {
        Err(err) => {
            tracing::error!("{err:?}");
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("{err}")).into_response()
        }
        Ok(descr) => axum::Json(descr).into_response(),
    }
}

async fn broadcast_(
    mut rx: Receiver,
    client_offer: RTCSessionDescription,
) -> Result<RTCSessionDescription> {
    let peer_connection = crate::utils::peer_connection(48000, 1).await?;
    let options = Some(webrtc::data_channel::data_channel_init::RTCDataChannelInit {
        ordered: Some(false),
        max_retransmits: Some(0u16),
        ..Default::default()
    });
    // As we generate an answer and not an offer here, note that this requires the
    // offer to contain m=application which can be triggered by adding a call
    // to createDataChannel in the browser code that generates the offer (even if
    // the returned channel ends up not being used).
    let data_channel = peer_connection.create_data_channel("data", options).await?;
    let output_track = crate::utils::output_track(&peer_connection, 48000, 1).await?;
    data_channel.on_open(Box::new(|| {
        tracing::info!("on_open");
        Box::pin(async move {})
    }));
    peer_connection.set_remote_description(client_offer).await?;

    let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);
    let (start_tx, mut start_rx) = tokio::sync::mpsc::channel::<()>(1);
    peer_connection.on_data_channel(Box::new(move |_| {
        tracing::info!("on_data_channel");
        Box::pin(async {})
    }));
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
            let event = match rx.recv().await {
                Ok(event) => event,
                Err(err) => {
                    tracing::error!(?err, "no event");
                    break;
                }
            };
            match event.as_ref() {
                BroadcastEvent::Sample(sample) => {
                    if let Err(err) = output_track.write_sample(sample).await {
                        tracing::error!("output track write_rtp got error: {err}");
                        break;
                    }
                }
                BroadcastEvent::Text(text) => {
                    if let Err(err) = data_channel.send_text(text).await {
                        tracing::error!("output data_channel got error: {err}");
                        continue;
                    }
                }
            }
            if done_rx.try_recv().is_ok() {
                tracing::info!("exiting the send loop");
                break;
            }
        }
        tracing::info!("closing peer connection");
        if let Err(err) = peer_connection.close().await {
            tracing::error!(?err, "closing pc")
        }
    });
    Ok(local_desc)
}

async fn connect(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    axum::extract::State(state): axum::extract::State<Arc<State>>,
    axum::extract::Json(req): axum::extract::Json<RTCSessionDescription>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    match connect_(state, req).await {
        Err(err) => {
            tracing::error!("{err:?}");
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("{err}")).into_response()
        }
        Ok(descr) => axum::Json(descr).into_response(),
    }
}

async fn connect_(
    state: Arc<State>,
    client_offer: RTCSessionDescription,
) -> Result<RTCSessionDescription> {
    use rubato::Resampler;

    let (pcm_in_tx, pcm_in_rx) = std::sync::mpsc::channel();
    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let peer_connection = crate::utils::peer_connection(48000, 1).await?;
    tracing::info!("ask for offer");
    peer_connection.set_remote_description(client_offer).await?;
    let out_tx = state.out_tx.clone();
    let mut encoder = opus::Encoder::new(48000, opus::Channels::Mono, opus::Application::Voip)?;
    let mut resampler = rubato::FftFixedIn::<f32>::new(24000, 48000, 1920, 1, 1)?;
    tokio::task::spawn_blocking(move || {
        use webrtc::rtp::codecs::opus::OpusPayloader;
        let mut bytes = vec![0u8; 10000];
        while let Ok(event) = event_rx.recv() {
            match event {
                LogEvent::Pcm(pcm) => {
                    let pcm = match resampler.process(&[&pcm], None) {
                        Ok(mut v) => v.remove(0),
                        Err(err) => {
                            tracing::error!(?err, "resample");
                            continue;
                        }
                    };
                    let duration = pcm.len() as f32 / 24000f32;
                    let size = match encoder.encode_float(&pcm, &mut bytes) {
                        Ok(size) => size,
                        Err(err) => {
                            tracing::error!(?err, "opus");
                            continue;
                        }
                    };
                    let bytes = hyper::body::Bytes::copy_from_slice(&bytes[..size]);
                    let payloads = match OpusPayloader.payload(1500, &bytes) {
                        Ok(payloads) => payloads,
                        Err(err) => {
                            tracing::error!(?err, "payload");
                            continue;
                        }
                    };
                    for payload in payloads {
                        let sample = webrtc::media::Sample {
                            data: payload,
                            duration: std::time::Duration::from_secs_f32(duration),
                            ..Default::default()
                        };
                        // Do not fail on error as this happens when there is no receiver.
                        let _ = out_tx.send(Arc::new(BroadcastEvent::Sample(sample)));
                    }
                }
                LogEvent::Text(text) => {
                    // Do not fail on error as this happens when there is no receiver.
                    let _ = out_tx.send(Arc::new(BroadcastEvent::Text(text)));
                }
                LogEvent::TextToken(_) | LogEvent::AudioTokens(_) => {}
            }
        }
    });
    tokio::task::spawn_blocking(move || {
        if let Err(err) = state.lm.run(pcm_in_rx, event_tx) {
            tracing::error!(?err, "state.lm.run")
        }
    });
    peer_connection.on_track(Box::new(move |track, _, _| {
        tracing::info!("connection established {track:?}");
        let pcm_in_tx = pcm_in_tx.clone();
        tokio::spawn(async move {
            if let Err(err) = handle_input_rtp(track, pcm_in_tx).await {
                tracing::error!("{err:?}")
            }
        });
        Box::pin(async {})
    }));
    let (done_tx, _done_rx) = tokio::sync::mpsc::channel::<()>(1);
    let (start_tx, _start_rx) = tokio::sync::mpsc::channel::<()>(1);
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
    tracing::info!("desc {local_desc:?}");
    Ok(local_desc)
}

async fn handle_input_rtp(
    track: Arc<webrtc::track::track_remote::TrackRemote>,
    pcm_in_tx: std::sync::mpsc::Sender<Vec<f32>>,
) -> Result<()> {
    use rubato::Resampler;
    use webrtc::rtp::codecs::opus::OpusPacket;
    let mut decoder = opus::Decoder::new(48000, opus::Channels::Mono)?;

    tracing::info!(
        "Track has started, of type {}: {}",
        track.payload_type(),
        track.codec().capability.mime_type
    );
    let mut pcm_buf = vec![0f32; 48000];
    let mut pcm_buf_idx = 0;
    let mut resampler = rubato::FftFixedOut::<f32>::new(48000, 24000, 1920, 1, 1)?;
    // Read RTP packets being sent to webrtc-rs
    while let Ok((rtp, _)) = track.read_rtp().await {
        if !rtp.payload.is_empty() {
            let payload = OpusPacket.depacketize(&rtp.payload)?;
            let size = decoder.decode_float(
                &payload,
                &mut pcm_buf[pcm_buf_idx..],
                /* fec? */ false,
            )?;
            pcm_buf_idx += size;
            let ifn = resampler.input_frames_next();
            while pcm_buf_idx >= ifn {
                let mut pcm_resampled = resampler.process(&[&pcm_buf[..ifn]], None)?;
                pcm_in_tx.send(pcm_resampled.remove(0))?;
                pcm_buf.copy_within(ifn..pcm_buf_idx, 0);
                pcm_buf_idx -= ifn;
            }
        }
    }
    tracing::info!(
        "on_track finished, of type {}: {}",
        track.payload_type(),
        track.codec().capability.mime_type
    );
    Ok(())
}
