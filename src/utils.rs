// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use std::sync::Arc;

use anyhow::Result;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_OPUS};
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::rtp_transceiver::rtp_codec::{
    RTCRtpCodecCapability, RTCRtpCodecParameters, RTPCodecType,
};
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use webrtc::track::track_local::TrackLocal;

// This must be an allowed value among 120, 240, 480, 960, 1920, and 2880.
// Using a different value would result in a BadArg "invalid argument" error when calling encode.
// https://opus-codec.org/docs/opus_api-1.2/group__opus__encoder.html#ga4ae9905859cd241ef4bb5c59cd5e5309
// Tweaking the values below doesn't seem to work well.
pub(crate) const OPUS_ENCODER_FRAME_SIZE: usize = 960;

pub async fn peer_connection(clock_rate: u32, channels: u16) -> Result<Arc<RTCPeerConnection>> {
    // Create a MediaEngine object to configure the supported codec
    let mut m = MediaEngine::default();
    m.register_codec(
        RTCRtpCodecParameters {
            capability: RTCRtpCodecCapability {
                mime_type: MIME_TYPE_OPUS.to_owned(),
                clock_rate,
                channels,
                ..Default::default()
            },
            payload_type: 120,
            ..Default::default()
        },
        RTPCodecType::Audio,
    )?;

    // Create a InterceptorRegistry. This is the user configurable RTP/RTCP Pipeline.
    // This provides NACKs, RTCP Reports and other features. If you use `webrtc.NewPeerConnection`
    // this is enabled by default. If you are manually managing You MUST create a InterceptorRegistry
    // for each PeerConnection.
    let mut registry = Registry::new();

    // Use the default set of Interceptors
    registry = register_default_interceptors(registry, &mut m)?;

    // Create the API object with the MediaEngine
    let api = webrtc::api::APIBuilder::new()
        .with_media_engine(m)
        .with_interceptor_registry(registry)
        .build();

    // Prepare the configuration
    let config = RTCConfiguration {
        bundle_policy: webrtc::peer_connection::policy::bundle_policy::RTCBundlePolicy::MaxBundle,
        rtcp_mux_policy:
            webrtc::peer_connection::policy::rtcp_mux_policy::RTCRtcpMuxPolicy::Require,
        ice_servers: vec![RTCIceServer {
            urls: vec!["stun:global.stun.twilio.com:3478".to_owned()],
            ..Default::default()
        }],
        ..Default::default()
    };
    Ok(Arc::new(api.new_peer_connection(config).await?))
}

pub async fn output_track(
    peer_connection: &RTCPeerConnection,
    clock_rate: u32,
    channels: u16,
) -> Result<Arc<TrackLocalStaticSample>> {
    let output_track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_OPUS.to_owned(),
            clock_rate,
            channels,
            ..Default::default()
        },
        "track-audio".into(),
        "webrtc-rs".into(),
    ));

    // Add this newly created track to the PeerConnection
    let rtp_sender = peer_connection
        .add_track(Arc::clone(&output_track) as Arc<dyn TrackLocal + Send + Sync>)
        .await?;

    // Read incoming RTCP packets
    // Before these packets are returned they are processed by interceptors. For things
    // like NACK this needs to be called.
    tokio::spawn(async move {
        let mut rtcp_buf = vec![0u8; 1500];
        while let Ok((_, _)) = rtp_sender.read(&mut rtcp_buf).await {}
        tracing::info!("audio rtp_sender.read loop exit");
        Result::<()>::Ok(())
    });
    Ok(output_track)
}

pub fn device(cpu: bool) -> Result<candle::Device> {
    use candle::Device;
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn replace_env_vars(input: &str) -> String {
    let re = regex::Regex::new(r"\$([A-Za-z_][A-Za-z0-9_]*)").unwrap();
    re.replace_all(input, |caps: &regex::Captures| {
        let var_name = &caps[1];
        std::env::var(var_name).unwrap_or_else(|_| "".to_string())
    })
    .to_string()
}

pub fn resolve_or_download(input: &str) -> Result<String> {
    let path = match input.strip_prefix("hf://") {
        None => replace_env_vars(input),
        Some(path) => {
            let s: Vec<&str> = path.split('/').collect();
            if s.len() < 3 {
                anyhow::bail!("unexpected format for hf path {input}")
            }
            let repo = format!("{}/{}", s[0], s[1]);
            let file = s[2..].join("/");
            let api = hf_hub::api::sync::Api::new()?.model(repo);
            api.get(&file)?.to_string_lossy().to_string()
        }
    };
    Ok(path)
}
