#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

mod codec;

pub use codec::{
    AudioOutputFormat, DecodedAudio, EncodedAudio, FfmpegAudioCodec, decode_audio_bytes,
    encode_tts_audio,
};
