#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

mod codec;

pub use codec::{
    AudioInputFormat, AudioOutputFormat, DecodedAudio, EncodedAudio, FfmpegAudioCodec,
    StreamingAudioDecoder, decode_audio_bytes, decode_audio_bytes_with_max_samples,
    decode_audio_file, decode_audio_file_with_max_samples, decode_pcm_s16le_bytes,
    encode_tts_audio,
};
