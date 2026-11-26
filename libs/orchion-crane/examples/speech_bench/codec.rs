use crate::{measure_tts, memory::Memory, streaming::device_for_benchmark};
use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Tensor};
use crane_core::models::qwen3_tts::speech_tokenizer_v2::NativeSpeechTokenizerDecoder;
use orchion_core::{DevicePreference, TtsAudio};
use std::path::Path;

pub(super) fn bench(
    preference: DevicePreference,
    model: &Path,
    schedule: &str,
    output: &Path,
    memory: &Memory,
) -> Result<()> {
    let stream = match schedule {
        "batch" => false,
        "stream" => true,
        _ => bail!("codec schedule must be batch or stream"),
    };
    let directory = model.join("speech_tokenizer");
    let directory = directory
        .to_str()
        .context("tokenizer directory is not UTF-8")?;
    let device = device_for_benchmark(preference)?;
    let decoder = NativeSpeechTokenizerDecoder::new(directory, &device, DType::F32)?;
    println!("codec_loaded {}", memory.snapshot()?);
    // Same deterministic, valid Qwen3-TTS codebook IDs for both schedules.
    let codes = (0_i64..16 * 333)
        .map(|index| (index * 31 + index / 19) % 2048)
        .collect::<Vec<_>>();
    let codes = Tensor::from_vec(codes, (1, 16, 333), &device)?;
    let warmup = decoder.decode_chunk(&codes.narrow(D::Minus1, 0, 5)?, 0)?;
    drop(warmup);
    let mut run: crate::Inference<'_, usize, TtsAudio> = Box::new(|&frames| {
        let codes = codes.narrow(D::Minus1, 0, frames)?;
        let samples = if stream {
            decode_stream_windows(&decoder, &codes, frames)?
        } else {
            decoder
                .chunked_decode(&codes, 300, 25)?
                .flatten_all()?
                .to_vec1()?
        };
        Ok(TtsAudio::new(samples, decoder.sample_rate()))
    });
    measure_tts(
        &mut run,
        &[("codec_82", &82), ("codec_333", &333)],
        memory,
        Some(output),
    )
}

fn decode_stream_windows(
    decoder: &NativeSpeechTokenizerDecoder,
    codes: &Tensor,
    frames: usize,
) -> Result<Vec<f32>> {
    let mut samples = Vec::new();
    let mut emitted = 0;
    while emitted < frames {
        let left = emitted.min(25);
        let new = if emitted == 0 { 5 } else { 25 };
        let new = new.min(frames - emitted);
        let window = codes.narrow(D::Minus1, emitted - left, left + new)?;
        let pcm = decoder
            .decode_chunk(&window, left)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        samples.extend(pcm);
        emitted += new;
    }
    Ok(samples)
}
