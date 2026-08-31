use std::error::Error;
use std::io;
use std::path::PathBuf;

use orchion::{Tts, TtsLanguage, TtsModel, TtsSpeaker, TtsVoice};

const USAGE: &str = "usage:
  tts-voice-modes preset <text> <output.wav> [cache_dir]
  tts-voice-modes design <text> <voice-prompt> <output.wav> [cache_dir]
  tts-voice-modes clone <text> <reference-audio> <reference-text> <output.wav> [cache_dir]";

type MainResult<T> = Result<T, Box<dyn Error>>;

#[tokio::main]
async fn main() -> MainResult<()> {
    let mut args = std::env::args().skip(1);
    let mode = required(&mut args)?;
    let (model_id, text, voice, output_path) = match mode.as_str() {
        "preset" => (
            "alibaba/qwen3-tts-12hz-0.6b-customvoice",
            required(&mut args)?,
            TtsVoice::Preset {
                speaker: TtsSpeaker::Ryan,
                language: TtsLanguage::English,
            },
            required(&mut args)?,
        ),
        "design" => (
            "alibaba/qwen3-tts-12hz-1.7b-voicedesign",
            required(&mut args)?,
            TtsVoice::Design {
                prompt: required(&mut args)?,
                language: TtsLanguage::English,
            },
            required(&mut args)?,
        ),
        "clone" => (
            "alibaba/qwen3-tts-12hz-0.6b-base",
            required(&mut args)?,
            TtsVoice::Clone {
                reference_audio: PathBuf::from(required(&mut args)?),
                reference_text: required(&mut args)?,
                language: TtsLanguage::English,
            },
            required(&mut args)?,
        ),
        _ => return Err(usage_error().into()),
    };
    let cache_dir = args.next().unwrap_or_else(|| "models".to_string());
    if args.next().is_some() {
        return Err(usage_error().into());
    }

    let model = TtsModel::parse(model_id)?;
    let tts = Tts::load_or_download(model, cache_dir).await?;
    tts.synthesize_to_file(text, voice, output_path).await?;
    Ok(())
}

fn required(args: &mut impl Iterator<Item = String>) -> Result<String, io::Error> {
    args.next().ok_or_else(usage_error)
}

fn usage_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, USAGE)
}
