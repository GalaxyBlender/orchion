use orchion_client::asr::{StreamingEvent, StreamingInputAudioFormat, StreamingStartRequest};
use orchion_client::{Client, ClientConfig};
use std::error::Error;
use std::io;
use std::path::Path;

const CHUNK_BYTES: usize = 64 * 1024;
const USAGE: &str = "usage: client-asr-streaming <base-url> <audio-file.wav> <model>";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = required_args(3)?;
    require_wav(&args[1])?;
    let audio = tokio::fs::read(&args[1]).await?;
    if audio.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "audio file is empty").into());
    }

    let client = make_client(&args[0])?;
    let request = StreamingStartRequest::new(&args[2], StreamingInputAudioFormat::Wav);
    // start_streaming validates and consumes the server's Ready event before returning.
    let mut session = client.asr().start_streaming(request).await?;
    for chunk in audio.chunks(CHUNK_BYTES) {
        session.send_audio(chunk.to_vec()).await?;
    }
    session.finish().await?;

    loop {
        match session.next_event().await? {
            Some(StreamingEvent::Partial { text }) => println!("partial: {text}"),
            Some(StreamingEvent::CaptionPartial { segment_id, text }) => {
                println!("partial[{segment_id}]: {text}");
            }
            Some(StreamingEvent::SegmentFinal {
                segment_id,
                text,
                start_ms,
                end_ms,
            }) => println!("segment[{segment_id}] {start_ms:?}-{end_ms:?} ms: {text}"),
            Some(StreamingEvent::Final { text }) => {
                println!("final: {text}");
                break;
            }
            Some(StreamingEvent::Completed) => break,
            Some(StreamingEvent::Error { error }) => {
                return Err(
                    io::Error::other(format!("server streaming error: {}", error.message)).into(),
                );
            }
            Some(StreamingEvent::Ready) => {
                return Err(io::Error::other("unexpected duplicate Ready event").into());
            }
            None => return Err(io::Error::other("stream closed without a terminal event").into()),
        }
    }
    Ok(())
}

fn require_wav(path: &str) -> Result<(), io::Error> {
    let is_wav = Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("wav"));
    if is_wav {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidInput, USAGE))
    }
}

fn required_args(count: usize) -> Result<Vec<String>, io::Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == count {
        Ok(args)
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidInput, USAGE))
    }
}

fn make_client(base_url: &str) -> Result<Client, Box<dyn Error>> {
    let mut config = ClientConfig::new(base_url)?;
    match std::env::var("ORCHION_API_KEY") {
        Ok(api_key) if !api_key.is_empty() => config = config.with_api_key(api_key),
        Ok(_) | Err(std::env::VarError::NotPresent) => {}
        Err(error) => return Err(error.into()),
    }
    Ok(Client::from_config(config)?)
}
