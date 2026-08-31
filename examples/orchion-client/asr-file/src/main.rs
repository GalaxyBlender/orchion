use orchion_client::asr::{
    TimestampGranularity, TranscriptionFormat, TranscriptionRequest, TranscriptionResponse,
};
use orchion_client::{Client, ClientConfig};
use std::error::Error;
use std::io;
use std::path::Path;

const USAGE: &str = "usage: client-asr-file <base-url> <audio-file> <model>";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = required_args(3)?;
    let filename = filename(&args[1])?;
    let client = make_client(&args[0])?;
    let request = TranscriptionRequest::new(&args[2], filename)
        .with_file_path(&args[1])
        .await?
        .with_response_format(TranscriptionFormat::VerboseJson)
        .with_timestamp_granularity(TimestampGranularity::Segment);

    match client.asr().transcribe(request).await? {
        TranscriptionResponse::VerboseJson(response) => {
            println!("{}", response.text);
            for segment in response.segments.unwrap_or_default() {
                println!(
                    "[{:.3} - {:.3}] {}",
                    segment.start, segment.end, segment.text
                );
            }
        }
        response => {
            return Err(io::Error::other(format!("unexpected response: {response:?}")).into());
        }
    }
    Ok(())
}

fn filename(path: &str) -> Result<String, io::Error> {
    Path::new(path)
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "audio path has no filename"))
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
