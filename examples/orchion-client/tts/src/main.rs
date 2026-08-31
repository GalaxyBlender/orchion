use orchion_client::tts::SpeechRequest;
use orchion_client::{Client, ClientConfig};
use std::error::Error;
use std::io;

const USAGE: &str = "usage: client-tts <base-url> <model> <text> <voice> <output-file>";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = required_args(5)?;
    let client = make_client(&args[0])?;
    let response = client
        .tts()
        .create_speech(SpeechRequest::new(&args[1], &args[2], &args[3]))
        .await?;
    if response.bytes.is_empty() {
        return Err(io::Error::other("server returned empty speech audio").into());
    }
    tokio::fs::write(&args[4], &response.bytes).await?;
    println!("wrote {} bytes to {}", response.bytes.len(), args[4]);
    if let Some(content_type) = response.content_type {
        println!("content-type: {content_type}");
    }
    Ok(())
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
