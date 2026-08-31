use orchion_client::ocr::{OcrRequest, OcrResponse, OcrResponseFormat};
use orchion_client::{Client, ClientConfig};
use std::error::Error;
use std::io;
use std::path::Path;

const USAGE: &str = "usage: client-ocr <base-url> <image-file> <model> [json|text|markdown|html]";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = ranged_args(3, 4)?;
    let filename = filename(&args[1])?;
    let mut request = OcrRequest::new(filename, &args[2])
        .with_file_path(&args[1])
        .await?;
    if let Some(format) = args.get(3) {
        request = request.with_response_format(parse_format(format)?);
    }

    let client = make_client(&args[0])?;
    match client.ocr().recognize(request).await? {
        OcrResponse::Json(response) => {
            println!("model: {}", response.model);
            println!("format: {:?}", response.format);
            println!("text:\n{}", response.text);
            if let Some(markdown) = response.markdown {
                println!("markdown:\n{markdown}");
            }
            if let Some(html) = response.html {
                println!("html:\n{html}");
            }
            println!("regions: {}", response.regions.len());
            println!("layout blocks: {}", response.layout_blocks.len());
            println!("usage: {:?}", response.usage);
        }
        OcrResponse::Text(response) => print!("{response}"),
    }
    Ok(())
}

fn parse_format(value: &str) -> Result<OcrResponseFormat, io::Error> {
    match value {
        "json" => Ok(OcrResponseFormat::Json),
        "text" => Ok(OcrResponseFormat::Text),
        "markdown" => Ok(OcrResponseFormat::Markdown),
        "html" => Ok(OcrResponseFormat::Html),
        _ => Err(io::Error::new(io::ErrorKind::InvalidInput, USAGE)),
    }
}

fn filename(path: &str) -> Result<String, io::Error> {
    Path::new(path)
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "image path has no filename"))
}

fn ranged_args(min: usize, max: usize) -> Result<Vec<String>, io::Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if (min..=max).contains(&args.len()) {
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
