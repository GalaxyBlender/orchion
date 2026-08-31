use orchion_client::pdf::PdfImagesRequest;
use orchion_client::{Client, ClientConfig};
use std::error::Error;
use std::io;
use std::path::Path;

const USAGE: &str = "usage: client-pdf <base-url> <pdf-file> <output.zip> [pages] [scale]";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = ranged_args(3, 5)?;
    let filename = filename(&args[1])?;
    let mut request = PdfImagesRequest::new(filename)
        .with_file_path(&args[1])
        .await?;
    if let Some(pages) = args.get(3) {
        request = request.with_pages(pages);
    }
    if let Some(scale) = args.get(4) {
        request = request.with_scale(scale.parse()?);
    }

    let client = make_client(&args[0])?;
    let response = client.pdf().render_images(request).await?;
    if response.bytes.is_empty() {
        return Err(io::Error::other("server returned an empty PDF image archive").into());
    }
    tokio::fs::write(&args[2], &response.bytes).await?;
    println!("wrote {} bytes to {}", response.bytes.len(), args[2]);
    println!("page_count: {:?}", response.page_count);
    println!("image_count: {:?}", response.image_count);
    println!("content_disposition: {:?}", response.content_disposition);
    Ok(())
}

fn filename(path: &str) -> Result<String, io::Error> {
    Path::new(path)
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "PDF path has no filename"))
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
