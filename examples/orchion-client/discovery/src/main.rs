use orchion_client::{Client, ClientConfig};
use std::error::Error;
use std::io;

const USAGE: &str = "usage: client-discovery <base-url>";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let base_url = required_args(1)?.remove(0);
    let client = make_client(&base_url)?;

    client.health().check().await?;
    println!("health: ok");

    let models = client.models().list().await?;
    println!("models:");
    for model in models.data {
        println!("  {} ({:?})", model.id, model.model_type);
        println!("    capabilities: {:?}", model.capabilities);
    }

    let statuses = client.models().list_statuses().await?;
    println!("residency:");
    for status in statuses.data {
        println!(
            "  {} [{:?}]: {:?}",
            status.id, status.service, status.status
        );
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
