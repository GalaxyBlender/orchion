use orchion_client::activity::{ActivityEvent, ActivityQuery};
use orchion_client::models::{ModelControlRequest, ModelService};
use orchion_client::{Client, ClientConfig};
use std::error::Error;
use std::io;
use std::time::Duration;

const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(10);
const USAGE: &str = "usage: client-operations <base-url> <model> <asr|tts|ocr|ocr-vl|llm>";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = required_args(3)?;
    let service = parse_service(&args[2])?;
    let client = make_client(&args[0])?;

    print_statuses("initial statuses", &client).await?;
    let loaded = client
        .models()
        .load(ModelControlRequest::new(&args[1], service))
        .await?;
    println!("load: {} -> {:?}", loaded.id, loaded.status);
    let unloaded = client
        .models()
        .unload(ModelControlRequest::new(&args[1], service))
        .await?;
    println!("unload: {} -> {:?}", unloaded.id, unloaded.status);
    print_statuses("final statuses", &client).await?;

    let activity = client
        .activity()
        .list(ActivityQuery::new().with_limit(20)?)
        .await?;
    println!(
        "activity: enabled={}, active={}, retained={}",
        activity.enabled, activity.summary.active, activity.summary.retained
    );
    for entry in activity.active.iter().chain(&activity.history) {
        println!(
            "  {} {:?} {:?} model={:?} outcome={:?}",
            entry.id, entry.operation, entry.state, entry.model, entry.outcome
        );
    }

    let mut stream = client.activity().subscribe().await?;
    let event = tokio::time::timeout(SNAPSHOT_TIMEOUT, stream.next_event())
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for activity snapshot",
            )
        })??;
    match event {
        Some(ActivityEvent::Snapshot {
            cursor,
            active,
            summary,
        }) => println!(
            "snapshot: cursor={cursor}, active={}, retained={}",
            active.len(),
            summary.retained
        ),
        Some(other) => {
            return Err(io::Error::other(format!("expected snapshot, received {other:?}")).into());
        }
        None => return Err(io::Error::other("activity stream closed before snapshot").into()),
    }
    Ok(())
}

async fn print_statuses(label: &str, client: &Client) -> Result<(), Box<dyn Error>> {
    let statuses = client.models().list_statuses().await?;
    println!("{label}:");
    for status in statuses.data {
        println!(
            "  {} [{:?}]: {:?}",
            status.id, status.service, status.status
        );
    }
    Ok(())
}

fn parse_service(value: &str) -> Result<ModelService, io::Error> {
    match value {
        "asr" => Ok(ModelService::Asr),
        "tts" => Ok(ModelService::Tts),
        "ocr" => Ok(ModelService::Ocr),
        "ocr-vl" | "ocr_vl" => Ok(ModelService::OcrVl),
        "llm" => Ok(ModelService::Llm),
        _ => Err(io::Error::new(io::ErrorKind::InvalidInput, USAGE)),
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
