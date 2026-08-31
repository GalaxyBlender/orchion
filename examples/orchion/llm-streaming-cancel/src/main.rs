use std::error::Error;
use std::io::{self, Write};

use orchion::{
    GenerationEvent, GenerationOptions, GenerationRequest, LlmDeployment, LlmEngine,
    LlmEngineConfig, LlmGeneration, LlmMessage, LlmModel, LlmRole, ModelId, ModelUrl,
};

const USAGE: &str = "usage: llm-streaming-cancel <model-id> <exact-model-url> <prompt> [cancel-after-deltas] [cache_dir]";
const DEFAULT_CANCEL_AFTER_DELTAS: usize = 8;

type MainResult<T> = Result<T, Box<dyn Error>>;

#[tokio::main]
async fn main() -> MainResult<()> {
    let mut args = std::env::args().skip(1);
    let model = LlmModel::new(ModelId::parse(&required(&mut args)?)?);
    let source = ModelUrl::parse(&required(&mut args)?)?;
    let prompt = required(&mut args)?;
    let cancel_after = args
        .next()
        .map_or(Ok(DEFAULT_CANCEL_AFTER_DELTAS), |value| value.parse())?;
    if cancel_after == 0 {
        return Err(usage_error().into());
    }
    let cache_dir = args.next().unwrap_or_else(|| "models".to_string());
    if args.next().is_some() {
        return Err(usage_error().into());
    }

    let deployment = LlmDeployment::provision(model, source, cache_dir).await?;
    let engine = LlmEngine::load_deployment(deployment, LlmEngineConfig::default()).await?;
    let result = match engine.stream(request(prompt)).await {
        Ok(generation) => stream_until_terminal(generation, cancel_after).await,
        Err(error) => Err(error.into()),
    };
    engine.shutdown();
    result
}

async fn stream_until_terminal(
    mut generation: LlmGeneration,
    cancel_after: usize,
) -> MainResult<()> {
    let mut deltas = 0;
    let mut cancellation_requested = false;
    while let Some(event) = generation.next().await? {
        match event {
            GenerationEvent::ContentDelta(delta) => {
                print!("{delta}");
                io::stdout().flush()?;
                deltas += 1;
                if deltas >= cancel_after && !cancellation_requested {
                    generation.cancel();
                    cancellation_requested = true;
                }
            }
            GenerationEvent::Finished { reason, usage } => {
                println!("\nfinish: {reason:?}");
                println!("usage: {usage:#?}");
                return Ok(());
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "LLM stream ended without a terminal event",
    )
    .into())
}

fn request(prompt: String) -> GenerationRequest {
    GenerationRequest {
        messages: vec![LlmMessage {
            role: LlmRole::User,
            content: prompt,
        }],
        options: GenerationOptions::default(),
    }
}

fn required(args: &mut impl Iterator<Item = String>) -> Result<String, io::Error> {
    args.next().ok_or_else(usage_error)
}

fn usage_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, USAGE)
}
