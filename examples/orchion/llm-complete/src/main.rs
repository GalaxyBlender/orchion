use std::error::Error;
use std::io;

use orchion::{
    GenerationOptions, GenerationRequest, LlmDeployment, LlmEngine, LlmEngineConfig, LlmMessage,
    LlmModel, LlmRole, ModelId, ModelUrl,
};

const USAGE: &str = "usage: llm-complete <model-id> <exact-model-url> <prompt> [cache_dir]";

type MainResult<T> = Result<T, Box<dyn Error>>;

#[tokio::main]
async fn main() -> MainResult<()> {
    let mut args = std::env::args().skip(1);
    let model = LlmModel::new(ModelId::parse(&required(&mut args)?)?);
    let source = ModelUrl::parse(&required(&mut args)?)?;
    let prompt = required(&mut args)?;
    let cache_dir = args.next().unwrap_or_else(|| "models".to_string());
    if args.next().is_some() {
        return Err(usage_error().into());
    }

    let deployment = LlmDeployment::provision(model, source, cache_dir).await?;
    let engine = LlmEngine::load_deployment(deployment, LlmEngineConfig::default()).await?;
    let completion = engine.complete(request(prompt)).await;
    engine.shutdown();
    let completion = completion?;

    println!("text: {}", completion.text);
    println!("usage: {:#?}", completion.usage);
    Ok(())
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
