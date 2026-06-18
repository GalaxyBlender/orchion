use anyhow::Context;
use clap::Parser;
use orchion_server::{config::ServerConfig, routes, state::AppState};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "orchion-server", about = "OpenAI-compatible ASR/TTS server")]
struct Cli {
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let config = ServerConfig::load(cli.config).context("load server config")?;
    let bind = config.server.bind;
    let state = AppState::load(config)
        .await
        .context("initialize app state")?;
    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!(%bind, "orchion server listening");
    axum::serve(listener, app).await.context("serve HTTP")
}
