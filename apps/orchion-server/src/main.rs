use anyhow::Context;
use clap::Parser;
use orchion_server::{
    api::http, infrastructure::orchion::AppState, logging, settings::ServerConfig,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(name = "orchion-server", about = "OpenAI-compatible ASR/TTS server")]
struct Cli {
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(error = %format!("{error:#}"), "orchion server failed");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let exe_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("orchion-server"));
    let work_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let rust_log = logging::init(&exe_path, &work_dir).context("initialize logging")?;
    tracing::debug!(
        %rust_log,
        exe_path = %exe_path.display(),
        work_dir = %work_dir.display(),
        "logging initialized"
    );

    let config = ServerConfig::load(cli.config).context("load server config")?;
    let bind = config.server.bind;
    tracing::debug!(
        %bind,
        asr_model = ?config.models.asr,
        tts_model = ?config.models.tts,
        "server config loaded"
    );
    tracing::debug!(
        config_path = %config.config_path.display(),
        models_dir = %config.models.dir.display(),
        max_upload_size = config.server.max_upload_size,
        model_source = ?config.models.source,
        default_tts_format = %config.defaults.tts.format,
        "server config details loaded"
    );
    let state = AppState::load(config)
        .await
        .context("initialize app state")?;
    let app = http::router(state);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!(%bind, "orchion server listening");
    axum::serve(listener, app).await.context("serve HTTP")
}
