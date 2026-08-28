use anyhow::Context;
use clap::Parser;
use orchion_server::{
    api::http, infrastructure::orchion::AppState, logging, settings::ServerConfig,
};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

#[derive(Debug, Parser)]
#[command(name = "orchion-server", about = "OpenAI-compatible ASR/TTS server")]
struct Cli {
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    models_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("orchion server failed: {error:#}");
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

    let mut config = ServerConfig::load(cli.config).context("load server config")?;
    if let Some(models_dir) = cli.models_dir {
        config.models.dir = if models_dir.is_absolute() {
            models_dir
        } else {
            work_dir.join(models_dir)
        };
    }
    let bind = config.server.bind;
    tracing::debug!(
        %bind,
        asr_service = ?config.services.asr,
        tts_service = ?config.services.tts,
        "server config loaded"
    );
    tracing::debug!(
        config_path = %config.config_path.display(),
        models_dir = %config.models.dir.display(),
        max_upload_size = config.server.max_upload_size,
        models_source = ?config.models.source,
        models_max_loaded = config.models.max_loaded,
        default_tts_format = %config.services.tts.format,
        "server config details loaded"
    );
    let state = AppState::load(config)
        .await
        .context("initialize app state")?;
    let app = http::router(Arc::clone(&state));
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!(%bind, "orchion server listening");
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serve HTTP");
    state.shutdown().await;
    result
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(signal) => signal,
                Err(error) => {
                    tracing::error!(%error, "failed to install SIGTERM handler");
                    if let Err(error) = tokio::signal::ctrl_c().await {
                        tracing::error!(%error, "failed to install shutdown signal handler");
                    }
                    return;
                }
            };
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                if let Err(error) = result {
                    tracing::error!(%error, "failed to install shutdown signal handler");
                }
            }
            _ = terminate.recv() => {}
        }
    }

    #[cfg(not(unix))]
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(%error, "failed to install shutdown signal handler");
    }
}
