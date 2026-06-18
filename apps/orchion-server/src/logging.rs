use std::io;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

const DEFAULT_RUST_LOG: &str = "orchion_server=info,orchion=info,tower_http=info";

#[must_use]
pub const fn default_rust_log() -> &'static str {
    DEFAULT_RUST_LOG
}

pub fn init(exe_path: &Path, work_dir: &Path) -> anyhow::Result<String> {
    let explicit = std::env::var("RUST_LOG").ok();
    let directive = rust_log_directive_from_sources(exe_path, work_dir, explicit.as_deref())?;
    let filter = EnvFilter::try_new(&directive)?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(directive)
}

pub fn rust_log_directive_from_sources(
    exe_path: &Path,
    work_dir: &Path,
    explicit: Option<&str>,
) -> io::Result<String> {
    if let Some(value) = non_empty_value(explicit) {
        return Ok(value.to_string());
    }

    for path in dotenv_candidates(exe_path, work_dir) {
        if let Some(value) = rust_log_from_dotenv(&path)? {
            return Ok(value);
        }
    }

    Ok(DEFAULT_RUST_LOG.to_string())
}

fn dotenv_candidates(exe_path: &Path, work_dir: &Path) -> Vec<PathBuf> {
    let exe_env = exe_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".env");
    let work_env = work_dir.join(".env");
    if exe_env == work_env {
        vec![exe_env]
    } else {
        vec![exe_env, work_env]
    }
}

fn rust_log_from_dotenv(path: &Path) -> io::Result<Option<String>> {
    let document = match std::fs::read_to_string(path) {
        Ok(document) => document,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };

    Ok(document.lines().find_map(parse_rust_log_line))
}

fn parse_rust_log_line(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
    let (key, value) = line.split_once('=')?;
    if key.trim() != "RUST_LOG" {
        return None;
    }
    non_empty_value(Some(value)).map(ToOwned::to_owned)
}

fn non_empty_value(value: Option<&str>) -> Option<&str> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    Some(unquote(value))
}

fn unquote(value: &str) -> &str {
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        &value[1..value.len() - 1]
    } else {
        value
    }
}
