use crate::infrastructure::orchion::AppState;
use axum::Router;
use axum::body::Body;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
#[cfg(not(debug_assertions))]
use include_dir::{Dir, include_dir};
use std::borrow::Cow;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;

const MISSING_DIST_MESSAGE: &str = "web/dist was not found. Run `bun install && bun run build` in the web/ directory before opening /ui in a debug build.";

#[cfg(not(debug_assertions))]
static UI_DIST: Dir<'_> = include_dir!("$OUT_DIR/ui-dist");

pub struct UiAssets {
    source: UiAssetSource,
}

enum UiAssetSource {
    Path(PathBuf),
    #[cfg(not(debug_assertions))]
    Embedded,
}

pub fn routes() -> Router<Arc<AppState>> {
    #[cfg(debug_assertions)]
    {
        routes_from_path(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/dist"))
    }

    #[cfg(not(debug_assertions))]
    {
        routes_for_assets(UiAssets {
            source: UiAssetSource::Embedded,
        })
    }
}

pub fn routes_from_path(path: impl Into<PathBuf>) -> Router<Arc<AppState>> {
    routes_for_assets(UiAssets {
        source: UiAssetSource::Path(path.into()),
    })
}

pub fn routes_for_assets(assets: UiAssets) -> Router<Arc<AppState>> {
    let assets = Arc::new(assets);
    Router::new()
        .route(
            "/ui",
            get({
                let assets = Arc::clone(&assets);
                move || serve_ui_path(Arc::clone(&assets), String::new())
            }),
        )
        .route(
            "/ui/",
            get({
                let assets = Arc::clone(&assets);
                move || serve_ui_path(Arc::clone(&assets), String::new())
            }),
        )
        .route(
            "/ui/{*path}",
            get({
                let assets = Arc::clone(&assets);
                move |Path(path): Path<String>| serve_ui_path(Arc::clone(&assets), path)
            }),
        )
}

async fn serve_ui_path(assets: Arc<UiAssets>, requested_path: String) -> Response {
    let Some(path) = normalize_path(&requested_path) else {
        return not_found_response();
    };
    let index = match assets.read_file("index.html").await {
        Ok(Some(index)) => index,
        Ok(None) => return missing_dist_response(),
        Err(error) => return read_error_response(error),
    };

    if path.is_empty() {
        return asset_response("index.html", index);
    }

    match assets.read_file(path).await {
        Ok(Some(asset)) => asset_response(path, asset),
        Ok(None) if has_file_extension(path) => not_found_response(),
        Ok(None) => asset_response("index.html", index),
        Err(error) => read_error_response(error),
    }
}

impl UiAssets {
    async fn read_file(&self, path: &str) -> std::io::Result<Option<Cow<'static, [u8]>>> {
        match &self.source {
            UiAssetSource::Path(root) => read_disk_file(root, path).await,
            #[cfg(not(debug_assertions))]
            UiAssetSource::Embedded => Ok(UI_DIST
                .get_file(path)
                .map(|file| Cow::Borrowed(file.contents()))),
        }
    }
}

async fn read_disk_file(root: &FsPath, path: &str) -> std::io::Result<Option<Cow<'static, [u8]>>> {
    match tokio::fs::read(root.join(path)).await {
        Ok(bytes) => Ok(Some(Cow::Owned(bytes))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn normalize_path(path: &str) -> Option<&str> {
    if path.is_empty() {
        return Some(path);
    }
    if path.starts_with('/')
        || path.contains(['\\', ':'])
        || path
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return None;
    }
    Some(path)
}

fn has_file_extension(path: &str) -> bool {
    FsPath::new(path).extension().is_some()
}

fn asset_response(path: &str, bytes: Cow<'static, [u8]>) -> Response {
    let body = match bytes {
        Cow::Borrowed(bytes) => Body::from(bytes),
        Cow::Owned(bytes) => Body::from(bytes),
    };
    response_with_body(StatusCode::OK, content_type(path), body)
}

fn missing_dist_response() -> Response {
    response_with_body(
        StatusCode::SERVICE_UNAVAILABLE,
        "text/plain; charset=utf-8",
        Body::from(MISSING_DIST_MESSAGE),
    )
}

fn not_found_response() -> Response {
    response_with_body(
        StatusCode::NOT_FOUND,
        "text/plain; charset=utf-8",
        Body::from("not found"),
    )
}

fn read_error_response(error: std::io::Error) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("failed to read UI asset: {error}"),
    )
        .into_response()
}

fn response_with_body(status: StatusCode, content_type: &'static str, body: Body) -> Response {
    match Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .body(body)
    {
        Ok(response) => response,
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to build UI response: {error}"),
        )
            .into_response(),
    }
}

fn content_type(path: &str) -> &'static str {
    match FsPath::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("ico") => "image/x-icon",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}
