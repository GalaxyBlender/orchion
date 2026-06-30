use crate::ModelId;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "schema")]
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum OcrResponseFormat {
    #[default]
    Json,
    Text,
    Markdown,
    Html,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum OcrTask {
    #[default]
    Ocr,
    Table,
    Formula,
    Chart,
    Spotting,
    Seal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrOptions {
    pub response_format: OcrResponseFormat,
    pub task: OcrTask,
    pub layout_model: Option<ModelId>,
    pub max_tokens: Option<usize>,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            response_format: OcrResponseFormat::Json,
            task: OcrTask::Ocr,
            layout_model: None,
            max_tokens: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct OcrRegion {
    pub text: String,
    pub confidence: Option<f32>,
    pub polygon: Vec<OcrPoint>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct OcrPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct OcrLayoutBlock {
    pub label: String,
    pub confidence: Option<f32>,
    pub polygon: Vec<OcrPoint>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct OcrUsage {
    pub input_pages: usize,
    pub output_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct OcrResult {
    pub model: ModelId,
    pub format: OcrResponseFormat,
    pub text: String,
    pub markdown: Option<String>,
    pub html: Option<String>,
    pub regions: Vec<OcrRegion>,
    pub layout_blocks: Vec<OcrLayoutBlock>,
    pub usage: OcrUsage,
}
