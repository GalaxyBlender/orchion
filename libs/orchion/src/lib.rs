//! In-process Rust facade for Orchion model inference.
//!
//! This crate keeps model selection, downloading, backend adapters, and runtime loading behind a
//! single interface. Features are opt-in so applications only compile the inference domains and
//! hardware backends they use.
//!
//! # Feature model
//!
//! - `asr-engine` and `tts-engine` expose adapter seams without selecting an implementation.
//! - `asr-qwen3`, `tts-qwen3`, `ocr`, `ocr-vl`, and `llm` select built-in implementations.
//! - `download-all` enables Hugging Face and `ModelScope` provisioning.
//! - `cpu`, `metal`, and `cuda` select hardware support for enabled implementations.
//! - `server-support` and `llm-test-support` are intended for Orchion Server integration, not
//!   ordinary SDK callers.
//!
//! # Runtime behavior
//!
//! ASR, TTS, and OCR loading and inference use asynchronous interfaces. LLM provisioning and
//! [`LlmEngine::load_deployment`] are asynchronous, while [`LlmEngine::load`] and
//! [`LlmEngine::load_deployment_blocking`] explicitly block until the native runtime is ready.
//! Dropping an asynchronous native-load future cannot cancel native work that has already started.

#![allow(clippy::missing_errors_doc, clippy::must_use_candidate)]

#[cfg(feature = "asr-engine")]
pub mod asr;

#[cfg(feature = "audio-vad")]
pub mod audio_vad;

#[cfg(feature = "tts-engine")]
pub mod tts;

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
pub mod ocr;

#[cfg(feature = "llm")]
mod llm;

pub use orchion_core::{
    ASR_SAMPLE_RATE, AsrModel, AsrOptions, AsrSegment, AsrStreamingOptions,
    AsrTimestampGranularity, AsrTranscript, DevicePreference, DownloadRetryability, KnownOcrModel,
    LlmModel, ModelCapabilities, ModelCapabilityRequirement, ModelCategory, ModelDescriptor,
    ModelId, ModelSourceLocators, ModelSpec, ModelUrl, ModelUrlSource, OcrLayoutBlock, OcrLimits,
    OcrModel, OcrModelAsset, OcrModelAssetKind, OcrModelAssetRole, OcrModelKind, OcrOptions,
    OcrPoint, OcrRegion, OcrResponseFormat, OcrResult, OcrTask, OcrUsage, OrchionError,
    ParseModelUrlError, Result, RuntimeProvider, TtsAudio, TtsLanguage, TtsModel, TtsOptions,
    TtsSpeaker, TtsVoice, ensure_voice_supported, model_descriptor, prepare_asr_samples,
    registered_model_descriptors,
};

#[cfg(feature = "audio-ffmpeg")]
pub use orchion_audio::{
    AudioInputFormat, AudioOutputFormat, DecodedAudio, EncodedAudio, FfmpegAudioCodec,
    StreamingAudioDecoder, decode_audio_bytes, decode_audio_bytes_with_max_samples,
    decode_audio_file, decode_audio_file_with_max_samples, decode_pcm_s16le_bytes,
    encode_tts_audio,
};

#[cfg(feature = "audio-vad")]
pub use audio_vad::{
    AudioVadConfig, AudioVadMode, AudioVadSegment, AudioVadSegmenter, AudioVadStreamingConfig,
    AudioVadStreamingEndpoint, AudioVadStreamingEvent,
};

#[cfg(feature = "download-all")]
pub use orchion_download::{
    ArtifactRequest, ArtifactRole, DeploymentArtifactPlan, DeploymentArtifactRequest,
    DeploymentArtifactSource, DeploymentPublication, DeploymentSourcePlan, DownloadSource,
    ModelDownloader, PublishedDeploymentArtifact,
};

#[cfg(feature = "docs")]
pub use orchion_docs as docs;

#[cfg(feature = "asr-engine")]
pub use asr::{Asr, AsrEngine, AsrEngineFuture, AsrStream, AsrStreamSession};

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
pub use ocr::{Ocr, OcrAssets, OcrDeployment, OcrEngine, OcrEngineFuture};
#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
pub use orchion_ocr::TableStructureAssets;

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
pub use orchion_ocr::validate_image_file as validate_ocr_image_file;

#[cfg(feature = "llm")]
pub use llm::{
    GenerationEvent, GenerationFinishReason, GenerationOptions, GenerationRequest,
    LlmAdvancedInput, LlmAdvancedRequest, LlmChoiceEvent, LlmChoiceFinishReason,
    LlmChoiceGeneration, LlmChoiceReservation, LlmComplete, LlmContentPart, LlmDeployment,
    LlmDeploymentKind, LlmEmbeddingConfig, LlmEmbeddingInput, LlmEmbeddingPooling,
    LlmEmbeddingRequest, LlmEmbeddingResult, LlmEngine, LlmEngineConfig, LlmGeneration,
    LlmImageFormat, LlmImageInput, LlmLogitBias, LlmLogprobsOptions, LlmMessage,
    LlmOutputConstraint, LlmPromptCacheConfig, LlmReasoningControl, LlmReasoningControlResult,
    LlmReasoningEffort, LlmReasoningOptions, LlmRichMessage, LlmRole, LlmSamplingExtensions,
    LlmSemanticDelta, LlmSemanticRole, LlmSemanticTokenCountRequest, LlmTemplateEngine, LlmTimings,
    LlmTokenAlternative, LlmTokenLogprobs, LlmToolCall, LlmToolChoice, LlmToolDefinition,
    LlmToolResult, LlmUsage, LlmVisionConfig, LlmVisionLimits, validate_llm_json_schema,
};

#[cfg(feature = "server-support")]
pub mod server_support {
    pub use crate::llm::{
        LlmBackendGuard, LlmEmbeddingOperation, LlmEmbeddingReservation, LlmReservation,
        initialize_llm_backend, llm_build_metadata_json,
    };
}

#[cfg(feature = "llm-test-support")]
pub mod llm_test_support {
    pub use crate::llm::{
        LlmScriptedControl, scripted_context_limit_llm_engine, scripted_embedding_llm_engine,
        scripted_failing_llm_engine, scripted_llm_engine, scripted_panicking_llm_engine,
        scripted_preparation_panicking_llm_engine, scripted_reasoning_llm_engine,
        scripted_slow_preparation_llm_engine,
    };
}

#[cfg(feature = "tts-engine")]
pub use tts::{Tts, TtsEngine, TtsEngineFuture};
