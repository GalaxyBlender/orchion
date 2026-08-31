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
pub use ocr::{Ocr, OcrAssets, OcrEngine, OcrEngineFuture};
#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
pub use orchion_ocr::TableStructureAssets;

#[cfg(any(feature = "ocr", feature = "ocr-vl"))]
pub use orchion_ocr::validate_image_file as validate_ocr_image_file;

#[cfg(feature = "llm")]
pub use llm::{
    GenerationEvent, GenerationFinishReason, GenerationOptions, GenerationRequest, LlmBackendGuard,
    LlmComplete, LlmEngine, LlmEngineConfig, LlmGeneration, LlmMessage, LlmReservation, LlmRole,
    LlmScriptedControl, LlmTemplateEngine, LlmTimings, LlmUsage, initialize_llm_backend,
    llm_build_metadata_json, scripted_context_limit_llm_engine, scripted_llm_engine,
    scripted_panicking_llm_engine, scripted_preparation_panicking_llm_engine,
    scripted_slow_preparation_llm_engine,
};

#[cfg(feature = "tts-engine")]
pub use tts::{Tts, TtsEngine, TtsEngineFuture};
