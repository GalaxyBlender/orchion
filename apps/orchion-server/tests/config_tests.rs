use orchion::{
    DevicePreference, KnownOcrModel, ModelId, ModelUrl, ModelUrlSource, OcrResponseFormat,
};
use orchion_server::config::{
    ConfigError, LlmDeploymentKind, LlmEmbeddingPooling, ModelSource, OcrModelDeployment,
    ServerConfig, TableStructureConfig, TableStructureType,
};
use orchion_server::state::AppState;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::Duration;

fn exe() -> &'static Path {
    Path::new("/tmp/orchion-server")
}

#[test]
fn defaults_are_executable_relative_and_use_neutral_deployments() {
    let exe_path = Path::new("/tmp/orchion/bin/orchion-server");
    let config = ServerConfig::default_for_exe(exe_path);

    assert_eq!(
        config.server.bind,
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9090)
    );
    assert_eq!(
        config.config_path,
        exe_path.parent().unwrap().join("config.toml")
    );
    assert_eq!(config.models.dir, exe_path.parent().unwrap().join("models"));
    assert_eq!(config.models.source, ModelSource::Auto);
    assert_eq!(config.services.asr.models.len(), 1);
    assert_eq!(
        config.services.asr.models[0].id.as_str(),
        "alibaba/qwen3-asr-0.6b"
    );
    assert_eq!(
        config.services.asr.models[0].model.source(),
        ModelUrlSource::Neutral
    );
    assert_eq!(
        config.services.tts.models[0].id.as_str(),
        "alibaba/qwen3-tts-12hz-0.6b-customvoice"
    );
    assert!(config.services.ocr.models.is_empty());
    assert!(config.services.ocr_vl.models.is_empty());
    assert_eq!(config.services.ocr.format, OcrResponseFormat::Json);
    assert_eq!(config.services.ocr_vl.format, OcrResponseFormat::Markdown);
}

#[test]
fn deployment_arrays_parse_and_replace_defaults() {
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
default_model = "alibaba/qwen3-asr-1.7b"

[[services.asr.models]]
id = "alibaba/qwen3-asr-1.7b"
name = "  Qwen3-ASR 1.7B  "
model = "ms://Qwen/Qwen3-ASR-1.7B"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(config.services.asr.models.len(), 1);
    let deployment = &config.services.asr.models[0];
    assert_eq!(deployment.id.as_str(), "alibaba/qwen3-asr-1.7b");
    assert_eq!(deployment.runtime.as_str(), deployment.id.as_str());
    assert_eq!(deployment.name.as_deref(), Some("Qwen3-ASR 1.7B"));
    assert_eq!(deployment.display_name(), "Qwen3-ASR 1.7B");
    assert_eq!(deployment.model.as_str(), "ms://Qwen/Qwen3-ASR-1.7B");
    assert_eq!(deployment.model.source(), ModelUrlSource::ModelScope);
}

#[test]
fn omitted_name_falls_back_to_model_id_name_segment() {
    let config = ServerConfig::from_toml_str(
        r#"
[[services.tts.models]]
id = "alibaba/qwen3-tts-12hz-0.6b-customvoice"
model = "//Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(
        config.services.tts.models[0].display_name(),
        "qwen3-tts-12hz-0.6b-customvoice"
    );
}

#[test]
fn blank_names_are_rejected() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
name = "   "
model = "//Qwen/Qwen3-ASR-0.6B"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::InvalidModelName { .. }));
}

#[test]
fn enabled_services_require_nonempty_models() {
    for section in ["asr", "tts", "ocr", "ocr-vl"] {
        let document = format!("[services.{section}]\nenabled = true\nmodels = []");
        let error = ServerConfig::from_toml_str(&document, exe()).unwrap_err();
        assert!(matches!(
            error,
            ConfigError::ServiceEnabledWithoutModels { .. }
        ));
    }
}

#[test]
fn defaults_must_match_exactly_one_local_deployment_even_when_disabled() {
    let error = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = false
default_model = "alibaba/qwen3-asr-1.7b"

[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::DefaultModelUnavailable { .. }));
    assert!(error.to_string().contains("services.asr.models"));
}

#[test]
fn model_ids_are_globally_unique_including_disabled_services() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B"

[[services.tts.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::DuplicateModelId { .. }));

    let error = ServerConfig::from_toml_str(
        r#"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"

[[services.ocr-vl.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::DuplicateModelId { .. }));
}

#[test]
fn duplicate_ids_within_one_disabled_service_are_rejected() {
    let error = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = false

[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "hf://PaddlePaddle/PP-OCRv6_tiny"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::DuplicateModelId { .. }));
}

#[test]
fn configured_ids_must_match_the_service_runtime_category() {
    let error = ServerConfig::from_toml_str(
        r#"
[services.asr]
default_model = "paddlepaddle/paddleocr-vl-1.6"
[[services.asr.models]]
id = "paddlepaddle/paddleocr-vl-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::UnsupportedRuntimeModel { .. }));

    let error = ServerConfig::from_toml_str(
        r#"
[[services.ocr.models]]
id = "paddlepaddle/paddleocr-vl-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::UnsupportedRuntimeModel { .. }));
}

#[test]
fn unregistered_speech_ids_remain_valid_runtime_models() {
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
default_model = "Acme/Private-ASR"
[[services.asr.models]]
id = "Acme/Private-ASR"
model = "hf://Acme/Private-ASR-Package"

[services.tts]
default_model = "Acme/Private-TTS"
[[services.tts.models]]
id = "Acme/Private-TTS"
model = "ms://Acme/Private-TTS-Package"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(
        config.services.asr.models[0].runtime.as_str(),
        "Acme/Private-ASR"
    );
    assert_eq!(
        config.services.tts.models[0].runtime.as_str(),
        "Acme/Private-TTS"
    );
}

#[test]
fn repository_runtime_rejects_exact_file_model_locator() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B/model.safetensors"
"#,
        exe(),
    )
    .unwrap_err();

    assert!(matches!(error, ConfigError::UnsupportedModelLocator { .. }));
}

#[test]
fn ocr_deployment_accepts_supported_layout_recipe_and_projects_runtime_id() {
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"

[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "hf://PaddlePaddle/PP-OCRv6_tiny"
layout_model = "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        exe(),
    )
    .unwrap();

    let deployment = &config.services.ocr.models[0];
    assert_eq!(deployment.model.source(), ModelUrlSource::HuggingFace);
    assert_eq!(
        deployment.layout_model.as_ref().unwrap().as_str(),
        "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
    );
    assert_eq!(
        config.services.ocr.layout_ids(),
        [ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap()]
    );
    assert_eq!(
        config
            .services
            .ocr
            .default_layout_runtime()
            .unwrap()
            .id()
            .as_str(),
        "PaddlePaddle/PP-DocLayoutV3"
    );
}

#[test]
fn unsupported_layout_recipe_is_rejected() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.ocr-vl.models]]
id = "paddlepaddle/paddleocr-vl-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
layout_model = "//Acme/Layout/model.onnx"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::UnsupportedLayoutModel { .. }));
}

#[test]
fn local_layout_artifact_uses_the_supported_layout_runtime_recipe() {
    let config = ServerConfig::from_toml_str(
        r#"
[[services.ocr-vl.models]]
id = "paddlepaddle/paddleocr-vl-1.6"
model = "//PaddlePaddle/PaddleOCR-VL-1.6"
layout_model = "file:///tmp/layout.onnx"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(
        config.services.ocr_vl.models[0]
            .layout_runtime
            .as_ref()
            .unwrap()
            .id()
            .as_str(),
        "PaddlePaddle/PP-DocLayoutV3"
    );
}

#[test]
fn separate_ocr_deployments_accept_distinct_layout_locators() {
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "paddlepaddle/pp-ocrv6-tiny"

[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "hf://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"

[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-small"
model = "//PaddlePaddle/PP-OCRv6_small"
layout_model = "ms://PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        exe(),
    )
    .unwrap();

    assert_eq!(
        config.services.ocr.models[0]
            .layout_model
            .as_ref()
            .unwrap()
            .source(),
        ModelUrlSource::HuggingFace
    );
    assert_eq!(
        config.services.ocr.models[1]
            .layout_model
            .as_ref()
            .unwrap()
            .source(),
        ModelUrlSource::ModelScope
    );
}

#[test]
fn layout_model_is_rejected_for_speech_deployments() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "//Qwen/Qwen3-ASR-0.6B"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(matches!(error, ConfigError::ParseToml(_)));
    assert!(error.to_string().contains("unknown field"));
}

fn table_config(extra: &str) -> String {
    format!(
        r#"
[[services.ocr.models]]
id = "paddlepaddle/pp-ocrv6-tiny"
model = "//PaddlePaddle/PP-OCRv6_tiny"
layout_model = "//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx"
table_structure = {{ model = "//Acme/Table/table.onnx", dictionary = "//Acme/Table/table_dict.txt", table_type = "wired"{extra} }}
"#
    )
}

fn programmatic_table_config(score_threshold: f32, max_structure_length: usize) -> ServerConfig {
    let mut config = ServerConfig::default_for_exe(exe());
    let mut deployment = OcrModelDeployment::from_runtime(KnownOcrModel::PpOcrV6Tiny.into_model())
        .with_supported_layout();
    deployment.table_structure = Some(TableStructureConfig {
        model: ModelUrl::parse("//Acme/Table/table.onnx").unwrap(),
        dictionary: ModelUrl::parse("//Acme/Table/table_dict.txt").unwrap(),
        table_type: TableStructureType::Wired,
        score_threshold,
        max_structure_length,
    });
    config.services.ocr.enabled = true;
    config.services.ocr.default_model = Some(deployment.id.clone());
    config.services.ocr.models = vec![deployment];
    config
}

#[test]
fn table_structure_defaults_and_overrides_parse() {
    let config = ServerConfig::from_toml_str(&table_config(""), exe()).unwrap();
    let table = config.services.ocr.models[0]
        .table_structure
        .as_ref()
        .unwrap();
    assert_eq!(table.table_type.as_str(), "wired");
    assert!((table.score_threshold - 0.5).abs() <= f32::EPSILON);
    assert_eq!(table.max_structure_length, 500);

    let config = ServerConfig::from_toml_str(
        &table_config(", score_threshold = 0.75, max_structure_length = 768"),
        exe(),
    )
    .unwrap();
    let table = config.services.ocr.models[0]
        .table_structure
        .as_ref()
        .unwrap();
    assert!((table.score_threshold - 0.75).abs() <= f32::EPSILON);
    assert_eq!(table.max_structure_length, 768);
}

#[test]
fn table_structure_accepts_wired_and_wireless_only() {
    for table_type in ["wired", "wireless"] {
        let document = table_config("").replace(
            "table_type = \"wired\"",
            &format!("table_type = \"{table_type}\""),
        );
        let config = ServerConfig::from_toml_str(&document, exe()).unwrap();
        assert_eq!(
            config.services.ocr.models[0]
                .table_structure
                .as_ref()
                .unwrap()
                .table_type
                .as_str(),
            table_type
        );
    }
    let error = ServerConfig::from_toml_str(
        &table_config("").replace("table_type = \"wired\"", "table_type = \"auto\""),
        exe(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ConfigError::InvalidTableStructureType { .. }
    ));
}

#[test]
fn table_structure_rejects_unknown_fields_and_missing_layout() {
    let error = ServerConfig::from_toml_str(&table_config(", input_shape = [488, 488]"), exe())
        .unwrap_err();
    assert!(error.to_string().contains("unknown field"));

    let document = table_config("").replace(
        "layout_model = \"//PaddlePaddle/PP-DocLayoutV3_onnx/inference.onnx\"\n",
        "",
    );
    let error = ServerConfig::from_toml_str(&document, exe()).unwrap_err();
    assert!(matches!(
        error,
        ConfigError::TableStructureRequiresLayout { .. }
    ));
}

#[test]
fn ocr_vl_rejects_table_structure() {
    let document = table_config("")
        .replace("services.ocr.models", "services.ocr-vl.models")
        .replace(
            "paddlepaddle/pp-ocrv6-tiny",
            "paddlepaddle/paddleocr-vl-1.6",
        )
        .replace(
            "//PaddlePaddle/PP-OCRv6_tiny",
            "//PaddlePaddle/PaddleOCR-VL-1.6",
        );
    let error = ServerConfig::from_toml_str(&document, exe()).unwrap_err();
    assert!(matches!(
        error,
        ConfigError::UnsupportedTableStructurePipeline { .. }
    ));
}

#[test]
fn table_structure_rejects_invalid_thresholds_and_zero_length() {
    for value in ["nan", "inf", "-inf", "-0.01", "1.01"] {
        let error = ServerConfig::from_toml_str(
            &table_config(&format!(", score_threshold = {value}")),
            exe(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ConfigError::InvalidTableStructureThreshold { .. }
        ));
    }
    let error = ServerConfig::from_toml_str(&table_config(", max_structure_length = 0"), exe())
        .unwrap_err();
    assert!(matches!(
        error,
        ConfigError::InvalidTableStructureLength { .. }
    ));
}

#[test]
fn programmatic_table_structure_rejects_invalid_thresholds_in_validate_and_app_state() {
    for score_threshold in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
        let config = programmatic_table_config(score_threshold, 500);
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigError::InvalidTableStructureThreshold {
                section: "services.ocr",
                ..
            }
        ));
        let Err(error) = AppState::from_prepared_config(config) else {
            panic!("invalid programmatic table threshold should reject AppState");
        };
        assert!(matches!(
            error.downcast_ref::<ConfigError>(),
            Some(ConfigError::InvalidTableStructureThreshold {
                section: "services.ocr",
                ..
            })
        ));
    }
}

#[test]
fn programmatic_table_structure_rejects_zero_length_in_validate_and_app_state() {
    let config = programmatic_table_config(0.5, 0);
    assert!(matches!(
        config.validate().unwrap_err(),
        ConfigError::InvalidTableStructureLength {
            section: "services.ocr"
        }
    ));
    let Err(error) = AppState::from_prepared_config(config) else {
        panic!("zero programmatic table length should reject AppState");
    };
    assert!(matches!(
        error.downcast_ref::<ConfigError>(),
        Some(ConfigError::InvalidTableStructureLength {
            section: "services.ocr"
        })
    ));
}

#[test]
fn all_removed_config_fields_are_rejected_without_aliases() {
    for field in [
        "available_models = []",
        "layout_available_models = []",
        "layout_default_model = \"PaddlePaddle/PP-DocLayoutV3\"",
    ] {
        let section = if field.starts_with("available") {
            "asr"
        } else {
            "ocr"
        };
        let document = format!("[services.{section}]\n{field}");
        let error = ServerConfig::from_toml_str(&document, exe()).unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{field}");
    }
}

#[test]
fn model_url_validation_runs_during_toml_deserialization() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.asr.models]]
id = "alibaba/qwen3-asr-0.6b"
model = "https://huggingface.co/Qwen/Qwen3-ASR-0.6B"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("invalid model URL"));
}

#[test]
fn llm_embedding_deployment_parses_and_generation_remains_the_default() {
    let config = ServerConfig::from_toml_str(
        r#"
[[services.llm.models]]
id = "qwen/generator"
model = "//Qwen/generator/model.gguf"

[[services.llm.models]]
id = "qwen/qwen3-embedding-0.6b"
kind = "embedding"
model = "//Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"
runtime = { batch_size = 8192, micro_batch_size = 8192 }
embeddings = { pooling = "last", min_dimensions = 32, max_input_tokens = 8192 }
"#,
        exe(),
    )
    .unwrap();
    assert_eq!(
        config.services.llm.models[0].kind,
        LlmDeploymentKind::Generation
    );
    let LlmDeploymentKind::Embeddings(embedding) = config.services.llm.models[1].kind else {
        panic!("expected embedding deployment");
    };
    assert_eq!(embedding.pooling, LlmEmbeddingPooling::Last);
    assert_eq!(embedding.min_dimensions, 32);
    assert_eq!(embedding.max_input_tokens, 8192);
}

#[test]
fn generation_deployment_accepts_multiple_execution_slots() {
    let config = ServerConfig::from_toml_str(
        r#"
        [[services.llm.models]]
        id = "qwen/generator"
        model = "//Qwen/generator/model.gguf"
        runtime = { parallel_sequences = 3, batch_size = 8 }
        "#,
        exe(),
    )
    .unwrap();

    assert_eq!(config.services.llm.models[0].runtime.parallel_sequences, 3);
}

#[test]
fn generation_prompt_cache_parses_defaults_and_strict_limits() {
    let config = ServerConfig::from_toml_str(
        r#"
[[services.llm.models]]
id = "qwen/generator"
model = "//Qwen/generator/model.gguf"
runtime = { context_size = 512 }
prompt_cache = { enabled = true, max_entries = 8, max_bytes = 1048576, min_prefix_tokens = 16 }
"#,
        exe(),
    )
    .unwrap();
    let cache = &config.services.llm.models[0].prompt_cache;
    assert!(cache.enabled);
    assert_eq!(cache.max_entries, 8);
    assert_eq!(cache.max_bytes, 1_048_576);
    assert_eq!(cache.min_prefix_tokens, 16);

    for field in [
        "max_entries = 0",
        "max_entries = 65",
        "max_bytes = 0",
        "max_bytes = 4294967297",
        "min_prefix_tokens = 0",
        "min_prefix_tokens = 512",
    ] {
        let document = format!(
            r#"
[[services.llm.models]]
id = "qwen/generator"
model = "//Qwen/generator/model.gguf"
runtime = {{ context_size = 512 }}
prompt_cache = {{ enabled = true, {field} }}
"#
        );
        assert!(
            ServerConfig::from_toml_str(&document, exe()).is_err(),
            "{field}"
        );
    }
}

#[test]
fn embedding_deployment_rejects_prompt_cache_as_generation_only() {
    let error = ServerConfig::from_toml_str(
        r#"
[[services.llm.models]]
id = "qwen/embedder"
kind = "embedding"
model = "//Qwen/embedder/model.gguf"
runtime = { batch_size = 512, micro_batch_size = 512 }
embeddings = { max_input_tokens = 512 }
prompt_cache = { enabled = false }
"#,
        exe(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("prompt_cache"));
}

#[test]
fn embedding_settings_require_embedding_kind_and_reject_mmproj() {
    for document in [
        r#"
[[services.llm.models]]
id = "qwen/test"
model = "//Qwen/test/model.gguf"
embeddings = { pooling = "last" }
"#,
        r#"
[[services.llm.models]]
id = "qwen/test"
kind = "embedding"
model = "//Qwen/test/model.gguf"
mmproj_model = "//Qwen/test/mmproj.gguf"
runtime = { batch_size = 8192, micro_batch_size = 8192 }
"#,
    ] {
        assert!(ServerConfig::from_toml_str(document, exe()).is_err());
    }
}

#[test]
fn general_server_and_service_overrides_still_parse() {
    let config = ServerConfig::from_toml_str(
        r#"
[server]
bind = "0.0.0.0:9000"
max_upload_size = "64M"
max_concurrent_inference = 4

[models]
dir = "cache/models"
source = "modelscope"
max_loaded = 3

[services.asr]
device = "cuda0"
idle_timeout = "5m"
stream_chunk_size = 1.5

[services.tts]
format = "mp3"
max_length = 1024
"#,
        Path::new("/opt/orchion/orchion-server"),
    )
    .unwrap();

    assert_eq!(config.server.bind.port(), 9000);
    assert_eq!(config.server.max_upload_size, 64 * 1024 * 1024);
    assert_eq!(config.models.source, ModelSource::ModelScope);
    assert_eq!(config.services.asr.device, DevicePreference::Cuda(Some(0)));
    assert_eq!(config.services.asr.idle_timeout, Duration::from_mins(5));
    assert_eq!(config.services.tts.format, "mp3");
    assert_eq!(config.services.tts.max_length, 1024);
}

#[test]
fn max_upload_size_rejects_values_above_the_hard_body_limit() {
    let error = ServerConfig::from_toml_str(
        r#"
[server]
max_upload_size = "129M"
"#,
        exe(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("must not exceed"));
}

#[test]
fn vision_base64_upper_bound_must_fit_global_body_limit() {
    let error = ServerConfig::from_toml_str(
        r#"
[server]
max_upload_size = "1M"

[services.llm]
enabled = true
default_model = "qwen/vision"

[[services.llm.models]]
id = "qwen/vision"
model = "//Qwen/test/model.gguf"
mmproj_model = "//Qwen/test/mmproj.gguf"
vision = { max_bytes_per_image = 786432, max_total_bytes = 786432 }
"#,
        exe(),
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("vision.max_total_bytes"));
    assert!(message.contains("server.max_upload_size"));
}

#[test]
fn shipped_config_uses_canonical_public_identity_and_preserves_artifact_locators() {
    let document = include_str!("../config.toml");
    let config = ServerConfig::from_toml_str(document, exe()).unwrap();

    assert!(config.services.asr.enabled);
    assert_eq!(config.services.asr.models.len(), 2);
    assert_eq!(config.services.tts.models.len(), 5);
    assert_eq!(config.services.ocr.models.len(), 5);
    assert_eq!(config.services.ocr_vl.models.len(), 2);
    assert_eq!(config.services.ocr.layout_ids().len(), 1);
    assert_eq!(
        config
            .services
            .asr
            .models
            .iter()
            .map(|model| (model.id.as_str(), model.display_name()))
            .collect::<Vec<_>>(),
        [
            ("alibaba/qwen3-asr-0.6b", "Qwen3-ASR 0.6B"),
            ("alibaba/qwen3-asr-1.7b", "Qwen3-ASR 1.7B"),
        ]
    );
    assert_eq!(
        config
            .services
            .tts
            .models
            .iter()
            .map(|model| (model.id.as_str(), model.display_name()))
            .collect::<Vec<_>>(),
        [
            (
                "alibaba/qwen3-tts-12hz-0.6b-base",
                "Qwen3-TTS 12Hz 0.6B Base"
            ),
            (
                "alibaba/qwen3-tts-12hz-0.6b-customvoice",
                "Qwen3-TTS 12Hz 0.6B CustomVoice",
            ),
            (
                "alibaba/qwen3-tts-12hz-1.7b-base",
                "Qwen3-TTS 12Hz 1.7B Base"
            ),
            (
                "alibaba/qwen3-tts-12hz-1.7b-customvoice",
                "Qwen3-TTS 12Hz 1.7B CustomVoice",
            ),
            (
                "alibaba/qwen3-tts-12hz-1.7b-voicedesign",
                "Qwen3-TTS 12Hz 1.7B VoiceDesign",
            ),
        ]
    );
    assert_eq!(
        config
            .services
            .ocr
            .models
            .iter()
            .map(|model| (model.id.as_str(), model.display_name()))
            .collect::<Vec<_>>(),
        [
            ("paddlepaddle/pp-ocrv5-mobile", "PP-OCRv5 Mobile"),
            ("paddlepaddle/pp-ocrv5-server", "PP-OCRv5 Server"),
            ("paddlepaddle/pp-ocrv6-tiny", "PP-OCRv6 Tiny"),
            ("paddlepaddle/pp-ocrv6-small", "PP-OCRv6 Small"),
            ("paddlepaddle/pp-ocrv6-medium", "PP-OCRv6 Medium"),
        ]
    );
    assert_eq!(
        config
            .services
            .ocr_vl
            .models
            .iter()
            .map(|model| (model.id.as_str(), model.display_name()))
            .collect::<Vec<_>>(),
        [
            ("paddlepaddle/paddleocr-vl-1.5", "PaddleOCR-VL 1.5"),
            ("paddlepaddle/paddleocr-vl-1.6", "PaddleOCR-VL 1.6"),
        ]
    );
    assert_eq!(config.services.llm.models.len(), 2);
    let llm = &config.services.llm.models[0];
    assert_eq!(llm.id.as_str(), "alibaba/qwen3.5-0.8b");
    assert_eq!(llm.name.as_deref(), Some("Qwen3.5 0.8B"));
    assert_eq!(
        llm.model.as_str(),
        "//unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
    );
    assert_eq!(
        llm.mmproj_model.as_ref().map(ModelUrl::as_str),
        Some("//unsloth/Qwen3.5-0.8B-GGUF/mmproj-BF16.gguf")
    );
}

#[test]
fn malformed_limits_and_unknown_fields_remain_rejected() {
    assert!(ServerConfig::from_toml_str("[models]\nmax_loaded = 0", exe()).is_err());
    assert!(ServerConfig::from_toml_str("[services.tts]\nmax_length = 0", exe()).is_err());
    let error = ServerConfig::from_toml_str("[services.tts]\nvoice = \"ryan\"", exe()).unwrap_err();
    assert!(error.to_string().contains("unknown field"));
}
