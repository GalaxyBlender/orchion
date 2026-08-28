use orchion::{AsrModel, ModelSpec, OcrModel, TtsModel};
use orchion_server::config::ServerConfig;
use orchion_server::model_cache::{ModelCache, ModelProvisionFuture, ModelProvisioner};
use orchion_server::state::AppState;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
struct FakeProvisioner {
    calls: Mutex<Vec<String>>,
    failures: Mutex<HashSet<String>>,
    active: AtomicUsize,
    max_active: AtomicUsize,
    delay: Mutex<Option<Duration>>,
}

impl FakeProvisioner {
    fn failing(model: impl Into<String>) -> Self {
        Self {
            failures: Mutex::new(HashSet::from([model.into()])),
            ..Self::default()
        }
    }

    fn delayed(delay: Duration) -> Self {
        Self {
            delay: Mutex::new(Some(delay)),
            ..Self::default()
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    fn provision<M: ModelSpec>(&self, model: M, models_dir: PathBuf) -> ModelProvisionFuture<'_> {
        let id = model.huggingface_repo().to_string();
        self.calls.lock().unwrap().push(id.clone());
        let should_fail = self.failures.lock().unwrap().contains(&id);
        let delay = *self.delay.lock().unwrap();
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        let path = model.cache_path(models_dir);
        drop(model);

        Box::pin(async move {
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
            if should_fail {
                anyhow::bail!("injected download failure for {id}");
            }
            Ok(path)
        })
    }
}

impl ModelProvisioner<AsrModel> for FakeProvisioner {
    fn provision(&self, model: AsrModel, models_dir: PathBuf) -> ModelProvisionFuture<'_> {
        self.provision(model, models_dir)
    }
}

impl ModelProvisioner<TtsModel> for FakeProvisioner {
    fn provision(&self, model: TtsModel, models_dir: PathBuf) -> ModelProvisionFuture<'_> {
        self.provision(model, models_dir)
    }
}

impl ModelProvisioner<OcrModel> for FakeProvisioner {
    fn provision(&self, model: OcrModel, models_dir: PathBuf) -> ModelProvisionFuture<'_> {
        self.provision(model, models_dir)
    }
}

#[tokio::test]
async fn available_model_download_failure_blocks_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true
default_model = "Qwen/Qwen3-ASR-0.6B"
available_models = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::failing("Qwen/Qwen3-ASR-1.7B"));

    let Err(error) = AppState::load_with_provisioner(config, Arc::clone(&provisioner)).await else {
        panic!("available model failure should block startup");
    };

    let calls = provisioner.calls();
    assert!(calls.contains(&"Qwen/Qwen3-ASR-0.6B".to_string()));
    assert!(calls.contains(&"Qwen/Qwen3-ASR-1.7B".to_string()));
    assert!(format!("{error:#}").contains("injected download failure for Qwen/Qwen3-ASR-1.7B"));
}

#[tokio::test]
async fn all_available_ocr_models_are_provisioned_at_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
available_models = ["PaddlePaddle/PP-OCRv6_tiny", "PaddlePaddle/PP-OCRv6_small"]
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::default());

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    let calls = provisioner.calls();
    assert!(calls.contains(&"PaddlePaddle/PP-OCRv6_tiny".to_string()));
    assert!(calls.contains(&"PaddlePaddle/PP-OCRv6_small".to_string()));
}

#[tokio::test]
async fn default_model_download_failure_still_blocks_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true
default_model = "Qwen/Qwen3-ASR-0.6B"
available_models = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::failing("Qwen/Qwen3-ASR-0.6B"));

    let Err(error) = AppState::load_with_provisioner(config, provisioner).await else {
        panic!("default model failure should block startup");
    };

    assert!(
        format!("{error:#}").contains("injected download failure for Qwen/Qwen3-ASR-0.6B"),
        "unexpected error: {error:#}"
    );
}

#[tokio::test]
async fn required_default_layout_download_failure_blocks_startup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_tiny"
available_models = ["PaddlePaddle/PP-OCRv6_tiny"]
layout_default_model = "PaddlePaddle/PP-DocLayoutV3"
layout_available_models = ["PaddlePaddle/PP-DocLayoutV3"]
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::failing("PaddlePaddle/PP-DocLayoutV3"));

    let Err(error) = AppState::load_with_provisioner(config, provisioner).await else {
        panic!("required default layout failure should block startup");
    };

    assert!(
        format!("{error:#}").contains("injected download failure for PaddlePaddle/PP-DocLayoutV3"),
        "unexpected error: {error:#}"
    );
}

#[tokio::test]
async fn non_default_model_is_provisioned_lazily_with_a_single_flight() {
    let temp_dir = tempfile::tempdir().unwrap();
    let default = AsrModel::parse("Qwen/Qwen3-ASR-0.6B").unwrap();
    let optional = AsrModel::parse("Qwen/Qwen3-ASR-1.7B").unwrap();
    let provisioner = Arc::new(FakeProvisioner::delayed(Duration::from_millis(20)));
    let cache = ModelCache::new_with_provisioner(
        "asr",
        vec![default, optional.clone()],
        Duration::from_mins(1),
        1,
        temp_dir.path().to_path_buf(),
        Arc::clone(&provisioner),
    );
    let engine_loads = Arc::new(AtomicUsize::new(0));

    let first = tokio::spawn({
        let cache = cache.clone();
        let optional = optional.clone();
        let engine_loads = Arc::clone(&engine_loads);
        async move {
            cache
                .get_or_load(optional, move |_, path| async move {
                    assert!(path.ends_with("Qwen/Qwen3-ASR-1.7B"));
                    Ok(engine_loads.fetch_add(1, Ordering::SeqCst) + 1)
                })
                .await
        }
    });
    let second = tokio::spawn({
        let cache = cache.clone();
        let engine_loads = Arc::clone(&engine_loads);
        async move {
            cache
                .get_or_load(optional, move |_, _| async move {
                    Ok(engine_loads.fetch_add(1, Ordering::SeqCst) + 1)
                })
                .await
        }
    });

    assert_eq!(*first.await.unwrap().unwrap().unwrap(), 1);
    assert_eq!(*second.await.unwrap().unwrap().unwrap(), 1);
    assert_eq!(provisioner.calls(), ["Qwen/Qwen3-ASR-1.7B"]);
    assert_eq!(engine_loads.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn startup_model_provisioning_has_bounded_concurrency() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ServerConfig::from_toml_str(
        r#"
[services.asr]
enabled = true

[services.tts]
enabled = true

[services.ocr]
enabled = true
default_model = "PaddlePaddle/PP-OCRv6_tiny"
available_models = ["PaddlePaddle/PP-OCRv6_tiny"]
layout_default_model = "PaddlePaddle/PP-DocLayoutV3"
layout_available_models = ["PaddlePaddle/PP-DocLayoutV3"]

[services.ocr-vl]
enabled = true
default_model = "PaddlePaddle/PaddleOCR-VL-1.6"
available_models = ["PaddlePaddle/PaddleOCR-VL-1.6"]
layout_default_model = "PaddlePaddle/PP-DocLayoutV3"
layout_available_models = ["PaddlePaddle/PP-DocLayoutV3"]
"#,
        &temp_dir.path().join("orchion-server"),
    )
    .unwrap();
    let provisioner = Arc::new(FakeProvisioner::delayed(Duration::from_millis(20)));

    AppState::load_with_provisioner(config, Arc::clone(&provisioner))
        .await
        .unwrap();

    let max_active = provisioner.max_active.load(Ordering::SeqCst);
    assert!(max_active > 1, "startup provisioning remained serial");
    assert!(
        max_active <= 2,
        "startup provisioning was not bounded: {max_active}"
    );
    assert_eq!(
        provisioner
            .calls()
            .iter()
            .filter(|model| model.as_str() == "PaddlePaddle/PP-DocLayoutV3")
            .count(),
        1
    );
}
