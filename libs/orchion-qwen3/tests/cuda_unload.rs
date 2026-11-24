//! Opt-in real-model tests. See `docs/model-unload-testing.md`.
#![cfg(all(feature = "cuda", target_os = "linux"))]

use candle_core::cuda::cudarc::driver::CudaContext;
use orchion_core::DevicePreference;
use std::path::PathBuf;
use std::process::Command;

const RETAINED_ALLOWANCE_MIB: u64 = 256;

fn process_gpu_mib() -> u64 {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("nvidia-smi is required");
    assert!(output.status.success(), "nvidia-smi failed: {output:?}");
    let pid = std::process::id().to_string();
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| line.split_once(','))
        .filter(|(process, _)| process.trim() == pid)
        .map(|(_, bytes)| bytes.trim().parse::<u64>().unwrap())
        .sum()
}

fn check_unload<T: Send + 'static>(
    load_and_infer: impl Fn(&tokio::runtime::Runtime, DevicePreference) -> T,
) {
    let ordinal = std::env::var("ORCHION_TEST_CUDA_ORDINAL").map_or(0, |n| n.parse().unwrap());
    let device = DevicePreference::Cuda(Some(ordinal));
    // Keep the primary context alive: destroying it would hide an incorrect free.
    let context = CudaContext::new(ordinal).unwrap();
    assert!(
        context.has_async_alloc(),
        "requires CUDA memory pool support"
    );
    let before_load = process_gpu_mib();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for round in 1..=3 {
        let engine = load_and_infer(&rt, device);
        let warm = process_gpu_mib();
        assert!(
            warm > before_load + RETAINED_ALLOWANCE_MIB,
            "test did not observe the model's CUDA allocations: {before_load} -> {warm} MiB"
        );
        // Idle cleanup can destroy the engine on a thread that never used CUDA.
        std::thread::spawn(move || drop(engine)).join().unwrap();
        let retired = process_gpu_mib();
        println!("round {round}: before={before_load}, loaded={warm}, unloaded={retired} MiB");
        assert!(
            retired <= before_load + RETAINED_ALLOWANCE_MIB,
            "round {round}: model allocations survived unload ({retired} MiB vs {before_load} MiB)"
        );
    }
    drop(context);
}

fn model_dir(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .unwrap_or_else(|| panic!("set {variable} to the model directory"))
        .into()
}

#[cfg(feature = "asr")]
#[test]
#[ignore = "requires a CUDA GPU and local Qwen3-ASR weights"]
fn asr_releases_gpu_memory_after_multiple_requests() {
    use orchion_core::{ASR_SAMPLE_RATE, AsrModel, AsrOptions};
    use orchion_qwen3::Asr;

    let path = model_dir("ORCHION_ASR_TEST_MODEL_DIR");
    let id =
        std::env::var("ORCHION_ASR_TEST_MODEL").unwrap_or_else(|_| "alibaba/qwen3-asr-0.6b".into());
    check_unload(|rt, device| {
        rt.block_on(async {
            let engine = Asr::load_with_device(AsrModel::parse(&id).unwrap(), &path, device)
                .await
                .unwrap();
            for seconds in [1, 2, 3] {
                let audio = vec![0.01; ASR_SAMPLE_RATE as usize * seconds];
                engine
                    .transcribe_samples_with(
                        &audio,
                        ASR_SAMPLE_RATE,
                        AsrOptions {
                            max_new_tokens: 8,
                            ..AsrOptions::default()
                        },
                    )
                    .await
                    .unwrap();
            }
            engine
        })
    });
}

#[cfg(feature = "tts")]
#[test]
#[ignore = "requires a CUDA GPU and local Qwen3-TTS 0.6B CustomVoice weights"]
fn tts_releases_gpu_memory_after_multiple_requests() {
    use orchion_core::{TtsLanguage, TtsModel, TtsOptions, TtsSpeaker, TtsVoice};
    use orchion_qwen3::Tts;

    let path = model_dir("ORCHION_TTS_TEST_MODEL_DIR");
    let model = TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-customvoice").unwrap();
    check_unload(|rt, device| {
        rt.block_on(async {
            let engine = Tts::load_with_device(model.clone(), &path, device)
                .await
                .unwrap();
            for text in ["Hello.", "This is a test.", "Can you hear me now?"] {
                let audio = engine
                    .synthesize_with(
                        text,
                        TtsVoice::Preset {
                            speaker: TtsSpeaker::Ryan,
                            language: TtsLanguage::English,
                        },
                        TtsOptions {
                            seed: Some(1),
                            max_length: 64,
                            ..TtsOptions::default()
                        },
                    )
                    .await
                    .unwrap();
                assert_ne!(audio.samples, [], "synthesis must produce audio");
            }
            engine
        })
    });
}
