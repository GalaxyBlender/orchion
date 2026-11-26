//! Opt-in real-model tests. See `docs/model-unload-testing.md`.
#![cfg(all(feature = "cuda", target_os = "linux"))]

use candle_core::cuda::cudarc::driver::CudaContext;
use candle_core::cuda::cudarc::driver::{result, sys};
use orchion_core::DevicePreference;
use std::path::PathBuf;
use std::process::Command;
use sys::CUmemPool_attribute_enum::{
    self, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
    CU_MEMPOOL_ATTR_USED_MEM_HIGH,
};

const RETAINED_ALLOWANCE_MIB: u64 = 256;
const MIB: u64 = 1024 * 1024;

struct Workspace<'ctx> {
    _context: &'ctx CudaContext,
    pool: sys::CUmemoryPool,
    resident_used: u64,
    resident_reserved: u64,
}

#[allow(unsafe_code)]
impl<'ctx> Workspace<'ctx> {
    fn after_load(context: &'ctx CudaContext) -> Self {
        context.synchronize().unwrap();
        // SAFETY: the CUDA device belongs to the borrowed live context.
        let pool = unsafe { result::device::get_mem_pool(context.cu_device()) }.unwrap();
        let mut workspace = Self {
            _context: context,
            pool,
            resident_used: 0,
            resident_reserved: 0,
        };
        workspace.resident_used = workspace.read(CU_MEMPOOL_ATTR_USED_MEM_CURRENT);
        workspace.resident_reserved = workspace.read(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
        let mut zero = 0_u64;
        // SAFETY: CUDA accepts a pointer to a uint64_t zero to reset this counter.
        unsafe {
            result::mem_pool::set_attribute(
                pool,
                CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                (&raw mut zero).cast(),
            )
        }
        .unwrap();
        workspace
    }

    fn read(&self, attribute: CUmemPool_attribute_enum) -> u64 {
        let mut bytes = 0_u64;
        // SAFETY: the three memory attributes read here use writable uint64_t storage.
        unsafe { result::mem_pool::get_attribute(self.pool, attribute, (&raw mut bytes).cast()) }
            .unwrap();
        bytes
    }

    fn assert_reclaimed(&self, max_workspace_mib: u64) {
        let peak = self.read(CU_MEMPOOL_ATTR_USED_MEM_HIGH);
        let reserved = self.read(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
        println!(
            "resident={} MiB, peak={} MiB, reserved after return={} MiB",
            self.resident_used / MIB,
            peak / MIB,
            reserved / MIB
        );
        assert!(
            peak <= self.resident_used + max_workspace_mib * MIB,
            "long inference exceeded the CUDA workspace budget"
        );
        assert_eq!(
            self.read(CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
            self.resident_used
        );
        assert!(
            reserved <= self.resident_reserved + RETAINED_ALLOWANCE_MIB * MIB,
            "temporary allocations stayed reserved after the request"
        );
    }
}

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
                assert!(!audio.samples.is_empty(), "synthesis must produce audio");
            }
            engine
        })
    });
}

#[cfg(feature = "tts")]
#[test]
#[ignore = "requires an isolated CUDA GPU and local Qwen3-TTS 0.6B CustomVoice weights"]
fn long_tts_bounds_workspace_and_releases_it_without_unloading_the_model() {
    use orchion_core::{TtsLanguage, TtsModel, TtsOptions, TtsSpeaker, TtsVoice};
    use orchion_qwen3::Tts;

    let ordinal = std::env::var("ORCHION_TEST_CUDA_ORDINAL").map_or(0, |n| n.parse().unwrap());
    let context = CudaContext::new(ordinal).unwrap();
    assert!(context.has_async_alloc());
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let model = TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-customvoice").unwrap();
    let engine = rt
        .block_on(Tts::load_with_device(
            model,
            model_dir("ORCHION_TTS_TEST_MODEL_DIR"),
            DevicePreference::Cuda(Some(ordinal)),
        ))
        .unwrap();
    let workspace = Workspace::after_load(&context);

    let audio = rt
        .block_on(engine.synthesize_with(
            "清晨的河岸很安静，薄雾从水面慢慢升起。沿着石板路往前走，可以看见几棵老柳树。\
             附近的面包店刚刚开门，烤面包的香气飘到街角。跑步的人放慢脚步，互相打个招呼。\
             等太阳越过屋顶，整条街便渐渐热闹起来，迎来了崭新的一天。",
            TtsVoice::Preset {
                speaker: TtsSpeaker::Vivian,
                language: TtsLanguage::Chinese,
            },
            TtsOptions {
                seed: Some(42),
                ..TtsOptions::default()
            },
        ))
        .unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    assert!(audio.samples.len() >= 20 * 24_000);
    assert!(audio.samples.iter().all(|x| x.is_finite()));

    workspace.assert_reclaimed(1024);
}

#[cfg(feature = "asr")]
#[test]
#[ignore = "requires an isolated CUDA GPU and local Qwen3-ASR 0.6B weights"]
fn long_asr_bounds_workspace_and_releases_it_without_unloading_the_model() {
    use orchion_core::{ASR_SAMPLE_RATE, AsrModel, AsrOptions};
    use orchion_qwen3::Asr;

    let ordinal = std::env::var("ORCHION_TEST_CUDA_ORDINAL").map_or(0, |n| n.parse().unwrap());
    let context = CudaContext::new(ordinal).unwrap();
    assert!(context.has_async_alloc());
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let model = AsrModel::parse("alibaba/qwen3-asr-0.6b").unwrap();
    let engine = rt
        .block_on(Asr::load_with_device(
            model,
            model_dir("ORCHION_ASR_TEST_MODEL_DIR"),
            DevicePreference::Cuda(Some(ordinal)),
        ))
        .unwrap();
    let workspace = Workspace::after_load(&context);

    // A non-round duration exercises the padded tail in the final convolution batch.
    let samples = vec![0.01; 4 * 60 * ASR_SAMPLE_RATE as usize + 8_000];
    rt.block_on(engine.transcribe_samples_with(
        &samples,
        ASR_SAMPLE_RATE,
        AsrOptions {
            max_new_tokens: 8,
            ..AsrOptions::default()
        },
    ))
    .unwrap();
    workspace.assert_reclaimed(1600);
}
