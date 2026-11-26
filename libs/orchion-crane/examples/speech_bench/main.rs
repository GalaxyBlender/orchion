//! Standalone A/B benchmark; see docs/crane-performance.md.
mod codec;
mod memory;
mod streaming;

use anyhow::{Context, Result, bail, ensure};
use orchion_core::{
    ASR_SAMPLE_RATE, AsrModel, AsrOptions, AsrStreamingOptions, AsrTranscript, DevicePreference,
    TtsAudio, TtsLanguage, TtsModel, TtsSpeaker, TtsVoice,
};
use std::{env, path::Path, time::Instant};
use tokio::runtime::Runtime;

const REPETITIONS: usize = 3;
const ASR_TOKEN_LIMIT: usize = 512;
const TTS_FRAME_LIMIT: usize = 512;
const SHORT: &str = "清晨的河岸很安静，薄雾从水面慢慢升起。";
const LONG: &str = "清晨的河岸很安静，薄雾从水面慢慢升起。沿着石板路往前走，可以看见几棵老柳树。\
    附近的面包店刚刚开门，烤面包的香气飘到街角。跑步的人放慢脚步，互相打个招呼。\
    等太阳越过屋顶，整条街便渐渐热闹起来，迎来了崭新的一天。";
const REFERENCE_TEXT: &str = "The weather is pleasant today, and we are taking a walk.";
const CUSTOMVOICE_CASES: [(&str, &str); 2] = [("tts_short", SHORT), ("tts_long", LONG)];

#[derive(Clone, Copy)]
enum Backend {
    Legacy,
    Crane,
}

type Inference<'a, I, O> = Box<dyn FnMut(&I) -> Result<O> + 'a>;

fn main() -> Result<()> {
    let arguments = env::args().collect::<Vec<_>>();
    let [_, backend, device, workload, model_dir, rest @ ..] = arguments.as_slice() else {
        bail!(
            "usage: speech_bench <legacy|crane> <cpu|cuda:0> <workload> <model_dir> [workload_args...]\n\
             asr: <pcm> [forced_language, legacy only]; asr-stream (legacy): <pcm>\n\
             tts, tts-stream (crane), design (legacy): [output_dir]\n\
             clone: <reference_wav> [output_dir]; reference (legacy): <output_wav>\n\
             or: speech_bench crane <cpu|cuda:0> codec <model_dir> <batch|stream> <output_dir>"
        );
    };
    let backend = match backend.as_str() {
        "legacy" => Backend::Legacy,
        "crane" => Backend::Crane,
        _ => bail!("unknown backend: {backend}"),
    };
    let device: DevicePreference = device.parse()?;
    ensure!(
        matches!(device, DevicePreference::Cpu | DevicePreference::Cuda(_)),
        "benchmark requires an explicit CPU or CUDA device"
    );
    if env::var_os("ORCHION_ASR_PROFILE").is_some() || env::var_os("ORCHION_TTS_PROFILE").is_some()
    {
        tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(std::io::stderr)
            .try_init()
            .map_err(|error| anyhow::anyhow!("start profiling logger: {error}"))?;
    }
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let path = Path::new(model_dir);
    let memory = memory::Memory::new(device)?;
    println!("baseline {}", memory.snapshot()?);
    let result = match (workload.as_str(), rest) {
        ("asr", [audio]) => bench_asr(backend, &runtime, device, path, audio, None, &memory),
        ("asr", [audio, language]) if matches!(backend, Backend::Legacy) => bench_asr(
            backend,
            &runtime,
            device,
            path,
            audio,
            Some(language),
            &memory,
        ),
        ("asr-stream", [audio]) if matches!(backend, Backend::Legacy) => {
            bench_asr_stream(&runtime, device, path, audio, &memory)
        }
        ("tts", [] | [_]) => bench_tts(
            backend,
            &runtime,
            device,
            path,
            None,
            &memory,
            rest.first().map(Path::new),
        ),
        ("tts-stream", [] | [_]) if matches!(backend, Backend::Crane) => {
            bench_streaming_tts(device, path, &memory, rest.first().map(Path::new))
        }
        ("codec", [schedule, output]) if matches!(backend, Backend::Crane) => {
            codec::bench(device, path, schedule, Path::new(output), &memory)
        }
        ("clone", [audio] | [audio, _]) => bench_tts(
            backend,
            &runtime,
            device,
            path,
            Some(audio.as_str()),
            &memory,
            rest.get(1).map(Path::new),
        ),
        ("design", [] | [_]) if matches!(backend, Backend::Legacy) => {
            bench_design(&runtime, device, path, &memory, rest.first().map(Path::new))
        }
        ("reference", [output]) => make_reference(&runtime, device, path, output),
        _ => bail!("invalid workload or audio path"),
    };
    println!("after_drop {}", memory.snapshot()?);
    result
}

fn read_asr_pcm(audio: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(audio)?;
    ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(2),
        "expected 16 kHz mono s16le PCM"
    );
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| f32::from(i16::from_le_bytes(b)) / 32768.0)
        .collect())
}

fn bench_asr(
    backend: Backend,
    runtime: &Runtime,
    device: DevicePreference,
    path: &Path,
    audio: &str,
    language: Option<&str>,
    memory: &memory::Memory,
) -> Result<()> {
    let short = read_asr_pcm(audio)?;
    let long = short.repeat(6);
    let started = Instant::now();
    let mut run: Inference<'_, [f32], AsrTranscript> = match backend {
        Backend::Legacy => {
            let model = AsrModel::parse("alibaba/qwen3-asr-0.6b")?;
            let engine =
                runtime.block_on(orchion_qwen3::Asr::load_with_device(model, path, device))?;
            Box::new(move |samples| {
                Ok(runtime.block_on(engine.transcribe_samples_with(
                    samples,
                    ASR_SAMPLE_RATE,
                    AsrOptions {
                        max_new_tokens: ASR_TOKEN_LIMIT,
                        language: language.map(str::to_owned),
                    },
                ))?)
            })
        }
        Backend::Crane => {
            let mut engine = orchion_crane::Asr::load(path, device)?;
            Box::new(move |samples| {
                Ok(engine.transcribe(samples, ASR_SAMPLE_RATE, ASR_TOKEN_LIMIT)?)
            })
        }
    };
    println!(
        "load ms={:.1} {}",
        started.elapsed().as_secs_f64() * 1000.0,
        memory.snapshot()?
    );
    ensure!(!run(&short)?.text.is_empty(), "empty ASR warm-up");
    for (label, samples) in [("asr_short", &short), ("asr_long", &long)] {
        let mut expected = None;
        for repetition in 1..=REPETITIONS {
            memory.reset_peak()?;
            let started = Instant::now();
            let out = measured(&mut run, samples.as_slice(), label, repetition, memory)?;
            let elapsed = started.elapsed().as_secs_f64();
            ensure!(!out.text.is_empty(), "empty ASR result");
            let duration = seconds(samples.len(), ASR_SAMPLE_RATE);
            println!(
                "{label} repetition={repetition} ms={:.1} audio_s={duration:.3} rtf={:.4} words={} {}",
                elapsed * 1000.0,
                elapsed / duration,
                out.text.split_whitespace().count(),
                memory.snapshot()?
            );
            if let Some(expected) = &expected {
                ensure!(
                    &out == expected,
                    "{label} transcript changed between requests"
                );
            } else {
                println!(
                    "{label} text={:?} language={:?} raw_output={:?}",
                    out.text, out.language, out.raw_output
                );
                expected = Some(out);
            }
        }
    }
    Ok(())
}

fn bench_asr_stream(
    runtime: &Runtime,
    device: DevicePreference,
    path: &Path,
    audio: &str,
    memory: &memory::Memory,
) -> Result<()> {
    let samples = read_asr_pcm(audio)?;
    let started = Instant::now();
    let model = AsrModel::parse("alibaba/qwen3-asr-0.6b")?;
    let engine = runtime.block_on(orchion_qwen3::Asr::load_with_device(model, path, device))?;
    println!(
        "load ms={:.1} {}",
        started.elapsed().as_secs_f64() * 1000.0,
        memory.snapshot()?
    );
    let run = || {
        runtime.block_on(async {
            let started = Instant::now();
            let mut stream = engine
                .start_streaming_with(AsrStreamingOptions::default())
                .await?;
            let mut partials = Vec::new();
            let mut first_partial = None;
            for chunk in samples.chunks(ASR_SAMPLE_RATE as usize) {
                let partial = stream.feed(chunk, ASR_SAMPLE_RATE).await?;
                if partial.as_ref().is_some_and(|out| !out.text.is_empty()) {
                    first_partial.get_or_insert_with(|| started.elapsed());
                }
                partials.push(partial);
            }
            let final_transcript = stream.finish().await?;
            let elapsed = started.elapsed();
            let first_partial = first_partial.context("no nonempty ASR partial")?;
            ensure!(
                !final_transcript.text.is_empty(),
                "empty final ASR transcript"
            );
            Ok(((partials, final_transcript), elapsed, first_partial))
        })
    };
    let (expected, _, _) = run()?;
    for repetition in 1..=REPETITIONS {
        memory.reset_peak()?;
        let (out, elapsed, first) = run()?;
        ensure!(
            out == expected,
            "stream transcripts changed between requests"
        );
        println!(
            "asr_stream repetition={repetition} ms={:.1} first_partial_ms={:.1} audio_s={:.3} {}",
            elapsed.as_secs_f64() * 1000.0,
            first.as_secs_f64() * 1000.0,
            seconds(samples.len(), ASR_SAMPLE_RATE),
            memory.snapshot()?
        );
    }
    println!("asr_stream transcripts={expected:?}");
    Ok(())
}

fn bench_tts(
    backend: Backend,
    runtime: &Runtime,
    device: DevicePreference,
    path: &Path,
    reference: Option<&str>,
    memory: &memory::Memory,
    output: Option<&Path>,
) -> Result<()> {
    let reference = reference.map(read_reference).transpose()?;
    let started = Instant::now();
    let mut run: Inference<'_, str, TtsAudio> = match backend {
        Backend::Legacy => {
            let (model, voice) = if let Some((path, _, _)) = &reference {
                (
                    "alibaba/qwen3-tts-12hz-0.6b-base",
                    TtsVoice::Clone {
                        reference_audio: (*path).into(),
                        reference_text: REFERENCE_TEXT.into(),
                        language: TtsLanguage::Chinese,
                    },
                )
            } else {
                (
                    "alibaba/qwen3-tts-12hz-0.6b-customvoice",
                    TtsVoice::Preset {
                        speaker: TtsSpeaker::Vivian,
                        language: TtsLanguage::Chinese,
                    },
                )
            };
            load_legacy_tts(runtime, device, path, model, voice)?
        }
        Backend::Crane => {
            let mut engine = orchion_crane::Tts::load(path, device)?;
            let voice = if let Some((_, samples, rate)) = &reference {
                orchion_crane::Voice::Clone {
                    samples,
                    sample_rate: *rate,
                    text: REFERENCE_TEXT,
                }
            } else {
                orchion_crane::Voice::Preset("vivian")
            };
            Box::new(move |text| {
                Ok(engine.synthesize(
                    text,
                    "chinese",
                    voice,
                    orchion_crane::TtsOptions {
                        max_frames: TTS_FRAME_LIMIT,
                        ..Default::default()
                    },
                )?)
            })
        }
    };
    println!(
        "load ms={:.1} {}",
        started.elapsed().as_secs_f64() * 1000.0,
        memory.snapshot()?
    );
    check_audio(&run(SHORT)?)?;
    let cases: &[_] = if reference.is_some() {
        &[("tts_clone", SHORT)]
    } else {
        &CUSTOMVOICE_CASES
    };
    measure_tts(&mut run, cases, memory, output)
}

fn load_legacy_tts<'a>(
    runtime: &'a Runtime,
    device: DevicePreference,
    path: &Path,
    model: &str,
    voice: TtsVoice,
) -> Result<Inference<'a, str, TtsAudio>> {
    let engine = runtime.block_on(orchion_qwen3::Tts::load_with_device(
        TtsModel::parse(model)?,
        path,
        device,
    ))?;
    Ok(Box::new(move |text| {
        Ok(runtime.block_on(engine.synthesize_with(text, voice.clone(), legacy_tts_options()))?)
    }))
}

fn bench_design(
    runtime: &Runtime,
    device: DevicePreference,
    path: &Path,
    memory: &memory::Memory,
    output: Option<&Path>,
) -> Result<()> {
    let started = Instant::now();
    let mut run = load_legacy_tts(
        runtime,
        device,
        path,
        "alibaba/qwen3-tts-12hz-1.7b-voicedesign",
        TtsVoice::Design {
            prompt: "A calm adult Mandarin voice, speaking clearly and gently.".into(),
            language: TtsLanguage::Chinese,
        },
    )?;
    println!(
        "load ms={:.1} {}",
        started.elapsed().as_secs_f64() * 1000.0,
        memory.snapshot()?
    );
    check_audio(&run(SHORT)?)?;
    measure_tts(&mut run, &[("tts_design", SHORT)], memory, output)
}

fn bench_streaming_tts(
    device: DevicePreference,
    path: &Path,
    memory: &memory::Memory,
    output: Option<&Path>,
) -> Result<()> {
    let started = Instant::now();
    let mut engine = streaming::Preset::load(path, device)?;
    println!(
        "load ms={:.1} {}",
        started.elapsed().as_secs_f64() * 1000.0,
        memory.snapshot()?
    );
    let mut run: Inference<'_, str, TtsAudio> = Box::new(|text| engine.synthesize(text));
    check_audio(&run(SHORT)?)?;
    measure_tts(&mut run, &CUSTOMVOICE_CASES, memory, output)
}

fn measure_tts<I: ?Sized>(
    run: &mut Inference<'_, I, TtsAudio>,
    cases: &[(&str, &I)],
    memory: &memory::Memory,
    output: Option<&Path>,
) -> Result<()> {
    for &(label, text) in cases {
        for repetition in 1..=REPETITIONS {
            memory.reset_peak()?;
            let started = Instant::now();
            let out = measured(&mut *run, text, label, repetition, memory)?;
            let elapsed = started.elapsed().as_secs_f64();
            check_audio(&out)?;
            let duration = seconds(out.samples.len(), out.sample_rate);
            println!(
                "{label} repetition={repetition} ms={:.1} audio_s={duration:.3} rtf={:.4} {}",
                elapsed * 1000.0,
                elapsed / duration,
                memory.snapshot()?
            );
            if repetition == 1
                && let Some(dir) = output
            {
                write_audio(&dir.join(format!("{label}.wav")), &out)?;
            }
        }
    }
    Ok(())
}

fn read_reference(path: &str) -> Result<(&str, Vec<f32>, u32)> {
    let mut wav = hound::WavReader::open(path)?;
    let format = wav.spec();
    ensure!(
        format.channels == 1 && format.bits_per_sample == 16,
        "expected mono PCM16 WAV"
    );
    let samples = wav
        .samples::<i16>()
        .map(|sample| sample.map(|sample| f32::from(sample) / 32768.0))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((path, samples, format.sample_rate))
}

fn measured<T: ?Sized, O>(
    mut run: impl FnMut(&T) -> Result<O>,
    input: &T,
    label: &str,
    repetition: usize,
    memory: &memory::Memory,
) -> Result<O> {
    run(input).inspect_err(|_| {
        if let Ok(snapshot) = memory.snapshot() {
            println!("{label} repetition={repetition} failed {snapshot}");
        }
    })
}

fn make_reference(
    runtime: &Runtime,
    device: DevicePreference,
    path: &Path,
    output: &str,
) -> Result<()> {
    let model = TtsModel::parse("alibaba/qwen3-tts-12hz-0.6b-customvoice")?;
    let voice = TtsVoice::Preset {
        speaker: TtsSpeaker::Ryan,
        language: TtsLanguage::English,
    };
    let engine = runtime.block_on(orchion_qwen3::Tts::load_with_device(model, path, device))?;
    let audio =
        runtime.block_on(engine.synthesize_with(REFERENCE_TEXT, voice, legacy_tts_options()))?;
    check_audio(&audio)?;
    write_audio(Path::new(output), &audio)
}

fn write_audio(output: &Path, audio: &TtsAudio) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav = hound::WavWriter::create(output, spec)
        .with_context(|| format!("create {}", output.display()))?;
    for &sample in &audio.samples {
        #[allow(clippy::cast_possible_truncation)]
        wav.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    wav.finalize()?;
    Ok(())
}

fn legacy_tts_options() -> orchion_core::TtsOptions {
    orchion_core::TtsOptions {
        max_length: TTS_FRAME_LIMIT,
        seed: Some(42),
        top_k: 50,
        ..Default::default()
    }
}

fn check_audio(audio: &TtsAudio) -> Result<()> {
    ensure!(
        audio.sample_rate == 24_000 && !audio.samples.is_empty(),
        "expected nonempty 24 kHz TTS output"
    );
    ensure!(
        audio.samples.iter().all(|x| x.is_finite()),
        "nonfinite TTS output"
    );
    Ok(())
}

fn seconds(samples: usize, rate: u32) -> f64 {
    f64::from(u32::try_from(samples).expect("benchmark clips must fit in u32")) / f64::from(rate)
}
