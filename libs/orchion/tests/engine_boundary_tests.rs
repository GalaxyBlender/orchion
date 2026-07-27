#![cfg(all(feature = "asr", feature = "tts"))]

use orchion::{
    Asr, AsrEngine, AsrEngineFuture, AsrModel, AsrOptions, AsrStreamSession, AsrStreamingOptions,
    AsrTranscript, Result, Tts, TtsAudio, TtsEngine, TtsEngineFuture, TtsLanguage, TtsModel,
    TtsOptions, TtsSpeaker, TtsVoice,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct TestAsrEngine {
    calls: Arc<AtomicUsize>,
}

impl AsrEngine for TestAsrEngine {
    fn model(&self) -> AsrModel {
        AsrModel::parse("Qwen/Qwen3-ASR-0.6B").unwrap()
    }

    fn transcribe_file_with(
        &self,
        path: PathBuf,
        _options: AsrOptions,
    ) -> AsrEngineFuture<'_, AsrTranscript> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(transcript(path.display().to_string())) })
    }

    fn transcribe_samples_with(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        _options: AsrOptions,
    ) -> AsrEngineFuture<'_, AsrTranscript> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(transcript(format!("{}@{sample_rate}", samples.len()))) })
    }

    fn start_streaming_with(
        &self,
        _options: AsrStreamingOptions,
    ) -> AsrEngineFuture<'_, Box<dyn AsrStreamSession>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(Box::new(TestAsrStream) as Box<dyn AsrStreamSession>) })
    }
}

struct TestAsrStream;

impl AsrStreamSession for TestAsrStream {
    fn feed(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> AsrEngineFuture<'_, Option<AsrTranscript>> {
        Box::pin(async move {
            Ok(Some(transcript(format!(
                "partial:{}@{sample_rate}",
                samples.len()
            ))))
        })
    }

    fn finish(self: Box<Self>) -> AsrEngineFuture<'static, AsrTranscript> {
        Box::pin(async { Ok(transcript("final")) })
    }
}

struct TestTtsEngine {
    calls: Arc<AtomicUsize>,
}

impl TtsEngine for TestTtsEngine {
    fn model(&self) -> TtsModel {
        TtsModel::parse("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").unwrap()
    }

    fn synthesize_with(
        &self,
        text: String,
        _voice: TtsVoice,
        _options: TtsOptions,
    ) -> TtsEngineFuture<'_, TtsAudio> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let text_len = u16::try_from(text.len()).expect("test text length fits u16");
        Box::pin(async move { Ok(TtsAudio::new(vec![f32::from(text_len)], 24_000)) })
    }

    fn synthesize_to_file(
        &self,
        _text: String,
        _voice: TtsVoice,
        _output_path: PathBuf,
    ) -> TtsEngineFuture<'_, ()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn asr_facade_dispatches_through_a_provider_neutral_engine_and_stream() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let asr = Asr::from_engine(Arc::new(TestAsrEngine {
        calls: Arc::clone(&calls),
    }));

    assert_eq!(asr.model().as_str(), "Qwen/Qwen3-ASR-0.6B");
    assert_eq!(
        asr.transcribe_samples(&[0.0, 0.5], 16_000).await?.text,
        "2@16000"
    );
    let mut stream = asr.start_streaming().await?;
    assert_eq!(
        stream.feed(&[0.0], 16_000).await?.unwrap().text,
        "partial:1@16000"
    );
    assert_eq!(stream.finish().await?.text, "final");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_send_sync(asr);
    Ok(())
}

#[tokio::test]
async fn tts_facade_dispatches_through_a_provider_neutral_engine() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let tts = Tts::from_engine(Arc::new(TestTtsEngine {
        calls: Arc::clone(&calls),
    }));
    let voice = TtsVoice::Preset {
        speaker: TtsSpeaker::Ryan,
        language: TtsLanguage::English,
    };

    assert_eq!(tts.model().as_str(), "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");
    assert_eq!(tts.synthesize("hello", voice).await?.samples, [5.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_send_sync(tts);
    Ok(())
}

fn transcript(text: impl Into<String>) -> AsrTranscript {
    AsrTranscript {
        text: text.into(),
        language: "en".to_string(),
        raw_output: String::new(),
        segments: Vec::new(),
    }
}

fn assert_send_sync<T: Send + Sync>(_value: T) {}
