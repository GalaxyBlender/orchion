use orchion::TtsAudio;
use orchion_server::audio::encode_wav;

#[test]
fn encodes_tts_audio_as_wav_bytes() {
    let audio = TtsAudio::new(vec![0.0, 0.5, -0.5], 24_000);
    let bytes = encode_wav(&audio).unwrap();

    assert!(bytes.starts_with(b"RIFF"));
    assert_eq!(&bytes[8..12], b"WAVE");
}
