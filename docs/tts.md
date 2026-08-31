# TTS API

[简体中文](tts.zh-CN.md)

Orchion exposes speech synthesis through `POST /v1/audio/speech`.

## Preset Voice

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "alibaba/qwen3-tts-12hz-0.6b-customvoice",
    "input": "Hello from Orchion.",
    "voice": "ryan",
    "seed": 42,
    "response_format": "wav"
  }' \
  --output speech.wav
```

## Voice Clone

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -F model=alibaba/qwen3-tts-12hz-0.6b-base \
  -F input='Read this with the reference voice.' \
  -F voice=clone \
  -F reference_audio=@reference.wav \
  -F reference_text='Text spoken in the reference audio.' \
  -F language=english \
  -F response_format=wav \
  --output cloned.wav
```

## Voice Design

```sh
curl http://127.0.0.1:9090/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "alibaba/qwen3-tts-12hz-1.7b-voicedesign",
    "input": "Read this with a designed voice.",
    "voice": "design",
    "voice_prompt": "A calm narrator with a warm studio tone.",
    "language": "english",
    "response_format": "wav"
  }' \
  --output designed.wav
```

Supported output formats are `wav`, `mp3`, `aac`, `opus`, `flac`, and `pcm`. If `response_format` is omitted, the server uses `[services.tts] format` from `config.toml`.

Qwen3 TTS requests also accept `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, and `max_length`.
