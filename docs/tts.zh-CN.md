# TTS API

[English](tts.md)

Orchion 通过 `POST /v1/audio/speech` 提供语音合成能力。

## 语言选择

省略 `language` 或设置为 `auto` 时自动选择语言。日文、韩文、中文和西里尔文字按文字类型识别；其他文本在识别置信度大于 `0.70` 且语言受支持时采用识别结果，置信度较低或无法识别时默认使用 English。高置信度识别出的不支持语言或文字类型仍会报错。显式指定 `language` 会跳过自动检测。

## 预设音色

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

## 音色克隆

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

## 音色设计

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

支持的输出格式为 `wav`、`mp3`、`aac`、`opus`、`flac` 和 `pcm`。如果省略 `response_format`，服务端会使用 `config.toml` 中 `[services.tts] format` 的配置。

Qwen3 TTS 请求也支持 `seed`、`temperature`、`top_k`、`top_p`、`repetition_penalty` 和 `max_length`。
