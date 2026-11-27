# Qwen3-TTS: packed projections and grouped CUDA decode

This follows the [RoPE and acoustic-code update](vendor-tts-attention-performance.md).
The reference is the vendored backend at `4f9ed98`. Tests use local
checkpoints and keep the seed, sampler settings, attention context,
and waveform decoder configuration fixed.

## Implementation

- File-backed model loading uploads weights on demand. Talker and
  CodePredictor pack Q/K/V one layer at a time; after packing, the
  original projections can be freed. Prefill still uses three
  projections through views into the packed storage. Single-token
  decode uses the combined projection.
- Without a speaker encoder the model cannot clone from reference
  audio, so it does not upload the speech tokenizer's unused encoder.
  Base models still load both encoders for reference-audio cloning.
- For single-token, unmasked CUDA decode, grouped attention reshapes
  Q so that query heads share their existing K/V cache rows. This is
  the approach in Crane 26.9.0's `GqaAttention`; the vendor path
  keeps score scaling after QK matmul as in its previous path.
  The rest of attention and the KV cache context are unchanged.

The old pack-at-construction trial kept the entire original weight
map on the GPU. VoiceDesign duplicated 488 MiB of projections at
load. Packing one layer at a time removes that cumulative copy.

## Windows timings

RTX 3080, WDDM, native MSVC CUDA Release, three calls after warm-up
per case. The [reference](windows-speech-validation.md#standalone-cuda-benchmark)
is the earlier Windows run on the same machine. These runs were made
in separate sessions. RTF divides latency by generated audio duration.

| Workload | Audio before → combined | Latency before → combined | RTF before → packed only → combined | Live peak before → combined |
| --- | ---: | ---: | ---: | ---: |
| CustomVoice, short | 6.00 → 5.52 s | 6.71 → 5.33 s | 1.119 → 1.025 → 0.965 | 2,917.1 → 2,700.6 MiB |
| CustomVoice, long | 26.64 → 25.92 s | 29.22 → 23.37 s | 1.097 → 0.975 → 0.902 | 2,918.2 → 2,701.8 MiB |
| Base clone | 5.04 → 5.12 s | 6.44 → 5.71 s | 1.278 → 1.188 → 1.115 | 2,912.1 → 2,912.1 MiB |
| VoiceDesign | 4.32 → 4.40 s | 4.79 → 4.35 s | 1.109 → 1.067 → 0.988 | 4,737.4 → 4,531.8 MiB |

The packed-only build includes the same loading changes but uses
the old expanded attention. Packing alone changed all four WAVs.
Grouped attention further reduced the long fixture's RTF by 7.5%
relative to packed-only, and the combined build reduced it by
17.8% relative to the reference. The combined build also improved
RTF on the other three fixtures.

Windows VoiceDesign's model allocation after each request fell from
4,308.0 to 4,091.6 MiB, reserved pages from 4,384 to 4,128 MiB,
and load peak from 4,375.3 to 4,124.8 MiB. All combined benchmark
processes reported zero live and reserved CUDA pool bytes after
unloading. Those counters are separate from per-process WDDM memory.

## Linux memory pressure

On the Linux RTX 3080, the reference build loaded VoiceDesign but
OOMed during warm-up with 5,225 MiB initially free. The final
combined build passed warm-up and three calls with 5,229 MiB initially
free. The median was 1.11 s for 4.40 s of audio (RTF 0.252).
It peaked at 4,531.8 MiB live, returned to 4,091.6 MiB after each
request, and released both live and reserved pool memory on drop.
Its WAV matched the Windows combined output byte for byte.

In separate Linux processes, the long CustomVoice fixture went from
6.28 s to 5.28 s (RTF 0.236 to 0.204). Live peak reached 2,701.8 MiB
instead of 2,918.2 MiB.

## Output interpretation

Grouping does not approximate attention or discard context. It
changes the GEMM shapes and strides, which can change the reduction
order and floating-point rounding. Packing the projections also
changes GEMM shape. In sampled autoregressive generation a small
change in logits can select a different codec ID with the same seed;
that change can propagate to later frames and change the stop point.
Different codec IDs and WAVs demonstrate a reproducibility change,
but by themselves say nothing about intelligibility or naturalness.

For these four fixtures, the old and combined WAVs were decoded by
Qwen3-ASR 0.6B after resampling to 16 kHz. The ASR transcripts matched
the input verbatim in both builds (17 characters each for short,
clone, and design; 93 for long, excluding punctuation). This checks
spoken content on the fixtures; it does not score naturalness or
voice-prompt adherence.

The Base checkpoint's ECAPA-TDNN speaker encoder gave baseline-to-
combined embedding cosine similarities of 0.9955 (short),
0.9982 (long), 0.9984 (clone), and 0.9954 (design). The Base clone's
similarity to the original reference WAV was 0.9875 before and
0.9871 after. These are model-based checks on the tested samples;
no human listening score was collected.

## Release checks

Windows MSVC/CUDA Release built successfully. The isolated server
passed automatic and forced-English ASR, WebSocket partials and
finals, and all three TTS modes. Two requests with the same seed and
sampling settings produced identical WAVs within this build.
ASR and CustomVoice each passed three HTTP unload cycles: dedicated
WDDM memory returned to the 235 MiB baseline every time. Warm
CustomVoice used 2,453 MiB, versus 2,709 MiB in the previous Windows
server check.

### Client timeout

Before publication, each Windows binary was tested on an isolated
loopback server with a warmed 0.6B CustomVoice model. The same long
text, seed 42, `top_k=50`, and `max_length=2048` were used for both.
The HTTP client closed the connection after a two-second timeout.

| Version | Frames at disconnect | Total generated frames | Time from disconnect to synthesis completion |
| --- | ---: | ---: | ---: |
| Previous | 22 | 333 | 30.58 s |
| Packed and grouped | 33 | 324 | 20.32 s |

Both versions continued generating frames after the disconnect, but
both stopped at EOS well below the cap. With `max_length=32`, the
control requests stopped at exactly 32 frames and returned 2.56 s of
audio. This reproduces a missing cancellation of dispatched TTS
work independently of late EOS. The HTTP request task keeps
dispatched operations alive, and the generation loop checks EOS
and the frame cap but does not check client disconnects.

The Windows build was published to
`E:\Playground\orchion\publish\orchion-server.exe` after these checks.
The previous binary is in
`E:\Playground\orchion\publish-backups\20261127-tts-packed-test\orchion-server.exe`.

On Linux, 128 Release unit tests, one doc test, six opt-in CUDA
checks (including streaming and model unload), the grouped CUDA
attention check, the non-CUDA build, and strict Clippy passed.
The long TTS workspace test peaked at 2,869 MiB and reclaimed its
workspace after the request. In three TTS unload cycles, the
process returned to its 262 MiB CUDA runtime baseline each time.

## Reproduce

Use the [existing local fixtures and offline build](vendor-asr-prefill-performance.md#reproduce).
Run the `legacy` backend with the `tts`, `clone`, and `design`
workloads. Leave `ORCHION_TTS_PROFILE` unset when measuring latency.
Export the WAVs from the same seed and options for the content check.
For each WAV, resample to 16 kHz mono PCM and pass it to the
`asr-stream` workload with the ASR 0.6B checkpoint:

```sh
ffmpeg -v error -i input.wav -ac 1 -ar 16000 -f s16le input.pcm
speech_bench legacy cuda:0 asr-stream "$ASR_MODEL_DIR" input.pcm
```
