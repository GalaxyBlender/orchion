#[allow(
    clippy::wildcard_imports,
    reason = "this submodule implements the parent stream state machine"
)]
use super::*;

#[allow(
    clippy::too_many_lines,
    reason = "the loop keeps websocket state transitions visible in one place"
)]
pub(super) async fn run(
    mut socket: WebSocket,
    start: TranscriptionStreamStart,
    asr: LeasedAsrModel,
    default_chunk_size_sec: f32,
    limits: TranscriptionStreamLimits,
    mut budget: TranscriptionStreamBudget,
) {
    let streaming_options = start.to_streaming_options(default_chunk_size_sec);
    if let Err(error) = validate_transcription_streaming_options(&streaming_options) {
        let _ = send_stream_error(&mut socket, error).await;
        return;
    }
    let chunk_size_sec = streaming_options.chunk_size_sec;
    let mut stream =
        match await_stream_operation(&budget, limits, asr.start(streaming_options)).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(error)) => {
                let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
                return;
            }
            Err(error) => {
                let _ = send_stream_error(&mut socket, error).await;
                return;
            }
        };
    let mut decoder =
        match await_stream_operation(&budget, limits, start.audio_decoder(limits.max_duration))
            .await
        {
            Ok(Ok(decoder)) => decoder,
            Ok(Err(error)) | Err(error) => {
                let _ = send_stream_error(&mut socket, error).await;
                return;
            }
        };
    let mut pcm_buffer = AsrPcmBuffer::new(chunk_size_sec);

    match await_stream_operation(&budget, limits, send_stream_ready(&mut socket)).await {
        Ok(Ok(())) => {}
        Ok(Err(_)) => return,
        Err(error) => {
            let _ = send_stream_error(&mut socket, error).await;
            return;
        }
    }

    loop {
        let message =
            match receive_transcription_stream_message(&mut socket, &mut budget, limits).await {
                Ok(Some(message)) => message,
                Ok(None) => return,
                Err(error) => {
                    let _ = send_stream_error(&mut socket, error).await;
                    return;
                }
            };
        match message {
            Message::Binary(bytes) => {
                let audio_chunk =
                    match await_stream_operation(&budget, limits, decoder.push(&bytes)).await {
                        Ok(Ok(audio_chunk)) => audio_chunk,
                        Ok(Err(error)) => {
                            let _ = send_stream_error(
                                &mut socket,
                                transcription_stream_decoder_error(error, limits),
                            )
                            .await;
                            return;
                        }
                        Err(error) => {
                            let _ = send_stream_error(&mut socket, error).await;
                            return;
                        }
                    };
                if let Err(error) = budget.record_decoded_samples(
                    audio_chunk.samples.len(),
                    audio_chunk.sample_rate,
                    limits,
                ) {
                    let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
                    return;
                }
                let chunks = match pcm_buffer.push(&audio_chunk.samples, audio_chunk.sample_rate) {
                    Ok(chunks) => chunks,
                    Err(error) => {
                        let _ = send_stream_error(&mut socket, ApiError::from(error)).await;
                        return;
                    }
                };
                match await_stream_operation(
                    &budget,
                    limits,
                    feed_chunks(&mut socket, &mut stream, chunks),
                )
                .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) | Err(error) => {
                        let _ = send_stream_error(&mut socket, error).await;
                        return;
                    }
                }
            }
            Message::Text(text) => match parse_transcription_stream_control(text.as_str()) {
                Ok(TranscriptionStreamControl::End) => {
                    let remaining = budget.remaining(limits);
                    let result = await_stream_finish(
                        remaining,
                        limits,
                        "legacy",
                        finish(
                            &mut socket,
                            decoder,
                            stream,
                            pcm_buffer,
                            &mut budget,
                            limits,
                        ),
                    )
                    .await;
                    if let Err(error) = result {
                        let _ = send_stream_error(&mut socket, error).await;
                    }
                    return;
                }
                Ok(TranscriptionStreamControl::Start) => {
                    let _ = send_stream_error(
                        &mut socket,
                        ApiError::invalid_request(
                            "transcription stream has already started",
                            Some("type"),
                            Some("invalid_stream_state"),
                        ),
                    )
                    .await;
                    return;
                }
                Err(error) => {
                    let _ = send_stream_error(&mut socket, error).await;
                    return;
                }
            },
            Message::Close(_) => return,
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
}

async fn finish(
    socket: &mut WebSocket,
    decoder: StreamingAudioDecoder,
    mut stream: LeasedAsrStream,
    mut pcm_buffer: AsrPcmBuffer,
    budget: &mut TranscriptionStreamBudget,
    limits: TranscriptionStreamLimits,
) -> Result<(), ApiError> {
    let final_audio = decoder
        .finish()
        .await
        .map_err(|error| transcription_stream_decoder_error(error, limits))?;
    budget.record_decoded_samples(final_audio.samples.len(), final_audio.sample_rate, limits)?;
    let chunks = pcm_buffer
        .push(&final_audio.samples, final_audio.sample_rate)
        .map_err(ApiError::from)?;
    feed_chunks(socket, &mut stream, chunks).await?;
    if let Some((samples, sample_rate)) = pcm_buffer.drain_remaining()
        && let Some(transcript) = stream
            .feed(&samples, sample_rate)
            .await
            .map_err(ApiError::from)?
    {
        send_stream_transcript(socket, "partial", &transcript)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
    }
    let transcript = stream.finish().await.map_err(ApiError::from)?;
    send_stream_transcript(socket, "final", &transcript)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))
}

async fn feed_chunks(
    socket: &mut WebSocket,
    stream: &mut LeasedAsrStream,
    chunks: Vec<(Vec<f32>, u32)>,
) -> Result<(), ApiError> {
    for (samples, sample_rate) in chunks {
        match stream.feed(&samples, sample_rate).await {
            Ok(Some(transcript)) => send_stream_transcript(socket, "partial", &transcript)
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?,
            Ok(None) => {}
            Err(error) => return Err(ApiError::from(error)),
        }
    }
    Ok(())
}
