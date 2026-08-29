#[allow(
    clippy::wildcard_imports,
    reason = "this submodule implements the parent stream transport"
)]
use super::*;
use crate::application::streaming_transcription::CaptionSession;

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the arguments and loop keep websocket transport transitions explicit"
)]
pub(super) async fn run(
    mut socket: WebSocket,
    start: TranscriptionStreamStart,
    asr: LeasedAsrModel,
    default_chunk_size_sec: f32,
    stream_target_segment_millis: u32,
    limits: TranscriptionStreamLimits,
    mut budget: TranscriptionStreamBudget,
    activity: Option<WebSocketActivity>,
) {
    let streaming_options = start.to_streaming_options(default_chunk_size_sec);
    let mut decoder =
        match await_stream_operation(&budget, limits, start.audio_decoder(limits.max_duration))
            .await
        {
            Ok(Ok(decoder)) => decoder,
            Ok(Err(error)) | Err(error) => {
                let _ = send_stream_error(&mut socket, error, activity.as_ref()).await;
                return;
            }
        };
    let mut endpoint = match AudioVadStreamingEndpoint::new(start.endpointing.to_vad_config()) {
        Ok(endpoint) => endpoint,
        Err(error) => {
            let _ = send_stream_error(&mut socket, ApiError::from(error), activity.as_ref()).await;
            return;
        }
    };
    let mut session = CaptionSession::new(asr, streaming_options, stream_target_segment_millis);
    match await_stream_operation(&budget, limits, send_stream_ready(&mut socket)).await {
        Ok(Ok(())) => {}
        Ok(Err(_)) => return,
        Err(error) => {
            let _ = send_stream_error(&mut socket, error, activity.as_ref()).await;
            return;
        }
    }

    loop {
        let message = match receive_transcription_stream_message(
            &mut socket,
            &mut budget,
            limits,
            activity.as_ref(),
        )
        .await
        {
            Ok(Some(message)) => message,
            Ok(None) => return,
            Err(error) => {
                let _ = send_stream_error(&mut socket, error, activity.as_ref()).await;
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
                                activity.as_ref(),
                            )
                            .await;
                            return;
                        }
                        Err(error) => {
                            let _ = send_stream_error(&mut socket, error, activity.as_ref()).await;
                            return;
                        }
                    };
                if let Err(error) = budget.record_decoded_samples(
                    audio_chunk.samples.len(),
                    audio_chunk.sample_rate,
                    limits,
                ) {
                    let _ =
                        send_stream_error(&mut socket, ApiError::from(error), activity.as_ref())
                            .await;
                    return;
                }
                let events = match endpoint.push(&audio_chunk.samples, audio_chunk.sample_rate) {
                    Ok(events) => events,
                    Err(error) => {
                        let _ = send_stream_error(
                            &mut socket,
                            ApiError::from(error),
                            activity.as_ref(),
                        )
                        .await;
                        return;
                    }
                };
                match await_stream_operation(
                    &budget,
                    limits,
                    apply_caption_events(
                        &mut socket,
                        &mut session,
                        events,
                        audio_chunk.sample_rate,
                    ),
                )
                .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) | Err(error) => {
                        let _ = send_stream_error(&mut socket, error, activity.as_ref()).await;
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
                        "caption",
                        finish(
                            &mut socket,
                            decoder,
                            endpoint,
                            &mut session,
                            &mut budget,
                            limits,
                        ),
                    )
                    .await;
                    if let Err(error) = result {
                        let _ = send_stream_error(&mut socket, error, activity.as_ref()).await;
                    } else if let Some(activity) = &activity {
                        activity.complete_success();
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
                        activity.as_ref(),
                    )
                    .await;
                    return;
                }
                Err(error) => {
                    let _ = send_stream_error(&mut socket, error, activity.as_ref()).await;
                    return;
                }
            },
            Message::Close(_) => return,
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
}

async fn apply_caption_events(
    socket: &mut WebSocket,
    session: &mut CaptionSession,
    events: Vec<orchion::AudioVadStreamingEvent>,
    sample_rate: u32,
) -> Result<(), ApiError> {
    let events = session
        .apply_vad_events(events, sample_rate)
        .await
        .map_err(ApiError::from)?;
    send_application_stream_events(socket, events).await
}

async fn finish(
    socket: &mut WebSocket,
    decoder: StreamingAudioDecoder,
    mut endpoint: AudioVadStreamingEndpoint,
    session: &mut CaptionSession,
    budget: &mut TranscriptionStreamBudget,
    limits: TranscriptionStreamLimits,
) -> Result<(), ApiError> {
    let final_audio = decoder
        .finish()
        .await
        .map_err(|error| transcription_stream_decoder_error(error, limits))?;
    budget.record_decoded_samples(final_audio.samples.len(), final_audio.sample_rate, limits)?;
    apply_caption_events(
        socket,
        session,
        endpoint
            .push(&final_audio.samples, final_audio.sample_rate)
            .map_err(ApiError::from)?,
        final_audio.sample_rate,
    )
    .await?;
    apply_caption_events(socket, session, endpoint.finish(), final_audio.sample_rate).await?;
    let events = session.complete().await.map_err(ApiError::from)?;
    send_application_stream_events(socket, events).await
}
