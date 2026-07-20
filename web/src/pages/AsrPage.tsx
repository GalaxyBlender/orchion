import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { asrLanguageOptions, asrParameterMetadata, asrResponseFormats, asrTimestampGranularityOptions } from "@/features/asr/metadata";
import { buildAsrCurl, buildAsrFormData, summarizeAsrRequest } from "@/features/asr/request";
import {
  acquireAsrMicrophoneStream,
  asrStreamEndpointPath,
  buildAsrStreamStartMessage,
  buildAsrStreamUrl,
  detectMicrophoneSupportIssue,
  detectAsrStreamInputFormat,
  formatAsrStreamResult,
  isAsrCaptionPartialEvent,
  parseAsrStreamEvent,
  preferredMicrophoneMimeType,
  upsertBoundedAsrCaptionSegments,
  validateAsrCaptionEndpointingOptions,
  waitForAsrStreamWritable,
} from "@/features/asr/streaming";
import type { AsrCaptionEndpointingOptions, AsrFormState, AsrMode, AsrRequestInput, AsrResponseFormat, AsrStreamEvent, AsrStreamInputFormat, AsrStreamInputMode, AsrStreamOutputMode } from "@/features/asr/types";
import { useModels } from "@/features/models/useModels";
import { apiUrl, authHeaders } from "@/shared/api/client";
import { parseApiError } from "@/shared/api/errors";
import { buildApiError, readResponsePayload, type SubmissionError } from "@/shared/api/apiHelpers";
import { copyTextToClipboard } from "@/shared/clipboard";
import {
  loadPersistentState,
  savePersistentState,
  type PersistentAsrState,
  type PersistentState,
} from "@/shared/storage/persistentState";
import { Card, FormField, Input, Select, TextArea, Button, Alert, StateView, ModelStatus, CodePreview, FileDropZone, useToast, MetadataPanel, SuggestionInput, Tabs } from "@/shared/ui";
import { Play, Clipboard, Mic, Square } from "lucide-react";

const endpointPath = "/v1/audio/transcriptions";
const previewFile = new File([""], "preview-audio-placeholder.wav", { type: "audio/wav" });
const streamFileChunkSize = 64 * 1024;
const asrStreamResponseFormats: AsrResponseFormat[] = ["json"];
const captionEndpointingDefaults: CaptionEndpointingFormState = {
  minSpeechMs: "300",
  minSilenceMs: "500",
  speechPaddingMs: "200",
};

type AsrStreamStatus = "idle" | "connecting" | "streaming" | "finishing";

interface CaptionSegmentView {
  id: number;
  text: string;
  startMs?: number;
  endMs?: number;
}

interface CaptionEndpointingFormState {
  minSpeechMs: string;
  minSilenceMs: string;
  speechPaddingMs: string;
}

export function AsrPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const [persistentState, setPersistentState] = useState<PersistentState>(() => loadPersistentState());
  const [form, setForm] = useState<AsrFormState>(() => asrStateToForm(persistentState.asr));
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState("");
  const [submitError, setSubmitError] = useState<SubmissionError | null>(null);
  const [result, setResult] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [mode, setMode] = useState<AsrMode>("file");
  const [streamInputMode, setStreamInputMode] = useState<AsrStreamInputMode>("microphone");
  const [streamOutputMode, setStreamOutputMode] = useState<AsrStreamOutputMode>("caption");
  const [captionEndpointing, setCaptionEndpointing] = useState<CaptionEndpointingFormState>(captionEndpointingDefaults);
  const [streamFile, setStreamFile] = useState<File | null>(null);
  const [streamStatus, setStreamStatus] = useState<AsrStreamStatus>("idle");
  const [partialTranscript, setPartialTranscript] = useState("");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [currentCaptionSegment, setCurrentCaptionSegment] = useState<CaptionSegmentView | null>(null);
  const [captionSegments, setCaptionSegments] = useState<CaptionSegmentView[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);
  const webSocketRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const streamEndedRef = useRef(false);
  const streamStopRequestedRef = useRef(false);
  const streamSessionRef = useRef(0);
  const settings = persistentState.settings;
  const models = useModels(settings);
  const asrModelIds = models.classified.asr.map((model) => model.id);
  
  const previewInput = useMemo(() => buildRequestInput(form, file ?? previewFile), [file, form]);
  
  const requestSummary = useMemo(
    () =>
      summarizeAsrRequest(previewInput, {
        model: (model) => t("asr.summary.model", { model }),
        file: (selectedFile) => t("asr.summary.file", { file: selectedFile }),
        responseFormat: (format) => t("asr.summary.responseFormat", { format }),
        language: (language) => t("asr.summary.language", { language }),
        timestamp: (values) => t("asr.summary.timestamp", { values }),
      }),
    [previewInput, t],
  );
  
  const curlPreview = useMemo(() => buildAsrCurl(settings, previewInput), [previewInput, settings]);
  const streamConnectionPreview = useMemo(() => `WebSocket ${buildAsrStreamUrl()}`, []);
  const streamRequestSummary = useMemo(
    () => [
      t("asr.stream.summary.endpoint", { endpoint: asrStreamEndpointPath, defaultValue: "WebSocket: {{endpoint}}" }),
      t("asr.summary.model", { model: form.model.trim() || "-" }),
      t("asr.summary.responseFormat", { format: "json" }),
      streamOutputMode === "caption"
        ? t("asr.stream.summary.caption", "Output: stable caption segments")
        : t("asr.stream.summary.transcript", "Output: live transcript"),
      ...(streamOutputMode === "caption" ? [captionEndpointingSummary(captionEndpointing, t)] : []),
      streamInputMode === "microphone"
        ? t("asr.stream.summary.microphone", "Input: live microphone audio")
        : t("asr.stream.summary.file", { file: streamFile?.name ?? "-", defaultValue: "Streaming file: {{file}}" }),
    ],
    [captionEndpointing, form.model, form.responseFormat, streamFile?.name, streamInputMode, streamOutputMode, t],
  );

  const responseFormatOptions = mode === "stream" ? asrStreamResponseFormats : asrResponseFormats;
  const isStreaming = streamStatus !== "idle";

  const modeTabs = useMemo(
    () => [
      { id: "file", label: t("asr.modes.file.0", "File transcription") },
      { id: "stream", label: t("asr.modes.stream.0", "Live transcription") },
    ],
    [t],
  );

  const streamInputTabs = useMemo(
    () => [
      { id: "microphone", label: t("asr.stream.input.microphone", "Live recording") },
      { id: "file", label: t("asr.stream.input.file", "Streaming file") },
    ],
    [t],
  );

  const streamOutputTabs = useMemo(
    () => [
      { id: "caption", label: t("asr.stream.output.caption", "Captions"), disabled: isStreaming },
      { id: "transcript", label: t("asr.stream.output.transcript", "Live transcript"), disabled: isStreaming },
    ],
    [isStreaming, t],
  );

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      closeStreamResources();
    };
  }, []);

  const updateMode = (nextMode: AsrMode) => {
    setMode(nextMode);
    setValidationError("");
    setSubmitError(null);
    setResult("");
    clearStreamResult();
    if (nextMode === "stream" && !isStreamingResponseFormat(form.responseFormat)) {
      updateForm("responseFormat", "json");
    }
  };

  const updateStreamInputMode = (nextMode: AsrStreamInputMode) => {
    setStreamInputMode(nextMode);
    setValidationError("");
    setSubmitError(null);
    setResult("");
    clearStreamResult();
  };

  const updateStreamOutputMode = (nextMode: AsrStreamOutputMode) => {
    if (isStreaming) {
      showValidationError(t("asr.stream.outputChangeDisabled", "Stop the current stream before changing the output mode."));
      return;
    }
    setStreamOutputMode(nextMode);
    setValidationError("");
    setSubmitError(null);
    setResult("");
    clearStreamResult();
  };

  const updateCaptionEndpointing = (field: keyof CaptionEndpointingFormState, value: string) => {
    setValidationError("");
    setSubmitError(null);
    setResult("");
    setCaptionEndpointing((current) => ({ ...current, [field]: value }));
  };

  const updateForm = <K extends keyof AsrFormState>(field: K, value: AsrFormState[K]) => {
    setValidationError("");
    setSubmitError(null);
    setResult("");
    setForm((currentForm) => {
      const nextForm = { ...currentForm, [field]: value };
      setPersistentState((currentState) => {
        const nextState: PersistentState = {
          ...currentState,
          asr: formToAsrState(nextForm),
        };
        savePersistentState(nextState);
        return nextState;
      });
      return nextForm;
    });
  };

  const handleFileSelect = (selectedFile: File | null) => {
    setValidationError("");
    setSubmitError(null);
    setResult("");
    setFile(selectedFile);
  };

  const handleStreamFileSelect = (selectedFile: File | null) => {
    setValidationError("");
    setSubmitError(null);
    setResult("");
    clearStreamResult();
    setStreamFile(selectedFile);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (mode !== "file") {
      return;
    }
    setSubmitError(null);
    setResult("");

    const model = form.model.trim();
    if (!file) {
      showValidationError(t("asr.missingFile"));
      return;
    }
    if (model === "") {
      showValidationError(t("asr.missingModel"));
      return;
    }

    setValidationError("");
    setIsSubmitting(true);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const response = await fetch(apiUrl(settings, endpointPath), {
        method: "POST",
        headers: authHeaders(settings),
        body: buildAsrFormData(buildRequestInput({ ...form, model }, file)),
        signal: abortController.signal,
      });

      if (!response.ok) {
        // Build API error using custom helpers
        throw parseApiError(response, await readResponsePayload(response));
      }

      setResult(await formatResponse(response, form.responseFormat));
      toast.success(t("common.success", "Transcription completed successfully!"));
    } catch (caughtError) {
      if (isAbortError(caughtError)) {
        toast.info(t("asr.cancelled"));
        return;
      }
      setSubmitError(buildApiError(caughtError));
      toast.error(t("common.error", "Transcription failed"));
    } finally {
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
      setIsSubmitting(false);
    }
  };

  const handleCancelSubmit = () => {
    abortControllerRef.current?.abort();
  };

  const handleStartStream = async () => {
    setSubmitError(null);
    setResult("");
    clearStreamResult();

    const model = form.model.trim();
    if (model === "") {
      showValidationError(t("asr.missingModel"));
      return;
    }
    const parsedEndpointing =
      streamOutputMode === "caption" ? parseCaptionEndpointing(captionEndpointing, t) : undefined;
    if (typeof parsedEndpointing === "string") {
      showValidationError(parsedEndpointing);
      return;
    }
    closeStreamResources();
    const streamSession = streamSessionRef.current;

    if (streamInputMode === "file") {
      if (!streamFile) {
        showValidationError(t("asr.stream.missingFile", "Select an audio file to stream."));
        return;
      }
      const inputAudioFormat = detectAsrStreamInputFormat(streamFile);
      if (!inputAudioFormat) {
        showValidationError(t("asr.stream.unsupportedFile", "Streaming file input supports wav, mp3, webm/opus, m4a, aac, flac, and ogg."));
        return;
      }
      const selectedStreamFile = streamFile;
      openTranscriptionStream(inputAudioFormat, parsedEndpointing, streamSession, (socket) => {
        void sendStreamFile(socket, selectedStreamFile, streamSession);
      });
      return;
    }

    const microphoneSupportIssue = detectMicrophoneSupportIssue();
    if (microphoneSupportIssue) {
      showValidationError(t(`asr.stream.microphoneSupport.${microphoneSupportIssue}`));
      return;
    }

    try {
      setStreamStatus("connecting");
      const mediaStream = await acquireAsrMicrophoneStream(
        () => navigator.mediaDevices.getUserMedia({ audio: true }),
        () => streamSessionRef.current === streamSession,
      );
      if (!mediaStream) {
        return;
      }
      const mimeType = preferredMicrophoneMimeType();
      const recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
      mediaStreamRef.current = mediaStream;
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size === 0) {
          return;
        }
        void event.data.arrayBuffer().then((buffer) => {
          if (streamSessionRef.current !== streamSession || mediaRecorderRef.current !== recorder) {
            return;
          }
          const socket = webSocketRef.current;
          if (socket?.readyState === WebSocket.OPEN) {
            socket.send(buffer);
          }
        });
      };
      recorder.onstop = () => {
        if (mediaRecorderRef.current === recorder) {
          mediaRecorderRef.current = null;
        }
        mediaStream.getTracks().forEach((track) => track.stop());
        if (mediaStreamRef.current === mediaStream) {
          mediaStreamRef.current = null;
        }
        if (streamSessionRef.current !== streamSession) {
          return;
        }
        const socket = webSocketRef.current;
        if (streamStopRequestedRef.current && socket?.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "end" }));
          setStreamStatus("finishing");
        }
      };
      openTranscriptionStream("auto", parsedEndpointing, streamSession, () => {
        if (recorder.state === "inactive") {
          recorder.start(500);
        }
      });
    } catch (caughtError) {
      closeStreamResources();
      setStreamStatus("idle");
      setSubmitError(buildApiError(caughtError));
      toast.error(t("common.error", "Transcription failed"));
    }
  };

  const handleStopStream = () => {
    streamStopRequestedRef.current = true;
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setStreamStatus("finishing");
      return;
    }
    stopRecordingResources();
    const socket = webSocketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "end" }));
      setStreamStatus("finishing");
      return;
    }
    closeStreamResources();
    setStreamStatus("idle");
  };

  const openTranscriptionStream = (
    inputAudioFormat: AsrStreamInputFormat,
    endpointing: AsrCaptionEndpointingOptions | undefined,
    streamSession: number,
    onReady: (socket: WebSocket) => void,
  ) => {
    if (streamSessionRef.current !== streamSession) {
      return;
    }
    streamEndedRef.current = false;
    streamStopRequestedRef.current = false;
    setValidationError("");
    setStreamStatus("connecting");

    const socket = new WebSocket(buildAsrStreamUrl());
    webSocketRef.current = socket;
    socket.onopen = () => {
      if (webSocketRef.current !== socket || streamSessionRef.current !== streamSession) {
        return;
      }
      socket.send(
        buildAsrStreamStartMessage({
          form,
          inputAudioFormat,
          outputMode: streamOutputMode,
          endpointing,
          apiKey: settings.apiKey,
        }),
      );
    };
    socket.onmessage = (event) => {
      if (webSocketRef.current !== socket || streamSessionRef.current !== streamSession) {
        return;
      }
      if (typeof event.data === "string") {
        try {
          handleStreamEvent(parseAsrStreamEvent(event.data), socket, onReady);
        } catch {
          handleStreamFailure(t("asr.stream.invalidEvent", "Streaming response was not valid JSON."));
        }
      }
    };
    socket.onerror = () => {
      if (webSocketRef.current !== socket || streamSessionRef.current !== streamSession) {
        return;
      }
      handleStreamFailure(t("asr.stream.connectionError", "Streaming connection failed."));
    };
    socket.onclose = () => {
      if (webSocketRef.current !== socket || streamSessionRef.current !== streamSession) {
        return;
      }
      if (!streamEndedRef.current) {
        handleStreamFailure(t("asr.stream.closed", "Streaming connection closed before a final result."));
      }
    };
  };

  const handleStreamEvent = (
    event: AsrStreamEvent,
    socket: WebSocket,
    onReady: (socket: WebSocket) => void,
  ) => {
    switch (event.type) {
      case "ready":
        setStreamStatus("streaming");
        onReady(socket);
        return;
      case "partial":
        if (isAsrCaptionPartialEvent(event)) {
          setCurrentCaptionSegment({ id: event.segment_id, text: event.text });
          setResult(formatAsrStreamResult(event));
          return;
        }
        setPartialTranscript(event.text);
        setResult(formatAsrStreamResult(event));
        return;
      case "final":
        streamEndedRef.current = true;
        setPartialTranscript("");
        setFinalTranscript(event.text);
        setResult(formatAsrStreamResult(event));
        setStreamStatus("idle");
        closeStreamResources();
        toast.success(t("common.success", "Transcription completed successfully!"));
        return;
      case "segment_final": {
        const segment: CaptionSegmentView = {
          id: event.segment_id,
          text: event.text,
          startMs: event.start_ms,
          endMs: event.end_ms,
        };
        setCurrentCaptionSegment((current) => (current?.id === event.segment_id ? null : current));
        setCaptionSegments((currentSegments) => upsertCaptionSegment(currentSegments, segment));
        setResult(formatAsrStreamResult(event));
        return;
      }
      case "completed":
        streamEndedRef.current = true;
        setPartialTranscript("");
        setCurrentCaptionSegment(null);
        setResult(formatAsrStreamResult(event));
        setStreamStatus("idle");
        closeStreamResources();
        toast.success(t("common.success", "Transcription completed successfully!"));
        return;
      case "error":
        streamEndedRef.current = true;
        setSubmitError({ type: "api", message: event.error.message, detail: event.error });
        setStreamStatus("idle");
        closeStreamResources();
        toast.error(t("common.error", "Transcription failed"));
        return;
    }
  };

  const sendStreamFile = async (socket: WebSocket, selectedFile: File, streamSession: number) => {
    try {
      for (let offset = 0; offset < selectedFile.size; offset += streamFileChunkSize) {
        const writable = await waitForAsrStreamWritable(
          socket,
          () => streamSessionRef.current === streamSession && !streamStopRequestedRef.current,
        );
        if (!writable) {
          return;
        }
        const chunk = selectedFile.slice(offset, offset + streamFileChunkSize);
        const buffer = await chunk.arrayBuffer();
        if (
          streamSessionRef.current !== streamSession ||
          streamStopRequestedRef.current ||
          socket.readyState !== WebSocket.OPEN
        ) {
          return;
        }
        socket.send(buffer);
      }
      if (!streamStopRequestedRef.current && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "end" }));
        setStreamStatus("finishing");
      }
    } catch (caughtError) {
      handleStreamFailure(caughtError instanceof Error ? caughtError.message : String(caughtError));
    }
  };

  const handleStreamFailure = (message: string) => {
    streamEndedRef.current = true;
    closeStreamResources();
    setStreamStatus("idle");
    setSubmitError({ type: "network", message });
    toast.error(t("common.error", "Transcription failed"));
  };

  function clearStreamResult(): void {
    setPartialTranscript("");
    setFinalTranscript("");
    setCurrentCaptionSegment(null);
    setCaptionSegments([]);
  }

  function closeStreamResources(): void {
    streamSessionRef.current += 1;
    stopRecordingResources();
    const socket = webSocketRef.current;
    if (socket && socket.readyState !== WebSocket.CLOSED) {
      streamEndedRef.current = true;
      socket.close();
    }
    webSocketRef.current = null;
  }

  function stopRecordingResources(): void {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
    mediaRecorderRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
  }

  const showValidationError = (message: string) => {
    setValidationError(message);
    toast.warning(message);
  };

  const handleCopyResult = async () => {
    try {
      await copyTextToClipboard(result);
      toast.success(t("common.copied", "Copied to clipboard!"));
    } catch {
      toast.error(t("common.error", "Failed to copy"));
    }
  };

  // Convert metadata notice parameters for common panel
  const parameterMetadataList = useMemo(() => {
    return asrParameterMetadata.map(param => ({
      name: param.name,
      label: t(`asr.metadata.${param.name}.0`),
      description: t(`asr.metadata.${param.name}.1`),
      required: param.required,
      supported: param.supported,
      notice: param.notice ? t(`asr.metadata.${param.name}.2`) : undefined,
      defaultValue: param.defaultValue,
      options: localizedAsrOptions(param.name, param.options, t)
    }));
  }, [t]);

  return (
    <div className="page animate-fade-in">
      <header className="page-header">
        <p className="card-eyebrow">{t("asr.kicker")}</p>
        <h2 className="page-title">{t("asr.title")}</h2>
        <p className="page-description">{t("asr.subtitle")}</p>
      </header>

      {validationError && (
        <Alert variant="warning" title={t("common.validationError")}>
          {validationError}
        </Alert>
      )}

      {submitError && (
        <Alert variant="danger" title={t("common.apiError")}>
          {submitError.message}
          {submitError.detail && (
            <span className="text-tertiary">
              {" "}
              {Object.entries(submitError.detail)
                .map(([name, value]) => `${name}: ${value}`)
                .join("; ")}
            </span>
          )}
        </Alert>
      )}

      <form onSubmit={handleSubmit} noValidate>
        <div
          className="grid gap-lg"
          style={{
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))"
          }}
        >
            {/* Main upload and configuration panel */}
            <Card variant="glass">
              <Card.Header eyebrow={t("asr.uploadPanelEyebrow")} title={t("asr.uploadPanelTitle")} />
              <Card.Body className="stack gap-md">
              <Tabs tabs={modeTabs} activeTab={mode} onChange={(id) => updateMode(id as AsrMode)} />
              <p className="text-xs text-muted mt-[-8px]">{t(`asr.modes.${mode}.1`, mode === "file" ? "Upload an audio file and run one transcription request." : "Use WebSocket streaming for live partial and final results.")}</p>

              {mode === "file" && (
                <FormField label={t("asr.metadata.file.0")} description={t("asr.fileDescription")}>
                  <FileDropZone
                    accept="audio/*"
                    selectedFile={file}
                    onFileSelect={handleFileSelect}
                    dropZoneText={t("asr.dropZoneText", "Drag & drop an audio file here, or click to select")}
                    dropZoneActiveText={t("asr.dropZoneActive", "Drop the audio file here...")}
                  />
                </FormField>
              )}

              <FormField label={t("asr.metadata.model.0")} description={t("asr.modelDescription")}>
                <SuggestionInput
                  id="asr-model"
                  value={form.model}
                  onChange={(val) => updateForm("model", val)}
                  suggestions={asrModelIds}
                  placeholder="Select or enter ASR model..."
                />
                <ModelStatus
                  models={asrModelIds}
                  isLoading={models.isLoading}
                  error={models.error}
                  kind="ASR"
                  listId="asr-model-suggestions"
                />
              </FormField>

              <div className="grid grid-cols-2 gap-md">
                <FormField label={t("asr.metadata.language.0")} description={t("asr.languageDescription")}>
                  <Select
                    id="asr-language"
                    name="language"
                    onChange={(event) => updateForm("language", event.target.value)}
                    value={form.language}
                  >
                    {asrLanguageOptions.map((language) => (
                      <option key={language || "auto"} value={language}>
                        {language || t("asr.autoDetect")}
                      </option>
                    ))}
                  </Select>
                </FormField>

                <FormField label={t("asr.metadata.response_format.0")} description={t("asr.responseFormatDescription")}>
                  <Select
                    id="asr-response-format"
                    name="response_format"
                    onChange={(event) => updateForm("responseFormat", event.target.value as AsrResponseFormat)}
                    value={form.responseFormat}
                  >
                    {responseFormatOptions.map((format) => (
                      <option key={format} value={format}>
                        {format}
                      </option>
                    ))}
                  </Select>
                </FormField>
              </div>

              {mode === "stream" && (
                <div className="stack gap-md">
                  <FormField
                    label={t("asr.stream.outputLabel", "Streaming output")}
                    description={
                      streamOutputMode === "caption"
                        ? t("asr.stream.outputCaptionDescription", "Server endpointing emits stable subtitle segments on one WebSocket session.")
                        : t("asr.stream.outputTranscriptDescription", "Continuous partial transcript events are followed by one final transcript after end.")
                    }
                  >
                    <Tabs tabs={streamOutputTabs} activeTab={streamOutputMode} onChange={(id) => updateStreamOutputMode(id as AsrStreamOutputMode)} />
                  </FormField>
                  {streamOutputMode === "caption" && (
                    <div className="result-block stack gap-sm">
                      <span className="card-eyebrow">{t("asr.stream.endpointingTitle", "Caption endpointing")}</span>
                      <p className="text-xs text-muted">
                        {t("asr.stream.endpointingDescription", "Tune how the server cuts stable subtitle segments. Values are sent only in Captions mode.")}
                      </p>
                      <div className="grid grid-cols-2 gap-md">
                        <FormField label={t("asr.stream.endpointingMinSpeech", "Min speech ms")} description={t("asr.stream.endpointingMinSpeechDescription", "Speech duration before a new segment can start.")}>
                          <Input
                            type="number"
                            min={1}
                            step={10}
                            inputMode="numeric"
                            disabled={isStreaming}
                            value={captionEndpointing.minSpeechMs}
                            onChange={(event) => updateCaptionEndpointing("minSpeechMs", event.target.value)}
                          />
                        </FormField>
                        <FormField label={t("asr.stream.endpointingMinSilence", "Min silence ms")} description={t("asr.stream.endpointingMinSilenceDescription", "Silence duration used to finalize a segment.")}>
                          <Input
                            type="number"
                            min={1}
                            step={10}
                            inputMode="numeric"
                            disabled={isStreaming}
                            value={captionEndpointing.minSilenceMs}
                            onChange={(event) => updateCaptionEndpointing("minSilenceMs", event.target.value)}
                          />
                        </FormField>
                        <FormField label={t("asr.stream.endpointingPadding", "Speech padding ms")} description={t("asr.stream.endpointingPaddingDescription", "Audio kept before speech starts.")}>
                          <Input
                            type="number"
                            min={0}
                            step={10}
                            inputMode="numeric"
                            disabled={isStreaming}
                            value={captionEndpointing.speechPaddingMs}
                            onChange={(event) => updateCaptionEndpointing("speechPaddingMs", event.target.value)}
                          />
                        </FormField>
                      </div>
                    </div>
                  )}
                  <Alert variant="info" title={t("asr.stream.noticeTitle", "Streaming API")}> 
                    {streamOutputMode === "caption"
                      ? t("asr.stream.noticeCaption", "Caption streaming returns partial current segments, stable segment_final events, and completed after end.")
                      : t("asr.stream.noticeTranscript", "Live transcript streaming returns partial text events and one final text event after end.")}
                  </Alert>
                  <div className="result-block stack gap-sm">
                    <span className="card-eyebrow">{t("asr.stream.protocolTitle", "WebSocket protocol")}</span>
                    <ul className="stack gap-xs text-sm list-disc pl-4 text-muted">
                      <li>{t("asr.stream.protocolEndpoint", "Endpoint: GET /v1/audio/transcriptions/stream with WebSocket upgrade.")}</li>
                      <li>
                        {streamOutputMode === "caption"
                          ? t("asr.stream.protocolStartCaption", "First message includes mode: \"caption\" plus model and input_audio_format.")
                          : t("asr.stream.protocolStartTranscript", "First message omits mode and includes model and input_audio_format.")}
                      </li>
                      <li>{t("asr.stream.protocolAudio", "Audio messages: binary chunks in the selected input_audio_format.")}</li>
                      <li>{t("asr.stream.protocolEnd", "End message: JSON { type: \"end\" } after the final audio chunk.")}</li>
                      <li>
                        {streamOutputMode === "caption"
                          ? t("asr.stream.protocolEventsCaption", "Server events: ready, partial with segment_id, segment_final, completed, or error.")
                          : t("asr.stream.protocolEventsTranscript", "Server events: ready, partial, final, or error; transcript events contain text only.")}
                      </li>
                    </ul>
                  </div>
                  <Tabs tabs={streamInputTabs} activeTab={streamInputMode} onChange={(id) => updateStreamInputMode(id as AsrStreamInputMode)} />
                  {streamInputMode === "file" ? (
                    <FormField label={t("asr.stream.fileLabel", "Streaming audio file")} description={t("asr.stream.fileDescription", "Supported streaming formats: wav, mp3, webm/opus, m4a, aac, flac, and ogg.")}>
                      <FileDropZone
                        accept=".wav,.mp3,.webm,.m4a,.aac,.flac,.ogg,.opus,audio/wav,audio/x-wav,audio/mpeg,audio/webm,audio/mp4,audio/aac,audio/flac,audio/ogg,audio/opus"
                        selectedFile={streamFile}
                        onFileSelect={handleStreamFileSelect}
                        dropZoneText={t("asr.stream.dropZoneText", "Select a supported audio file to stream")}
                        dropZoneActiveText={t("asr.dropZoneActive", "Drop the audio file here...")}
                      />
                    </FormField>
                  ) : (
                    <Alert variant="info" title={t("asr.stream.microphoneTitle", "Microphone recording")}> 
                      {t("asr.stream.microphoneDescription", "The browser records webm/opus chunks and sends them to the streaming endpoint.")}
                    </Alert>
                  )}
                </div>
              )}

              {mode === "stream" && (
                <FormField label={t("asr.metadata.prompt.0")} description={t("asr.promptDescription")}>
                  <TextArea
                    id="asr-prompt"
                    name="prompt"
                    onChange={(event) => updateForm("prompt", event.target.value)}
                    value={form.prompt}
                  />
                </FormField>
              )}

              {mode === "file" && (
                <FormField label={t("asr.metadata.timestamp_granularities.0")} description={t("asr.timestampDescription")}>
                  <Select
                    id="asr-timestamp-granularities"
                    name="timestamp_granularities"
                    onChange={(event) => updateForm("timestampGranularities", event.target.value === "" ? [] : [event.target.value])}
                    value={selectedTimestampGranularity(form.timestampGranularities)}
                  >
                    {asrTimestampGranularityOptions.map((granularity) => (
                      <option disabled={granularity === "word"} key={granularity || "none"} value={granularity}>
                        {timestampGranularityLabel(granularity, t)}
                      </option>
                    ))}
                  </Select>
                </FormField>
              )}

              <div className="pt-2 stack gap-sm">
                {mode === "file" ? (
                  <>
                    <Button
                      variant={isSubmitting ? "danger" : "primary"}
                      size="lg"
                      type={isSubmitting ? "button" : "submit"}
                      icon={isSubmitting ? <Square size={18} /> : <Play size={18} />}
                      fullWidth
                      onClick={isSubmitting ? handleCancelSubmit : undefined}
                    >
                      {isSubmitting ? t("common.cancel") : t("asr.submit")}
                    </Button>
                  </>
                ) : (
                  <Button
                    variant={isStreaming ? "danger" : "primary"}
                    size="lg"
                    type="button"
                    icon={isStreaming ? <Square size={18} /> : <Mic size={18} />}
                    fullWidth
                    onClick={isStreaming ? handleStopStream : handleStartStream}
                  >
                    {isStreaming ? t("asr.stream.stop", "Stop streaming") : t("asr.stream.start", "Start streaming")}
                  </Button>
                )}
              </div>
            </Card.Body>
          </Card>

          {/* Request summary preview panel */}
          <div className="stack gap-lg">
            <Card>
              <Card.Header eyebrow={t("asr.previewEyebrow")} title={t("asr.previewTitle")} />
              <Card.Body className="stack gap-md">
                {!file && (
                  <Alert variant="info" title={t("asr.previewNoticeTitle")}>
                    {t("asr.previewNotice")}
                  </Alert>
                )}
                <div className="result-block stack gap-sm">
                  <span className="card-eyebrow">{t("asr.summaryLabel")}</span>
                  <ul className="stack gap-xs text-sm list-disc pl-4 text-muted">
                    {(mode === "stream" ? streamRequestSummary : requestSummary).map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                </div>
                <CodePreview label={mode === "stream" ? t("asr.stream.connection", "Streaming connection") : t("common.curlPreview")}>
                  {mode === "stream" ? streamConnectionPreview : curlPreview}
                </CodePreview>
              </Card.Body>
            </Card>

            <MetadataPanel metadataList={parameterMetadataList} />
          </div>
        </div>
      </form>

      {mode === "stream" && (
        <Card variant="elevated">
          <Card.Header eyebrow={t("asr.stream.responseEyebrow", "Live result")} title={t("asr.stream.responseTitle", "Streaming transcription")} />
          <Card.Body className="stack gap-md">
            <div className="result-block stack gap-sm">
              <span className="card-eyebrow">{t("asr.stream.status", "Status")}</span>
              <p className="text-sm text-muted">{streamStatusLabel(streamStatus, t)}</p>
            </div>
            {streamOutputMode === "caption" ? (
              <>
                {currentCaptionSegment && (
                  <div className="result-block stack gap-sm">
                    <span className="card-eyebrow">{t("asr.stream.captionCurrent", "Current caption")}</span>
                    <p className="text-sm">{currentCaptionSegment.text}</p>
                  </div>
                )}
                {captionSegments.length > 0 && (
                  <div className="result-block stack gap-sm">
                    <span className="card-eyebrow">{t("asr.stream.captionSegments", "Stable captions")}</span>
                    <ol className="stack gap-sm text-sm pl-4">
                      {captionSegments.map((segment) => (
                        <li key={segment.id}>
                          <span className="text-tertiary">{captionSegmentTimeLabel(segment, t)}</span>
                          <p>{segment.text}</p>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
                {!currentCaptionSegment && captionSegments.length === 0 && (
                  <StateView
                    type="empty"
                    title={t("asr.stream.captionEmptyTitle", "No captions yet")}
                    description={t("asr.stream.captionEmpty", "Start recording or stream a file to see caption segments.")}
                  />
                )}
              </>
            ) : (
              <>
                {partialTranscript && (
                  <div className="result-block stack gap-sm">
                    <span className="card-eyebrow">{t("asr.stream.partial", "Partial")}</span>
                    <p className="text-sm">{partialTranscript}</p>
                  </div>
                )}
                {finalTranscript && (
                  <div className="result-block stack gap-sm">
                    <span className="card-eyebrow">{t("asr.stream.final", "Final")}</span>
                    <p className="text-sm">{finalTranscript}</p>
                  </div>
                )}
                {!partialTranscript && !finalTranscript && (
                  <StateView
                    type="empty"
                    title={t("asr.stream.emptyTitle", "No streaming transcript yet")}
                    description={t("asr.stream.empty", "Start recording or stream a file to see partial results.")}
                  />
                )}
              </>
            )}
          </Card.Body>
        </Card>
      )}

      {/* Response Panel */}
      <Card variant="elevated">
        <Card.Header eyebrow={t("asr.responseEyebrow")} title={t("asr.responseTitle")}>
          {result && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              icon={<Clipboard size={14} />}
              onClick={handleCopyResult}
            >
              {t("common.copy")}
            </Button>
          )}
        </Card.Header>
        <Card.Body>
          {result ? (
            <pre className="code-preview" style={{ maxHeight: "400px", overflow: "auto" }}>
              <code>{result}</code>
            </pre>
          ) : (
            <StateView
              type="empty"
              title={t("asr.resultEmptyTitle")}
              description={t("asr.resultEmpty")}
            />
          )}
        </Card.Body>
      </Card>
    </div>
  );
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function asrStateToForm(asr: PersistentAsrState): AsrFormState {
  return {
    model: asr.model,
    language: asr.language,
    responseFormat: asr.responseFormat,
    prompt: asr.prompt,
    temperature: asr.temperature,
    timestampGranularities: [...asr.timestampGranularities],
  };
}

function formToAsrState(form: AsrFormState): PersistentAsrState {
  return {
    model: form.model,
    language: form.language,
    responseFormat: form.responseFormat,
    prompt: form.prompt,
    temperature: form.temperature,
    timestampGranularities: [...form.timestampGranularities],
  };
}

function buildRequestInput(form: AsrFormState, file: File): AsrRequestInput {
  return {
    ...form,
    file,
  };
}

async function formatResponse(response: Response, responseFormat: AsrResponseFormat): Promise<string> {
  const text = await response.text();
  if (responseFormat === "text" || responseFormat === "srt") {
    return text;
  }

  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

function selectedTimestampGranularity(values: readonly string[]): string {
  return values.includes("segment") ? "segment" : "";
}

function isStreamingResponseFormat(format: AsrResponseFormat): boolean {
  return format === "json";
}

function streamStatusLabel(status: AsrStreamStatus, t: TFunction): string {
  switch (status) {
    case "connecting":
      return t("asr.stream.statusConnecting", "Connecting");
    case "streaming":
      return t("asr.stream.statusStreaming", "Streaming");
    case "finishing":
      return t("asr.stream.statusFinishing", "Finishing");
    case "idle":
      return t("asr.stream.statusIdle", "Idle");
  }
}

function upsertCaptionSegment(segments: CaptionSegmentView[], segment: CaptionSegmentView): CaptionSegmentView[] {
  return upsertBoundedAsrCaptionSegments(segments, segment);
}

function captionSegmentTimeLabel(segment: CaptionSegmentView, t: TFunction): string {
  if (segment.startMs === undefined || segment.endMs === undefined) {
    return t("asr.stream.captionNoTime", "Segment {{id}}", { id: segment.id });
  }
  return t("asr.stream.captionTime", "Segment {{id}} · {{start}}-{{end}} ms", {
    id: segment.id,
    start: segment.startMs,
    end: segment.endMs,
  });
}

function captionEndpointingSummary(endpointing: CaptionEndpointingFormState, t: TFunction): string {
  return t(
    "asr.stream.summary.endpointing",
    "Endpointing: speech {{minSpeech}} ms, silence {{minSilence}} ms, padding {{padding}} ms",
    {
      minSpeech: endpointing.minSpeechMs || "-",
      minSilence: endpointing.minSilenceMs || "-",
      padding: endpointing.speechPaddingMs || "-",
    },
  );
}

function parseCaptionEndpointing(
  endpointing: CaptionEndpointingFormState,
  t: TFunction,
): AsrCaptionEndpointingOptions | string {
  const minSpeechMs = parseIntegerMilliseconds(endpointing.minSpeechMs);
  const minSilenceMs = parseIntegerMilliseconds(endpointing.minSilenceMs);
  const speechPaddingMs = parseIntegerMilliseconds(endpointing.speechPaddingMs);

  if (
    minSpeechMs === null ||
    minSilenceMs === null ||
    speechPaddingMs === null ||
    minSpeechMs <= 0 ||
    minSilenceMs <= 0 ||
    speechPaddingMs < 0
  ) {
    return t(
      "asr.stream.endpointingInvalid",
      "Endpointing values must be whole milliseconds. Min speech and min silence must be greater than 0; speech padding can be 0 or greater.",
    );
  }
  const parsedEndpointing = {
    min_speech_ms: minSpeechMs,
    min_silence_ms: minSilenceMs,
    speech_padding_ms: speechPaddingMs,
  };
  const endpointingError = validateAsrCaptionEndpointingOptions(parsedEndpointing);
  if (endpointingError === "invalid_candidate_window") {
    return t("asr.stream.endpointingInvalidWindow", "Min speech ms plus speech padding ms must be 60000 ms or less.");
  }
  if (endpointingError === "invalid_rounded_window") {
    return t("asr.stream.endpointingInvalidRoundedWindow", "Min speech ms plus speech padding ms must cover the rounded 30 ms VAD frame window.");
  }
  if (endpointingError === "invalid") {
    return t(
      "asr.stream.endpointingInvalid",
      "Endpointing values must be whole milliseconds. Min speech and min silence must be greater than 0; speech padding can be 0 or greater.",
    );
  }

  return parsedEndpointing;
}

function parseIntegerMilliseconds(value: string): number | null {
  const trimmedValue = value.trim();
  if (!/^\d+$/.test(trimmedValue)) {
    return null;
  }
  const parsedValue = Number(trimmedValue);
  return Number.isSafeInteger(parsedValue) ? parsedValue : null;
}

function timestampGranularityLabel(granularity: string, t: TFunction): string {
  if (granularity === "") {
    return t("asr.timestampNone");
  }
  if (granularity === "word") {
    return t("asr.timestampWordUnsupported");
  }
  return granularity;
}

function localizedAsrOptions(name: string, options: readonly string[] | undefined, t: TFunction): string[] | undefined {
  if (!options) {
    return undefined;
  }
  if (name === "language") {
    return options.map((option) => option || t("asr.autoDetect"));
  }
  if (name === "timestamp_granularities") {
    return options.map((option) => timestampGranularityLabel(option, t));
  }
  return [...options];
}
