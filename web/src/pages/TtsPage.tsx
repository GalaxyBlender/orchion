import { type ChangeEvent, type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { ttsLanguageOptions, ttsModes, ttsParameterMetadata, ttsResponseFormats, ttsSpeakerOptions } from "@/features/tts/metadata";
import { buildTtsCurl, buildTtsPayload, summarizeTtsRequest } from "@/features/tts/request";
import type { TtsFormState, TtsMode, TtsResponseFormat, TtsRequestInput } from "@/features/tts/types";
import { useModels } from "@/features/models/useModels";
import { apiUrl, authHeaders } from "@/shared/api/client";
import { parseNetworkError, parseApiError } from "@/shared/api/errors";
import { ApiRequestError } from "@/shared/api/types";
import { buildApiError, readResponsePayload, type SubmissionError } from "@/shared/api/apiHelpers";
import {
  loadPersistentState,
  savePersistentState,
  type PersistentState,
  type PersistentTtsState,
} from "@/shared/storage/persistentState";
import { Card, FormField, Input, Select, TextArea, Button, Alert, StateView, ModelStatus, CodePreview, FileDropZone, useToast, MetadataPanel, Tabs, Slider, SuggestionInput } from "@/shared/ui";
import { Play, Download, Clipboard, Square } from "lucide-react";

const endpointPath = "/v1/audio/speech";
const previewReferenceAudioName = "reference-audio-placeholder.wav";

interface AudioResult {
  url: string;
  contentType: string;
  size: number;
  headers: Array<[string, string]>;
  fileName: string;
}

export function TtsPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const [persistentState, setPersistentState] = useState<PersistentState>(() => loadPersistentState());
  const [form, setForm] = useState<TtsFormState>(() => ttsStateToForm(persistentState.tts));
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);
  const [validationError, setValidationError] = useState("");
  const [submitError, setSubmitError] = useState<SubmissionError | null>(null);
  const [audioResult, setAudioResult] = useState<AudioResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const settings = persistentState.settings;
  const models = useModels(settings);
  const allTtsModelIds = useMemo(() => models.classified.tts.map((model) => model.id), [models.classified.tts]);
  const recommendedTtsModelIds = useMemo(() => {
    switch (form.mode) {
      case "preset":
        return models.classified.ttsPresetVoice.map((model) => model.id);
      case "clone":
        return models.classified.ttsVoiceClone.map((model) => model.id);
      case "design":
        return models.classified.ttsVoiceDesign.map((model) => model.id);
    }
  }, [form.mode, models.classified.ttsPresetVoice, models.classified.ttsVoiceClone, models.classified.ttsVoiceDesign]);
  const recommendedTtsModelSet = useMemo(() => new Set(recommendedTtsModelIds), [recommendedTtsModelIds]);
  
  const previewInput = useMemo(() => buildRequestInput(form, referenceAudio, true), [form, referenceAudio]);
  
  const requestSummary = useMemo(
    () =>
      summarizeTtsRequest(previewInput, {
        modeClone: t("tts.summary.modeClone"),
        modeDesign: t("tts.summary.modeDesign"),
        modePreset: t("tts.summary.modePreset"),
        model: (model) => t("tts.summary.model", { model }),
        format: (format) => t("tts.summary.format", { format }),
        language: (language) => t("tts.summary.language", { language: language === "" ? t("tts.autoLanguage") : language }),
        speaker: (speaker) => t("tts.summary.speaker", { speaker }),
        referenceAudio: (value) => t("tts.summary.referenceAudio", { value }),
        referenceText: (value) => t("tts.summary.referenceText", { value }),
        voicePrompt: (value) => t("tts.summary.voicePrompt", { value }),
        speed: (speed) => t("tts.summary.speed", { speed }),
        sampling: (seed, temperature, topK, topP, repetitionPenalty, maxLength) =>
          t("tts.summary.sampling", { seed, temperature, topK, topP, repetitionPenalty, maxLength }),
        constraints: t("tts.summary.constraints"),
        notSelected: t("tts.summary.notSelected"),
        omitted: t("tts.summary.omitted"),
        sent: t("tts.summary.sent"),
      }),
    [previewInput, t],
  );
  
  const curlPreview = useMemo(() => buildTtsCurl(settings, previewInput), [previewInput, settings]);

  useEffect(() => {
    return () => {
      if (audioResult) {
        URL.revokeObjectURL(audioResult.url);
      }
    };
  }, [audioResult]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const updateForm = <K extends keyof TtsFormState>(field: K, value: TtsFormState[K]) => {
    setValidationError("");
    setSubmitError(null);
    clearAudioResult();
    setForm((currentForm) => {
      const nextForm = { ...currentForm, [field]: value };
      setPersistentState((currentState) => {
        const nextState: PersistentState = {
          ...currentState,
          tts: formToTtsState(nextForm, currentState.tts),
        };
        savePersistentState(nextState);
        return nextState;
      });
      return nextForm;
    });
  };

  const updateMode = (mode: TtsMode) => {
    setValidationError("");
    setSubmitError(null);
    clearAudioResult();
    setPersistentState((currentState) => {
      const nextModel = modelForMode(currentState.tts, mode);
      const nextForm = { ...form, mode, model: nextModel };
      const nextState: PersistentState = {
        ...currentState,
        tts: formToTtsState(nextForm, currentState.tts),
      };
      savePersistentState(nextState);
      setForm(nextForm);
      return nextState;
    });
  };

  const updateModel = (model: string) => {
    updateForm("model", model);
  };

  const handleReferenceAudioSelect = (file: File | null) => {
    setValidationError("");
    setSubmitError(null);
    clearAudioResult();
    setReferenceAudio(file);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitError(null);
    clearAudioResult();

    const model = form.model.trim();
    const input = form.input.trim();
    const speaker = form.speaker.trim();
    const referenceText = form.referenceText.trim();
    const voicePrompt = form.voicePrompt.trim();

    if (model === "") {
      showValidationError(t("tts.missingModel"));
      return;
    }
    if (input === "") {
      showValidationError(t("tts.missingInput"));
      return;
    }
    if (form.mode === "preset" && speaker === "") {
      showValidationError(t("tts.missingSpeaker"));
      return;
    }
    if (form.mode === "clone" && !referenceAudio) {
      showValidationError(t("tts.missingReferenceAudio"));
      return;
    }
    if (form.mode === "clone" && referenceText === "") {
      showValidationError(t("tts.missingReferenceText"));
      return;
    }
    if (form.mode === "design" && voicePrompt === "") {
      showValidationError(t("tts.missingVoicePrompt"));
      return;
    }

    setValidationError("");
    setIsSubmitting(true);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const payload = buildTtsPayload(buildRequestInput({ ...form, model, input, speaker, referenceText, voicePrompt }, referenceAudio, false));
      const response = await fetch(apiUrl(settings, endpointPath), {
        method: "POST",
        headers: buildHeaders(payload.headers),
        body: payload.kind === "json" ? JSON.stringify(payload.body) : payload.formData,
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw await parseApiError(response, await readResponsePayload(response));
      }

      setAudioResult(await buildAudioResult(response, form.responseFormat));
      toast.success(t("common.success", "Synthesis completed successfully!"));
    } catch (caughtError) {
      if (isAbortError(caughtError)) {
        toast.info(t("tts.cancelled"));
        return;
      }
      setSubmitError(buildApiError(caughtError));
      toast.error(t("common.error", "Synthesis failed"));
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

  const showValidationError = (message: string) => {
    setValidationError(message);
    toast.warning(message);
  };

  function clearAudioResult(): void {
    setAudioResult(null);
  }

  function buildHeaders(payloadHeaders: HeadersInit): Headers {
    const headers = new Headers(authHeaders(settings));
    new Headers(payloadHeaders).forEach((value, key) => headers.set(key, value));
    return headers;
  }

  // Convert metadata notice parameters for common panel
  const parameterMetadataList = useMemo(() => {
    return ttsParameterMetadata.map(param => ({
      name: param.name,
      label: t(`tts.metadata.${param.name}.0`),
      description: t(`tts.metadata.${param.name}.1`),
      required: param.required,
      supported: param.supported,
      notice: param.notice ? t(`tts.metadata.${param.name}.2`) : undefined,
      defaultValue: param.defaultValue,
      options: localizedTtsOptions(param.name, param.options, t)
    }));
  }, [t]);

  const modeTabs = useMemo(() => [
    { id: "preset", label: t("tts.modes.preset.0") },
    { id: "clone", label: t("tts.modes.clone.0") },
    { id: "design", label: t("tts.modes.design.0") }
  ], [t]);

  return (
    <div className="page animate-fade-in">
      <header className="page-header">
        <p className="card-eyebrow">{t("tts.kicker")}</p>
        <h2 className="page-title">{t("tts.title")}</h2>
        <p className="page-description">{t("tts.subtitle")}</p>
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
          {/* Main TTS Form panel */}
          <Card variant="glass">
            <Card.Header eyebrow={t("tts.panelEyebrow")} title={t("tts.panelTitle")} />
            <Card.Body className="stack gap-md">
              <Tabs
                tabs={modeTabs}
                activeTab={form.mode}
                onChange={(mode) => updateMode(mode as TtsMode)}
              />
              <p className="text-xs text-muted mt-[-8px]">{t(`tts.modes.${form.mode}.1`)}</p>

              <FormField label={t("tts.metadata.model.0")} description={t("tts.modelDescription")}>
                <SuggestionInput
                  id="tts-model"
                  value={form.model}
                  onChange={updateModel}
                  suggestions={allTtsModelIds}
                  isSuggestionRecommended={(modelId) => recommendedTtsModelSet.has(modelId)}
                  placeholder="Select or enter TTS model..."
                />
                <ModelStatus
                  models={recommendedTtsModelIds}
                  isLoading={models.isLoading}
                  error={models.error}
                  kind="TTS"
                  listId="tts-model-suggestions"
                />
              </FormField>

              <FormField label={t("tts.metadata.input.0")} description={t("tts.inputDescription")}>
                <TextArea
                  id="tts-input"
                  name="input"
                  onChange={(event) => updateForm("input", event.target.value)}
                  value={form.input}
                  aria-required="true"
                />
              </FormField>

              <div className="grid grid-cols-2 gap-md">
                <FormField label={t("tts.metadata.language.0")} description={t("tts.languageDescription")}>
                  <Select
                    id="tts-language"
                    name="language"
                    onChange={(event) => updateForm("language", event.target.value)}
                    value={form.language}
                  >
                    {ttsLanguageOptions.map((language) => (
                      <option key={language || "auto"} value={language}>
                        {language || t("tts.autoLanguage")}
                      </option>
                    ))}
                  </Select>
                </FormField>

                <FormField label={t("tts.metadata.response_format.0")} description={t("tts.responseFormatDescription")}>
                  <Select
                    id="tts-response-format"
                    name="response_format"
                    onChange={(event) => updateForm("responseFormat", event.target.value as TtsResponseFormat)}
                    value={form.responseFormat}
                  >
                    {ttsResponseFormats.map((format) => (
                      <option key={format} value={format}>
                        {format}
                      </option>
                    ))}
                  </Select>
                </FormField>
              </div>

              {/* Mode-specific Fields */}
              {form.mode === "preset" && (
                <FormField label={t("tts.metadata.speaker.0")} description={t("tts.speakerDescription")}>
                  <Select
                    id="tts-speaker"
                    name="speaker"
                    onChange={(event) => updateForm("speaker", event.target.value)}
                    value={form.speaker}
                  >
                    <option value="">{t("tts.selectSpeaker")}</option>
                    {ttsSpeakerOptions.map((speaker) => (
                      <option key={speaker.value} value={speaker.value}>
                        {speaker.label}
                      </option>
                    ))}
                  </Select>
                </FormField>
              )}

              {form.mode === "clone" && (
                <div className="stack gap-md">
                  <FormField label={t("tts.metadata.reference_audio.0")} description={t("tts.referenceAudioDescription")}>
                    <FileDropZone
                      accept="audio/*"
                      selectedFile={referenceAudio}
                      onFileSelect={handleReferenceAudioSelect}
                      dropZoneText={t("tts.dropZoneText", "Drag & drop reference audio here, or click to select")}
                      dropZoneActiveText={t("tts.dropZoneActive", "Drop the reference audio here...")}
                    />
                  </FormField>

                  <FormField label={t("tts.metadata.reference_text.0")} description={t("tts.referenceTextDescription")}>
                    <TextArea
                      id="tts-reference-text"
                      name="reference_text"
                      onChange={(event) => updateForm("referenceText", event.target.value)}
                      value={form.referenceText}
                      aria-required="true"
                    />
                  </FormField>
                </div>
              )}

              {form.mode === "design" && (
                <FormField label={t("tts.metadata.voice_prompt.0")} description={t("tts.voicePromptDescription")}>
                  <TextArea
                    id="tts-voice-prompt"
                    name="voice_prompt"
                    onChange={(event) => updateForm("voicePrompt", event.target.value)}
                    value={form.voicePrompt}
                    aria-required="true"
                  />
                </FormField>
              )}

              {/* Parameters Slider & Inputs */}
              <div className="grid grid-cols-2 gap-md border-t border-subtle pt-4" style={{ borderTop: "1px solid var(--color-border-subtle)", paddingTop: "var(--space-4)" }}>
                <FormField label={t("tts.metadata.speed.0")} description={t("tts.speedDescription")}>
                  <Slider
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    value={parseFloat(form.speed) || 1.0}
                    onChange={(val) => updateForm("speed", String(val))}
                  />
                </FormField>

                <FormField label={t("tts.metadata.seed.0")} description={t("tts.seedDescription")}>
                  <Input
                    id="tts-seed"
                    inputMode="numeric"
                    name="seed"
                    onChange={(event) => updateForm("seed", event.target.value)}
                    value={form.seed}
                  />
                </FormField>

                <FormField label={t("tts.metadata.temperature.0")} description={t("tts.temperatureDescription")}>
                  <Input
                    id="tts-temperature"
                    inputMode="decimal"
                    name="temperature"
                    onChange={(event) => updateForm("temperature", event.target.value)}
                    value={form.temperature}
                  />
                </FormField>

                <FormField label={t("tts.metadata.top_k.0")} description={t("tts.topKDescription")}>
                  <Input
                    id="tts-top-k"
                    inputMode="numeric"
                    name="top_k"
                    onChange={(event) => updateForm("topK", event.target.value)}
                    value={form.topK}
                  />
                </FormField>

                <FormField label={t("tts.metadata.top_p.0")} description={t("tts.topPDescription")}>
                  <Input
                    id="tts-top-p"
                    inputMode="decimal"
                    name="top_p"
                    onChange={(event) => updateForm("topP", event.target.value)}
                    value={form.topP}
                  />
                </FormField>

                <FormField label={t("tts.metadata.repetition_penalty.0")} description={t("tts.repetitionPenaltyDescription")}>
                  <Input
                    id="tts-repetition-penalty"
                    inputMode="decimal"
                    name="repetition_penalty"
                    onChange={(event) => updateForm("repetitionPenalty", event.target.value)}
                    value={form.repetitionPenalty}
                  />
                </FormField>

                <FormField label={t("tts.metadata.max_length.0")} description={t("tts.maxLengthDescription")} className="grid-cols-span-2">
                  <Input
                    id="tts-max-length"
                    inputMode="numeric"
                    name="max_length"
                    onChange={(event) => updateForm("maxLength", event.target.value)}
                    value={form.maxLength}
                  />
                </FormField>
              </div>

              <div className="pt-2 stack gap-sm">
                <Button
                  variant={isSubmitting ? "danger" : "primary"}
                  size="lg"
                  type={isSubmitting ? "button" : "submit"}
                  icon={isSubmitting ? <Square size={18} /> : <Play size={18} />}
                  fullWidth
                  onClick={isSubmitting ? handleCancelSubmit : undefined}
                >
                  {isSubmitting ? t("common.cancel") : t("tts.submit")}
                </Button>
              </div>
            </Card.Body>
          </Card>

          {/* Request Preview & Notes */}
          <div className="stack gap-lg">
            <Card>
              <Card.Header eyebrow={t("tts.previewEyebrow")} title={t("tts.previewTitle")} />
              <Card.Body className="stack gap-md">
                {form.mode === "clone" && !referenceAudio && (
                  <Alert variant="info" title={t("tts.previewNoticeTitle")}>
                    {t("tts.previewNotice", { name: previewReferenceAudioName })}
                  </Alert>
                )}
                <div className="result-block stack gap-sm">
                  <span className="card-eyebrow">{t("tts.summaryLabel")}</span>
                  <ul className="stack gap-xs text-sm list-disc pl-4 text-muted">
                    {requestSummary.map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                </div>
                <CodePreview label={t("common.curlPreview")}>{curlPreview}</CodePreview>
              </Card.Body>
            </Card>

            <MetadataPanel metadataList={parameterMetadataList} />
          </div>
        </div>
      </form>

      {/* Response Audio Panel */}
      <Card variant="elevated">
        <Card.Header eyebrow={t("tts.responseEyebrow")} title={t("tts.responseTitle")} />
        <Card.Body>
          {audioResult ? (
            <div className="stack gap-md animate-fade-in">
              <Alert variant="success" title={t("tts.audioReady")}>
                {t("tts.audioNotice", { contentType: audioResult.contentType, size: audioResult.size })}
              </Alert>
              
              <div
                className="hstack justify-center p-6 bg-sunken rounded-lg"
                style={{
                  background: "var(--color-bg-sunken)",
                  padding: "var(--space-6)",
                  borderRadius: "var(--radius-lg)",
                  display: "flex",
                  justifyContent: "center"
                }}
              >
                <audio controls src={audioResult.url} style={{ width: "100%", maxWidth: "500px" }}>
                  <track kind="captions" />
                </audio>
              </div>

              <div className="hstack gap-sm flex-wrap">
                <a
                  className="btn btn-primary"
                  download={audioResult.fileName}
                  href={audioResult.url}
                  style={{ textDecoration: "none" }}
                >
                  <Download size={16} style={{ marginRight: "8px" }} />
                  {t("tts.download")}
                </a>
              </div>

              <div className="result-block stack gap-sm">
                <span className="card-eyebrow">{t("tts.headers")}</span>
                <ul className="stack gap-xs text-xs text-mono text-muted pl-4 list-disc">
                  {audioResult.headers.map(([name, value]) => (
                    <li key={name}>{`${name}: ${value}`}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <StateView
              type="empty"
              title={t("tts.resultEmptyTitle")}
              description={t("tts.resultEmpty")}
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

function ttsStateToForm(tts: PersistentTtsState): TtsFormState {
  return {
    mode: tts.mode,
    model: modelForMode(tts, tts.mode),
    input: tts.input,
    language: tts.language,
    responseFormat: tts.responseFormat,
    speaker: normalizeSpeakerValue(tts.speaker),
    referenceAudioName: "",
    referenceText: tts.referenceText,
    voicePrompt: tts.voicePrompt,
    speed: tts.speed,
    seed: tts.seed,
    temperature: tts.temperature,
    topK: tts.topK,
    topP: tts.topP,
    repetitionPenalty: tts.repetitionPenalty,
    maxLength: tts.maxLength,
  };
}

function formToTtsState(form: TtsFormState, previous: PersistentTtsState): PersistentTtsState {
  const models = {
    ...previous.models,
    [form.mode]: form.model,
  };

  return {
    mode: form.mode,
    model: form.model,
    models,
    input: form.input,
    language: form.language,
    responseFormat: form.responseFormat,
    speaker: form.speaker,
    referenceText: form.referenceText,
    voicePrompt: form.voicePrompt,
    speed: form.speed,
    seed: form.seed,
    temperature: form.temperature,
    topK: form.topK,
    topP: form.topP,
    repetitionPenalty: form.repetitionPenalty,
    maxLength: form.maxLength,
  };
}

function modelForMode(tts: PersistentTtsState, mode: TtsMode): string {
  return tts.models[mode] || (tts.mode === mode ? tts.model : "");
}

function buildRequestInput(form: TtsFormState, referenceAudio: File | null, usePreviewPlaceholder: boolean): TtsRequestInput {
  return {
    ...form,
    referenceAudio,
    referenceAudioName: referenceAudio?.name ?? (usePreviewPlaceholder && form.mode === "clone" ? previewReferenceAudioName : ""),
  };
}

function normalizeSpeakerValue(speaker: string): string {
  const normalizedSpeaker = speaker.trim().toLowerCase();
  const speakerOption = ttsSpeakerOptions.find(
    (option) => option.value.toLowerCase() === normalizedSpeaker || option.label.toLowerCase() === normalizedSpeaker,
  );
  return speakerOption?.value ?? speaker;
}

function localizedTtsOptions(name: string, options: readonly string[] | undefined, t: TFunction): string[] | undefined {
  if (!options) {
    return undefined;
  }
  if (name === "language") {
    return options.map((option) => option || t("tts.autoLanguage"));
  }
  return [...options];
}

async function buildAudioResult(response: Response, responseFormat: TtsResponseFormat): Promise<AudioResult> {
  const blob = await response.blob();
  const contentType = (response.headers.get("Content-Type") ?? blob.type) || "unknown";

  return {
    url: URL.createObjectURL(blob),
    contentType,
    size: blob.size,
    headers: Array.from(response.headers.entries()),
    fileName: `speech.${responseFormat}`,
  };
}
