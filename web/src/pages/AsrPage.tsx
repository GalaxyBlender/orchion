import { type ChangeEvent, type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { asrLanguageOptions, asrParameterMetadata, asrResponseFormats, asrTimestampGranularityOptions } from "@/features/asr/metadata";
import { buildAsrCurl, buildAsrFormData, summarizeAsrRequest } from "@/features/asr/request";
import type { AsrFormState, AsrRequestInput, AsrResponseFormat } from "@/features/asr/types";
import { useModels } from "@/features/models/useModels";
import { apiUrl, authHeaders } from "@/shared/api/client";
import { parseNetworkError, parseApiError } from "@/shared/api/errors";
import { ApiRequestError } from "@/shared/api/types";
import { buildApiError, readResponsePayload, type SubmissionError } from "@/shared/api/apiHelpers";
import {
  loadPersistentState,
  savePersistentState,
  type PersistentAsrState,
  type PersistentState,
} from "@/shared/storage/persistentState";
import { Card, FormField, Input, Select, TextArea, Button, Alert, StateView, ModelStatus, CodePreview, FileDropZone, useToast, MetadataPanel, SuggestionInput } from "@/shared/ui";
import { Sparkles, Play, Clipboard } from "lucide-react";

const endpointPath = "/v1/audio/transcriptions";
const previewFile = new File([""], "preview-audio-placeholder.wav", { type: "audio/wav" });

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
  const abortControllerRef = useRef<AbortController | null>(null);
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
        prompt: t("asr.summary.prompt"),
        temperature: t("asr.summary.temperature"),
        timestamp: (values) => t("asr.summary.timestamp", { values }),
      }),
    [previewInput, t],
  );
  
  const curlPreview = useMemo(() => buildAsrCurl(settings, previewInput), [previewInput, settings]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

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

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
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

  const showValidationError = (message: string) => {
    setValidationError(message);
    toast.warning(message);
  };

  const handleCopyResult = async () => {
    try {
      await navigator.clipboard.writeText(result);
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
              <FormField label={t("asr.metadata.file.0")} description={t("asr.fileDescription")}>
                <FileDropZone
                  accept="audio/*"
                  selectedFile={file}
                  onFileSelect={handleFileSelect}
                  dropZoneText={t("asr.dropZoneText", "Drag & drop an audio file here, or click to select")}
                  dropZoneActiveText={t("asr.dropZoneActive", "Drop the audio file here...")}
                />
              </FormField>

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
                    {asrResponseFormats.map((format) => (
                      <option key={format} value={format}>
                        {format}
                      </option>
                    ))}
                  </Select>
                </FormField>
              </div>

              <FormField label={t("asr.metadata.prompt.0")} description={t("asr.promptDescription")}>
                <TextArea
                  id="asr-prompt"
                  name="prompt"
                  onChange={(event) => updateForm("prompt", event.target.value)}
                  value={form.prompt}
                />
              </FormField>

              <div className="grid grid-cols-2 gap-md">
                <FormField label={t("asr.metadata.temperature.0")} description={t("asr.temperatureDescription")}>
                  <Input
                    id="asr-temperature"
                    inputMode="decimal"
                    name="temperature"
                    onChange={(event) => updateForm("temperature", event.target.value)}
                    value={form.temperature}
                  />
                </FormField>

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
              </div>

              <div className="pt-2 stack gap-sm">
                <Button
                  variant="primary"
                  size="lg"
                  type="submit"
                  loading={isSubmitting}
                  icon={<Play size={18} />}
                  fullWidth
                >
                  {t("asr.submit")}
                </Button>
                {isSubmitting && (
                  <Button
                    variant="danger"
                    size="md"
                    type="button"
                    className="btn-cancel-request"
                    onClick={handleCancelSubmit}
                  >
                    {t("common.cancel")}
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

      {/* Response Panel */}
      <Card variant="elevated">
        <Card.Header eyebrow={t("asr.responseEyebrow")} title={t("asr.responseTitle")}>
          {result && (
            <Button
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
