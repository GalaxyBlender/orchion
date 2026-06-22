import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ocrParameterMetadata, ocrResponseFormats, ocrTaskOptions } from "@/features/ocr/metadata";
import { buildOcrCurl, buildOcrFormData, summarizeOcrRequest } from "@/features/ocr/request";
import type { OcrFormState, OcrRequestInput, OcrResponseFormat, OcrTask } from "@/features/ocr/types";
import { useModels } from "@/features/models/useModels";
import { apiUrl, authHeaders } from "@/shared/api/client";
import { buildApiError, readResponsePayload, type SubmissionError } from "@/shared/api/apiHelpers";
import { parseApiError } from "@/shared/api/errors";
import { copyTextToClipboard } from "@/shared/clipboard";
import {
  loadPersistentState,
  savePersistentState,
  type PersistentOcrState,
  type PersistentState,
} from "@/shared/storage/persistentState";
import { Alert, Button, Card, CodePreview, FileDropZone, FormField, Input, MetadataPanel, ModelStatus, Select, StateView, SuggestionInput, useToast } from "@/shared/ui";
import { Clipboard, FileText, Play } from "lucide-react";

const endpointPath = "/v1/ocr";
const previewFile = new File([""], "preview-document-placeholder.png", { type: "image/png" });

export function OcrPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const [persistentState, setPersistentState] = useState<PersistentState>(() => loadPersistentState());
  const [form, setForm] = useState<OcrFormState>(() => ocrStateToForm(persistentState.ocr));
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState("");
  const [submitError, setSubmitError] = useState<SubmissionError | null>(null);
  const [result, setResult] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const settings = persistentState.settings;
  const models = useModels(settings);
  const ocrModelIds = models.classified.ocr
    .map((model) => model.id)
    .filter((modelId) => !isLayoutModelId(modelId));
  const layoutModelIds = useMemo(
    () => models.classified.ocr.map((model) => model.id).filter(isLayoutModelId),
    [models.classified.ocr],
  );

  const previewInput = useMemo(() => buildRequestInput(form, file ?? previewFile), [file, form]);
  const requestSummary = useMemo(
    () => summarizeOcrRequest(previewInput, {
      model: (model) => t("ocr.summary.model", { model }),
      file: (selectedFile) => t("ocr.summary.file", { file: selectedFile }),
      responseFormat: (format) => t("ocr.summary.responseFormat", { format }),
      task: (task) => t("ocr.summary.task", { task }),
      layoutModel: (model) => t("ocr.summary.layoutModel", { model }),
      maxTokens: (value) => t("ocr.summary.maxTokens", { value }),
    }),
    [previewInput, t],
  );
  const curlPreview = useMemo(() => buildOcrCurl(settings, previewInput), [previewInput, settings]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const updateForm = <K extends keyof OcrFormState>(field: K, value: OcrFormState[K]) => {
    setValidationError("");
    setSubmitError(null);
    setResult("");
    setForm((currentForm) => {
      const nextForm = { ...currentForm, [field]: value };
      setPersistentState((currentState) => {
        const nextState: PersistentState = {
          ...currentState,
          ocr: formToOcrState(nextForm),
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
      showValidationError(t("ocr.missingFile"));
      return;
    }
    if (model === "") {
      showValidationError(t("ocr.missingModel"));
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
        body: buildOcrFormData(buildRequestInput({ ...form, model }, file)),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw parseApiError(response, await readResponsePayload(response));
      }

      setResult(await formatResponse(response, form.responseFormat));
      toast.success(t("common.success", "OCR completed successfully!"));
    } catch (caughtError) {
      if (isAbortError(caughtError)) {
        toast.info(t("ocr.cancelled"));
        return;
      }
      setSubmitError(buildApiError(caughtError));
      toast.error(t("common.error", "OCR failed"));
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
      await copyTextToClipboard(result);
      toast.success(t("common.copied", "Copied to clipboard!"));
    } catch {
      toast.error(t("common.error", "Failed to copy"));
    }
  };

  const parameterMetadataList = useMemo(() => {
    return ocrParameterMetadata.map((param) => ({
      name: param.name,
      label: t(`ocr.metadata.${param.name}.0`),
      description: t(`ocr.metadata.${param.name}.1`),
      required: param.required,
      supported: param.supported,
      notice: param.notice ? t(`ocr.metadata.${param.name}.2`) : undefined,
      defaultValue: param.defaultValue,
      options: param.options ? [...param.options] : undefined,
    }));
  }, [t]);

  return (
    <div className="page animate-fade-in">
      <header className="page-header">
        <p className="card-eyebrow">{t("ocr.kicker")}</p>
        <h2 className="page-title">{t("ocr.title")}</h2>
        <p className="page-description">{t("ocr.subtitle")}</p>
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
        <div className="grid gap-lg" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
          <Card variant="glass">
            <Card.Header eyebrow={t("ocr.panelEyebrow")} title={t("ocr.panelTitle")} />
            <Card.Body className="stack gap-md">
              <FormField label={t("ocr.metadata.file.0")} description={t("ocr.fileDescription")}>
                <FileDropZone
                  accept="image/*,.pdf"
                  selectedFile={file}
                  onFileSelect={handleFileSelect}
                  dropZoneText={t("ocr.dropZoneText")}
                  dropZoneActiveText={t("ocr.dropZoneActive")}
                />
              </FormField>

              <FormField label={t("ocr.metadata.model.0")} description={t("ocr.modelDescription")}>
                <SuggestionInput
                  id="ocr-model"
                  value={form.model}
                  onChange={(value) => updateForm("model", value)}
                  suggestions={ocrModelIds}
                  placeholder={t("ocr.modelPlaceholder")}
                />
                <ModelStatus
                  models={ocrModelIds}
                  isLoading={models.isLoading}
                  error={models.error}
                  kind="OCR"
                  listId="ocr-model-suggestions"
                />
              </FormField>

              <div className="grid grid-cols-2 gap-md">
                <FormField label={t("ocr.metadata.response_format.0")} description={t("ocr.responseFormatDescription")}>
                  <Select
                    id="ocr-response-format"
                    name="response_format"
                    onChange={(event) => updateForm("responseFormat", event.target.value as OcrResponseFormat)}
                    value={form.responseFormat}
                  >
                    {ocrResponseFormats.map((format) => (
                      <option key={format} value={format}>
                        {format}
                      </option>
                    ))}
                  </Select>
                </FormField>

                <FormField label={t("ocr.metadata.task.0")} description={t("ocr.taskDescription")}>
                  <Select
                    id="ocr-task"
                    name="task"
                    onChange={(event) => updateForm("task", event.target.value as OcrTask)}
                    value={form.task}
                  >
                    {ocrTaskOptions.map((task) => (
                      <option key={task} value={task}>
                        {task}
                      </option>
                    ))}
                  </Select>
                </FormField>
              </div>

              <FormField label={t("ocr.metadata.layout_model.0")} description={t("ocr.layoutModelDescription")}>
                <Select
                  id="ocr-layout-model"
                  name="layout_model"
                  onChange={(event) => updateForm("layoutModel", event.target.value)}
                  value={form.layoutModel}
                >
                  <option value="">{t("ocr.layoutModelPlaceholder")}</option>
                  {layoutModelIds.map((modelId) => (
                    <option key={modelId} value={modelId}>
                      {modelId}
                    </option>
                  ))}
                </Select>
              </FormField>

              <FormField label={t("ocr.metadata.max_tokens.0")} description={t("ocr.maxTokensDescription")}>
                <Input
                  id="ocr-max-tokens"
                  inputMode="numeric"
                  name="max_tokens"
                  onChange={(event) => updateForm("maxTokens", event.target.value)}
                  placeholder={t("ocr.maxTokensPlaceholder")}
                  value={form.maxTokens}
                />
              </FormField>

              <div className="hstack gap-sm flex-wrap">
                <Button type="submit" loading={isSubmitting} icon={<Play size={16} />} iconPosition="left">
                  {isSubmitting ? t("ocr.submitting") : t("ocr.submit")}
                </Button>
                {isSubmitting && (
                  <Button type="button" variant="secondary" onClick={handleCancelSubmit}>
                    {t("common.cancel")}
                  </Button>
                )}
              </div>
            </Card.Body>
          </Card>

          <div className="stack gap-md">
            <Card>
              <Card.Header eyebrow={t("ocr.previewEyebrow")} title={t("ocr.previewTitle")} />
              <Card.Body className="stack gap-md">
                {!file && (
                  <Alert variant="info" title={t("ocr.previewNoticeTitle")}>
                    {t("ocr.previewNotice")}
                  </Alert>
                )}
                <div className="result-block stack gap-sm">
                  <span className="card-eyebrow">{t("ocr.summaryLabel")}</span>
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

      <Card variant="elevated">
        <Card.Header eyebrow={t("ocr.responseEyebrow")} title={t("ocr.responseTitle")}>
          {result !== "" && (
            <Button type="button" variant="ghost" size="sm" onClick={handleCopyResult} icon={<Clipboard size={14} />} iconPosition="left">
              {t("common.copy")}
            </Button>
          )}
        </Card.Header>
        <Card.Body>
          {result === "" ? (
            <StateView
              type="empty"
              title={t("ocr.resultEmptyTitle")}
              description={t("ocr.resultEmpty")}
            />
          ) : (
            <div className="stack gap-sm">
              <span className="card-eyebrow">{t("ocr.resultLabel")}</span>
              {form.responseFormat === "json" ? (
                <pre className="code-preview"><code>{result}</code></pre>
              ) : (
                <div className="result-block stack gap-sm">
                  <FileText size={18} className="text-accent" />
                  <pre className="whitespace-pre-wrap text-sm text-primary">{result}</pre>
                </div>
              )}
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
}

function buildRequestInput(form: OcrFormState, selectedFile: File): OcrRequestInput {
  return { ...form, file: selectedFile };
}

function isLayoutModelId(modelId: string): boolean {
  return modelId.toLowerCase().includes("doclayout");
}

function ocrStateToForm(state: PersistentOcrState): OcrFormState {
  return {
    model: state.model,
    responseFormat: state.responseFormat,
    task: state.task,
    layoutModel: "",
    maxTokens: state.maxTokens,
  };
}

function formToOcrState(form: OcrFormState): PersistentOcrState {
  return {
    model: form.model,
    responseFormat: form.responseFormat,
    task: form.task,
    layoutModel: form.layoutModel,
    maxTokens: form.maxTokens,
  };
}

async function formatResponse(response: Response, responseFormat: OcrResponseFormat): Promise<string> {
  if (responseFormat === "json") {
    try {
      return JSON.stringify(await response.clone().json(), null, 2);
    } catch {
      return response.text();
    }
  }

  return response.text();
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
