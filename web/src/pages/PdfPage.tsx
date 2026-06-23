import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { pdfImageFormats, pdfParameterMetadata } from "@/features/pdf/metadata";
import { buildPdfCurl, buildPdfFormData, pdfImagesEndpoint, summarizePdfRequest } from "@/features/pdf/request";
import type { PdfFormState, PdfImageFormat, PdfRequestInput, PdfResult } from "@/features/pdf/types";
import { readZipWithImagePreviews, revokePreviewImages } from "@/features/pdf/zipPreview";
import { apiUrl, authHeaders } from "@/shared/api/client";
import { buildApiError, readResponsePayload, type SubmissionError } from "@/shared/api/apiHelpers";
import { parseApiError } from "@/shared/api/errors";
import { loadPersistentState, type PersistentState } from "@/shared/storage/persistentState";
import { Alert, Button, Card, CodePreview, FileDropZone, FormField, Input, MetadataPanel, Select, StateView, useToast } from "@/shared/ui";
import { Download, FileArchive, Play, Square } from "lucide-react";

const previewFile = new File([""], "preview-document-placeholder.pdf", { type: "application/pdf" });

const initialFormState: PdfFormState = {
  file: null,
  responseFormat: "png",
  pages: "",
  scale: "1.0",
};

export function PdfPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const [persistentState] = useState<PersistentState>(() => loadPersistentState());
  const [form, setForm] = useState<PdfFormState>(initialFormState);
  const [validationError, setValidationError] = useState("");
  const [submitError, setSubmitError] = useState<SubmissionError | null>(null);
  const [result, setResult] = useState<PdfResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const resultRef = useRef<PdfResult | null>(null);
  const settings = persistentState.settings;

  const previewInput = useMemo(() => buildRequestInput(form, form.file ?? previewFile), [form]);
  const requestSummary = useMemo(() => summarizePdfRequest(previewInput), [previewInput]);
  const curlPreview = useMemo(() => buildPdfCurl(settings, previewInput), [previewInput, settings]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      if (resultRef.current) {
        revokePreviewImages(resultRef.current.images);
      }
    };
  }, []);

  const updateForm = <K extends keyof PdfFormState>(field: K, value: PdfFormState[K]) => {
    setValidationError("");
    setSubmitError(null);
    clearResult();
    setForm((currentForm) => ({ ...currentForm, [field]: value }));
  };

  const handleFileSelect = (selectedFile: File | null) => {
    updateForm("file", selectedFile);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitError(null);
    clearResult();

    if (!form.file) {
      showValidationError(t("pdf.missingFile", "Select a PDF file before converting."));
      return;
    }

    setValidationError("");
    setIsSubmitting(true);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const response = await fetch(apiUrl(settings, pdfImagesEndpoint), {
        method: "POST",
        headers: authHeaders(settings),
        body: buildPdfFormData(buildRequestInput(form, form.file)),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw parseApiError(response, await readResponsePayload(response));
      }

      setPdfResult(buildPendingPdfResult(response));
      setPdfResult(await buildPdfResult(response, (image) => {
        setPdfResult((currentResult) => currentResult ? {
          ...currentResult,
          images: [...currentResult.images, image],
        } : currentResult);
      }));
      toast.success(t("pdf.completed", "PDF images are ready."));
    } catch (caughtError) {
      if (isAbortError(caughtError)) {
        toast.info(t("pdf.cancelled", "PDF conversion cancelled."));
        return;
      }
      setSubmitError(buildApiError(caughtError));
      toast.error(t("pdf.failed", "PDF conversion failed"));
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

  const handleDownload = () => {
    if (!result) {
      return;
    }

    const objectUrl = URL.createObjectURL(result.blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = result.fileName;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(objectUrl);
  };

  const clearResult = () => {
    setPdfResult((currentResult) => {
      if (currentResult) {
        revokePreviewImages(currentResult.images);
      }
      return null;
    });
  };

  const setPdfResult = (nextResult: PdfResult | null | ((currentResult: PdfResult | null) => PdfResult | null)) => {
    setResult((currentResult) => {
      const resolvedResult = typeof nextResult === "function" ? nextResult(currentResult) : nextResult;
      resultRef.current = resolvedResult;
      return resolvedResult;
    });
  };

  const showValidationError = (message: string) => {
    setValidationError(message);
    toast.warning(message);
  };

  const parameterMetadataList = useMemo(() => {
    return pdfParameterMetadata.map((param) => ({
      name: param.name,
      label: t(`pdf.metadata.${param.name}.0`, param.label),
      description: t(`pdf.metadata.${param.name}.1`, param.description),
      required: param.required,
      supported: param.supported,
      defaultValue: param.defaultValue,
      options: param.options ? [...param.options] : undefined,
    }));
  }, [t]);

  return (
    <div className="page animate-fade-in">
      <header className="page-header">
        <p className="card-eyebrow">{t("pdf.kicker", "PDF to Images")}</p>
        <h2 className="page-title">{t("pdf.title", "Convert PDF pages into images")}</h2>
        <p className="page-description">{t("pdf.subtitle", "Upload a PDF, choose render settings, and download a ZIP of page images.")}</p>
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
            <Card.Header eyebrow={t("pdf.panelEyebrow", "Conversion request")} title={t("pdf.panelTitle", "PDF input")} />
            <Card.Body className="stack gap-md">
              <FormField label={t("pdf.metadata.file.0", "PDF file")} description={t("pdf.fileDescription", "Upload the PDF document to render into images.")}>
                <FileDropZone
                  accept="application/pdf,.pdf"
                  selectedFile={form.file}
                  onFileSelect={handleFileSelect}
                  dropZoneText={t("pdf.dropZoneText", "Drag & drop a PDF here, or click to select")}
                  dropZoneActiveText={t("pdf.dropZoneActive", "Drop the PDF here...")}
                />
              </FormField>

              <div className="grid grid-cols-2 gap-md">
                <FormField label={t("pdf.metadata.response_format.0", "Output image format")} description={t("pdf.responseFormatDescription", "Choose the image format for rendered pages.")}>
                  <Select
                    id="pdf-response-format"
                    name="response_format"
                    onChange={(event) => updateForm("responseFormat", event.target.value as PdfImageFormat)}
                    value={form.responseFormat}
                  >
                    {pdfImageFormats.map((format) => (
                      <option key={format} value={format}>
                        {format}
                      </option>
                    ))}
                  </Select>
                </FormField>

                <FormField label={t("pdf.metadata.scale.0", "Render scale")} description={t("pdf.scaleDescription", "Use 1.0 for native scale, or increase for higher resolution.")}>
                  <Input
                    id="pdf-scale"
                    inputMode="decimal"
                    name="scale"
                    onChange={(event) => updateForm("scale", event.target.value)}
                    placeholder={t("pdf.scalePlaceholder", "1.0")}
                    value={form.scale}
                  />
                </FormField>
              </div>

              <FormField label={t("pdf.metadata.pages.0", "Pages")} description={t("pdf.pagesDescription", "Optional 1-based page selectors, for example 1,3-5. Leave blank for all pages.")}>
                <Input
                  id="pdf-pages"
                  name="pages"
                  onChange={(event) => updateForm("pages", event.target.value)}
                  placeholder={t("pdf.pagesPlaceholder", "all pages")}
                  value={form.pages}
                />
              </FormField>

              <div className="pt-2 stack gap-sm">
                <Button
                  variant={isSubmitting ? "danger" : "primary"}
                  size="lg"
                  type={isSubmitting ? "button" : "submit"}
                  icon={isSubmitting ? <Square size={18} /> : <Play size={18} />}
                  fullWidth
                  onClick={isSubmitting ? handleCancelSubmit : undefined}
                >
                  {isSubmitting ? t("common.cancel") : t("pdf.submit", "Convert PDF")}
                </Button>
              </div>
            </Card.Body>
          </Card>

          <div className="stack gap-md">
            <Card>
              <Card.Header eyebrow={t("pdf.previewEyebrow", "Request preview")} title={t("pdf.previewTitle", "Summary and cURL")} />
              <Card.Body className="stack gap-md">
                {!form.file && (
                  <Alert variant="info" title={t("pdf.previewNoticeTitle", "Preview uses a placeholder file")}>
                    {t("pdf.previewNotice", "Select a PDF to update the request preview with its filename.")}
                  </Alert>
                )}
                <div className="result-block stack gap-sm">
                  <span className="card-eyebrow">{t("pdf.summaryLabel", "Request summary")}</span>
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
        <Card.Header eyebrow={t("pdf.responseEyebrow", "ZIP result")} title={t("pdf.responseTitle", "Rendered images")}>
          {result && (
            <Button type="button" size="sm" onClick={handleDownload} icon={<Download size={14} />} iconPosition="left">
              {t("pdf.download", "Download ZIP")}
            </Button>
          )}
        </Card.Header>
        <Card.Body>
          {result ? (
            <div className="stack gap-md animate-fade-in">
              <Alert variant="success" title={t("pdf.resultReady", "PDF images are ready")}>
                {t("pdf.resultNotice", "Download the ZIP archive to save the rendered page images.")}
              </Alert>

              <div className="result-block stack gap-sm">
                <FileArchive size={20} className="text-accent" />
                <span className="font-semibold text-primary">{result.fileName}</span>
                <ul className="stack gap-xs text-sm list-disc pl-4 text-muted">
                  <li>{t("pdf.resultPageCount", { count: result.pageCount ?? "unknown", defaultValue: "PDF pages: {{count}}" })}</li>
                  <li>{t("pdf.resultImageCount", { count: result.imageCount ?? "unknown", defaultValue: "Images in ZIP: {{count}}" })}</li>
                  <li>{t("pdf.resultZipSize", { size: formatBytes(result.blob.size), defaultValue: "ZIP size: {{size}}" })}</li>
                </ul>
              </div>

              <div className="result-block stack gap-sm">
                <span className="card-eyebrow">{t("pdf.previewImages", "Image preview")}</span>
                {result.images.length > 0 ? (
                  <div className="grid gap-md" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
                    {result.images.map((image) => (
                      <figure key={image.name} className="stack gap-xs" style={{ margin: 0 }}>
                        <img
                          alt={image.name}
                          src={image.url}
                          style={{
                            width: "100%",
                            aspectRatio: "3 / 4",
                            objectFit: "contain",
                            background: "var(--color-bg-sunken)",
                            border: "1px solid var(--color-border-subtle)",
                            borderRadius: "var(--radius-lg)",
                          }}
                        />
                        <figcaption className="text-xs text-muted">{image.name} · {formatBytes(image.size)}</figcaption>
                      </figure>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted">{t("pdf.previewImagesEmpty", "No previewable image entries were found in the ZIP archive.")}</p>
                )}
              </div>
            </div>
          ) : (
            <StateView
              type="empty"
              title={t("pdf.resultEmptyTitle", "No PDF images yet")}
              description={t("pdf.resultEmpty", "Submit a PDF conversion request to download the rendered images as a ZIP archive.")}
            />
          )}
        </Card.Body>
      </Card>
    </div>
  );
}

function buildRequestInput(form: PdfFormState, selectedFile: File): PdfRequestInput {
  return {
    file: selectedFile,
    responseFormat: form.responseFormat,
    pages: form.pages,
    scale: form.scale,
  };
}

async function buildPdfResult(response: Response, onImage: (image: PdfResult["images"][number]) => void): Promise<PdfResult> {
  const contentType = response.headers.get("Content-Type") ?? "application/zip";
  const streamedResult = await readZipWithImagePreviews(response, { contentType, onImage });

  return {
    blob: streamedResult.blob,
    ...resultMetadata(response),
    images: streamedResult.images,
  };
}

function buildPendingPdfResult(response: Response): PdfResult {
  const contentType = response.headers.get("Content-Type") ?? "application/zip";

  return {
    blob: new Blob([], { type: contentType }),
    ...resultMetadata(response),
    images: [],
  };
}

function resultMetadata(response: Response): Pick<PdfResult, "fileName" | "pageCount" | "imageCount"> {
  return {
    fileName: fileNameFromContentDisposition(response.headers.get("Content-Disposition")) ?? "pdf-images.zip",
    pageCount: response.headers.get("x-pdf-page-count"),
    imageCount: response.headers.get("x-pdf-image-count"),
  };
}

function fileNameFromContentDisposition(contentDisposition: string | null): string | null {
  if (!contentDisposition) {
    return null;
  }

  const utf8Match = /filename\*=UTF-8''([^;]+)/i.exec(contentDisposition);
  if (utf8Match) {
    return decodeURIComponent(utf8Match[1].trim().replace(/^"|"$/g, ""));
  }

  const fallbackMatch = /filename=("[^"]+"|[^;]+)/i.exec(contentDisposition);
  return fallbackMatch?.[1].trim().replace(/^"|"$/g, "") || null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
