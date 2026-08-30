import { type FormEvent, type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Bot, ChevronDown, Eraser, Send, Square, User } from "lucide-react";
import { useModels } from "@/features/models/useModels";
import { buildLlmCurl, buildLlmRequest, llmEndpointPath } from "@/features/llm/request";
import { consumeChatCompletionStream } from "@/features/llm/stream";
import type { LlmFormState, LlmMessage, LlmUsage } from "@/features/llm/types";
import { requestStream } from "@/shared/api/client";
import { buildApiError, type SubmissionError } from "@/shared/api/apiHelpers";
import {
  loadPersistentState,
  savePersistentState,
  type PersistentLlmState,
  type PersistentState,
} from "@/shared/storage/persistentState";
import {
  Alert,
  Badge,
  Button,
  Card,
  CodePreview,
  FormField,
  Input,
  ModelStatus,
  StateView,
  SuggestionInput,
  TextArea,
  useToast,
} from "@/shared/ui";

export function LlmPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const [persistentState, setPersistentState] = useState<PersistentState>(() => loadPersistentState());
  const [form, setForm] = useState<LlmFormState>(() => llmStateToForm(persistentState.llm));
  const [messages, setMessages] = useState<LlmMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [validationError, setValidationError] = useState("");
  const [submitError, setSubmitError] = useState<SubmissionError | null>(null);
  const [usage, setUsage] = useState<LlmUsage | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const settings = persistentState.settings;
  const models = useModels(settings);
  const llmModelIds = useMemo(() => models.classified.llm.map((model) => model.id), [models.classified.llm]);
  const previewMessages = useMemo<LlmMessage[]>(() => {
    const contextMessages = messages.filter((message) => message.includeInContext !== false);
    if (draft.trim() !== "") {
      return [...contextMessages, { id: "preview", role: "user", content: draft.trim() }];
    }
    if (contextMessages.length > 0) {
      return contextMessages;
    }
    return [{ id: "preview", role: "user", content: t("llm.previewPlaceholder") }];
  }, [draft, messages, t]);
  const curlPreview = useMemo(
    () => buildLlmCurl(settings, buildLlmRequest(form, previewMessages)),
    [form, previewMessages, settings],
  );

  useEffect(() => () => abortControllerRef.current?.abort(), []);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ block: "end", behavior: isGenerating ? "smooth" : "auto" });
  }, [isGenerating, messages]);

  const updateForm = <K extends keyof LlmFormState>(field: K, value: LlmFormState[K]) => {
    setValidationError("");
    setSubmitError(null);
    setForm((currentForm) => {
      const nextForm = { ...currentForm, [field]: value };
      setPersistentState((currentState) => {
        const nextState: PersistentState = { ...currentState, llm: nextForm };
        savePersistentState(nextState);
        return nextState;
      });
      return nextForm;
    });
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const prompt = draft.trim();
    const error = validateForm(form, prompt, {
      missingModel: t("llm.validation.missingModel"),
      missingPrompt: t("llm.validation.missingPrompt"),
      temperature: t("llm.validation.temperature"),
      topP: t("llm.validation.topP"),
      maxTokens: t("llm.validation.maxTokens"),
    });

    if (error) {
      setValidationError(error);
      toast.warning(error);
      return;
    }

    const userMessage: LlmMessage = { id: nextMessageId(), role: "user", content: prompt };
    const assistantMessage: LlmMessage = {
      id: nextMessageId(),
      role: "assistant",
      content: "",
      streamState: "streaming",
    };
    const requestMessages = [
      ...messages.filter((message) => message.includeInContext !== false),
      userMessage,
    ];
    const abortController = new AbortController();

    setDraft("");
    setValidationError("");
    setSubmitError(null);
    setUsage(null);
    setMessages([...messages, userMessage, assistantMessage]);
    setIsGenerating(true);
    abortControllerRef.current = abortController;

    try {
      const response = await requestStream(settings, llmEndpointPath, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildLlmRequest(form, requestMessages)),
        signal: abortController.signal,
      });

      await consumeChatCompletionStream(response.body, {
        onDelta: (delta) => {
          setMessages((current) => current.map((message) => (
            message.id === assistantMessage.id
              ? { ...message, content: message.content + delta }
              : message
          )));
        },
        onUsage: setUsage,
      });
      setMessages((current) => current.map((message) => (
        message.id === assistantMessage.id ? { ...message, streamState: "complete" } : message
      )));
    } catch (caughtError) {
      const wasStopped = isAbortError(caughtError);
      if (wasStopped) {
        toast.info(t("llm.cancelled"));
      } else {
        setSubmitError(buildApiError(caughtError));
        toast.error(t("llm.failed"));
      }
      setMessages((current) => current.flatMap((message) => {
        if (message.id === userMessage.id) {
          return [{ ...message, includeInContext: false }];
        }
        if (message.id !== assistantMessage.id) {
          return [message];
        }
        if (message.content === "") {
          return [];
        }
        return [{
          ...message,
          includeInContext: false,
          streamState: wasStopped ? "stopped" : "error",
        }];
      }));
    } finally {
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
      setIsGenerating(false);
    }
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const clearConversation = () => {
    setMessages([]);
    setUsage(null);
    setSubmitError(null);
  };

  return (
    <div className="page animate-fade-in llm-page">
      <header className="page-header">
        <p className="card-eyebrow">{t("llm.kicker")}</p>
        <h2 className="page-title">{t("llm.title")}</h2>
        <p className="page-description">{t("llm.subtitle")}</p>
      </header>

      {validationError && <Alert variant="warning" title={t("common.validationError")}>{validationError}</Alert>}
      {submitError && <Alert variant="danger" title={t("common.apiError")}>{submitError.message}</Alert>}

      <div className="llm-workbench">
        <aside className="llm-control-column stack gap-md">
          <Card variant="glass">
            <Card.Header eyebrow={t("llm.controlsEyebrow")} title={t("llm.controlsTitle")} />
            <Card.Body className="stack gap-md">
              <FormField htmlFor="llm-model" label={t("llm.modelLabel")} description={t("llm.modelDescription")}>
                <SuggestionInput
                  id="llm-model"
                  value={form.model}
                  onChange={(value) => updateForm("model", value)}
                  suggestions={llmModelIds}
                  placeholder={t("llm.modelPlaceholder")}
                  disabled={isGenerating}
                />
                <ModelStatus
                  models={llmModelIds}
                  isLoading={models.isLoading}
                  error={models.error}
                  kind="LLM"
                  listId="llm-model-suggestions"
                />
              </FormField>

              <FormField htmlFor="llm-system-prompt" label={t("llm.systemPromptLabel")} description={t("llm.systemPromptDescription")}>
                <TextArea
                  id="llm-system-prompt"
                  value={form.systemPrompt}
                  onChange={(event) => updateForm("systemPrompt", event.target.value)}
                  rows={4}
                  disabled={isGenerating}
                  placeholder={t("llm.systemPromptPlaceholder")}
                />
              </FormField>

              <div className="llm-parameter-grid">
                <NumericField id="llm-temperature" label={t("llm.temperatureLabel")} value={form.temperature} min="0" max="2" step="0.1" disabled={isGenerating} onChange={(value) => updateForm("temperature", value)} />
                <NumericField id="llm-top-p" label={t("llm.topPLabel")} value={form.topP} min="0.01" max="1" step="0.01" disabled={isGenerating} onChange={(value) => updateForm("topP", value)} />
                <NumericField id="llm-max-tokens" label={t("llm.maxTokensLabel")} value={form.maxCompletionTokens} min="1" step="1" disabled={isGenerating} onChange={(value) => updateForm("maxCompletionTokens", value)} />
              </div>
            </Card.Body>
          </Card>

          <details className="card llm-request-preview">
            <summary>
              <span className="card-title-group">
                <span className="card-eyebrow">{t("llm.requestPreviewEyebrow")}</span>
                <span className="card-title">{t("llm.requestPreview")}</span>
              </span>
              <ChevronDown className="llm-request-preview-chevron" size={18} aria-hidden="true" />
            </summary>
            <div className="llm-request-preview-body">
              <CodePreview label="post /v1/chat/completions">{curlPreview}</CodePreview>
            </div>
          </details>
        </aside>

        <Card className="llm-conversation-card">
          <Card.Header eyebrow={t("llm.conversationEyebrow")} title={t("llm.conversationTitle")}>
            <div className="hstack gap-sm flex-wrap justify-end">
              {usage && (
                <div className="llm-usage" aria-label={t("llm.usageLabel")}>
                  <Badge>{t("llm.usagePrompt", { count: usage.promptTokens })}</Badge>
                  <Badge>{t("llm.usageCompletion", { count: usage.completionTokens })}</Badge>
                  <Badge variant="accent">{t("llm.usageTotal", { count: usage.totalTokens })}</Badge>
                </div>
              )}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="btn-icon-only"
                onClick={clearConversation}
                disabled={isGenerating || messages.length === 0}
                title={t("llm.clear")}
                aria-label={t("llm.clear")}
              >
                <Eraser size={16} />
              </Button>
            </div>
          </Card.Header>

          <Card.Body className="llm-conversation-body">
            <div className="llm-message-list" aria-live="polite">
              {messages.length === 0 ? (
                <StateView type="empty" title={t("llm.emptyTitle")} description={t("llm.emptyDescription")} />
              ) : messages.map((message, index) => (
                <article key={message.id} className={`llm-message llm-message-${message.role}`}>
                  <div className="llm-message-avatar" aria-hidden="true">
                    {message.role === "user" ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className="llm-message-content">
                    <span className="llm-message-role">{t(`llm.roles.${message.role}`)}</span>
                    <div className="llm-message-text">
                      {message.content || (isGenerating && index === messages.length - 1 ? <span className="llm-thinking">{t("llm.thinking")}</span> : t("llm.emptyResponse"))}
                    </div>
                    {(message.streamState === "stopped" || message.streamState === "error") && (
                      <span className={`llm-message-state llm-message-state-${message.streamState}`}>
                        {t(`llm.incomplete.${message.streamState}`)}
                      </span>
                    )}
                  </div>
                </article>
              ))}
              <div ref={conversationEndRef} />
            </div>
          </Card.Body>

          <div className="llm-composer">
            <form onSubmit={handleSubmit} className="llm-composer-form">
              <TextArea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                disabled={isGenerating}
                rows={3}
                placeholder={t("llm.composerPlaceholder")}
                aria-label={t("llm.composerLabel")}
              />
              <div className="llm-composer-actions">
                {isGenerating ? (
                  <Button type="button" variant="secondary" onClick={() => abortControllerRef.current?.abort()} icon={<Square size={15} />}>
                    {t("llm.stop")}
                  </Button>
                ) : (
                  <Button type="submit" variant="primary" disabled={draft.trim() === ""} icon={<Send size={16} />}>
                    {t("llm.send")}
                  </Button>
                )}
              </div>
            </form>
          </div>
        </Card>
      </div>
    </div>
  );
}

interface NumericFieldProps {
  id: string;
  label: string;
  value: string;
  min: string;
  max?: string;
  step: string;
  disabled: boolean;
  placeholder?: string;
  onChange: (value: string) => void;
}

function NumericField({ id, label, value, min, max, step, disabled, placeholder, onChange }: NumericFieldProps) {
  return (
    <FormField htmlFor={id} label={label}>
      <Input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </FormField>
  );
}

interface ValidationText {
  missingModel: string;
  missingPrompt: string;
  temperature: string;
  topP: string;
  maxTokens: string;
}

function validateForm(form: LlmFormState, prompt: string, text: ValidationText): string {
  if (form.model.trim() === "") return text.missingModel;
  if (prompt === "") return text.missingPrompt;
  if (!validOptionalNumber(form.temperature, (value) => value >= 0 && value <= 2)) return text.temperature;
  if (!validOptionalNumber(form.topP, (value) => value > 0 && value <= 1)) return text.topP;
  if (!validOptionalNumber(form.maxCompletionTokens, (value) => Number.isInteger(value) && value > 0)) return text.maxTokens;
  return "";
}

function validOptionalNumber(value: string, predicate: (value: number) => boolean): boolean {
  if (value.trim() === "") return true;
  const number = Number(value);
  return Number.isFinite(number) && predicate(number);
}

let messageSequence = 0;

function nextMessageId(): string {
  messageSequence += 1;
  return `llm-message-${messageSequence}`;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function llmStateToForm(state: PersistentLlmState): LlmFormState {
  return { ...state };
}
