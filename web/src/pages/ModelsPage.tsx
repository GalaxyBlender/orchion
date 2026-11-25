import { useState } from "react";
import { useTranslation } from "react-i18next";
import { modelIds } from "@/features/models/modelCatalog";
import { useModelLifecycle, type ModelLifecycle } from "@/features/models/useModelLifecycle";
import { useModels } from "@/features/models/useModels";
import type { ModelObject, ModelService } from "@/shared/api/types";
import { loadPersistentState } from "@/shared/storage/persistentState";
import { Card, Button, Badge, Alert, StateView } from "@/shared/ui";
import { Mic, Volume2, Cpu, RefreshCw, ScanText, MessageSquareText } from "lucide-react";

export function ModelsPage() {
  const { t } = useTranslation();
  const [settings] = useState(() => loadPersistentState().settings);
  const catalog = useModels(settings);
  const lifecycle = useModelLifecycle(settings, catalog.reload);
  const allModelIds = modelIds(catalog.models);

  const handleReload = () => {
    catalog.reload();
    lifecycle.reload();
  };

  return (
    <div className="page animate-fade-in">
      <header className="page-header">
        <div className="page-title-row">
          <div className="stack gap-xs">
            <p className="card-eyebrow">{t("models.kicker")}</p>
            <h2 className="page-title">{t("models.title")}</h2>
          </div>
          <Button
            variant="secondary"
            loading={catalog.isLoading || lifecycle.isRefreshing}
            onClick={handleReload}
            icon={<RefreshCw size={16} />}
            iconPosition="left"
          >
            {t("models.reload")}
          </Button>
        </div>
        <p className="page-description">{t("models.subtitle")}</p>
      </header>

      {/* Mini overview cards */}
      <div className="grid gap-md" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" }}>
        <Card variant="glass">
          <Card.Header eyebrow={t("models.llmModels")} title={catalog.classified.llm.length.toString()} />
        </Card>
        <Card variant="glass">
          <Card.Header eyebrow={t("models.asrModels")} title={catalog.classified.asr.length.toString()} />
        </Card>
        <Card variant="glass">
          <Card.Header eyebrow={t("models.ttsModels")} title={catalog.classified.tts.length.toString()} />
        </Card>
        <Card variant="glass">
          <Card.Header eyebrow={t("models.ocrModels")} title={catalog.classified.ocr.length.toString()} />
        </Card>
        <Card variant="glass">
          <Card.Header eyebrow={t("models.totalModels")} title={catalog.models.length.toString()} />
        </Card>
      </div>

      {catalog.error && (
        <Alert variant="warning" title={t("models.unavailable")}>
          {t("models.unavailableMessage", { message: catalog.error.message })}
        </Alert>
      )}
      {lifecycle.statusError && (
        <Alert variant="warning" title={t("models.residencyUnavailable")}>
          {lifecycle.statusError}
        </Alert>
      )}

      {catalog.isLoading && catalog.models.length === 0 ? (
        <StateView type="loading" message={t("models.loadingStatus")} />
      ) : catalog.models.length === 0 ? (
        <StateView
          type="empty"
          title={t("models.noModels")}
          description={catalog.error ? catalog.error.message : undefined}
          action={<Button onClick={handleReload}>{t("models.reload")}</Button>}
        />
      ) : (
        <>
          <div className="stack gap-md">
            <Card>
              <Card.Header eyebrow={t("models.classified")} title={t("models.llmModels")} />
              <Card.Body>
                <ModelGroup
                  emptyText={t("models.noLlm")}
                  icon={<MessageSquareText size={16} className="text-accent-blue" />}
                  models={catalog.classified.llm}
                  badgeVariant="accent-blue"
                  controls={{ service: "llm", lifecycle }}
                />
              </Card.Body>
            </Card>

            <div className="grid grid-cols-2 gap-md models-pair-row">
              <Card>
                <Card.Header eyebrow={t("models.classified")} title={t("models.asrModels")} />
                <Card.Body>
                  <ModelGroup
                    emptyText={t("models.noAsr")}
                    icon={<Mic size={16} className="text-accent" />}
                    models={catalog.classified.asr}
                    badgeVariant="accent"
                    controls={{ service: "asr", lifecycle }}
                  />
                </Card.Body>
              </Card>

              <Card>
                <Card.Header eyebrow={t("models.classified")} title={t("models.ttsModels")} />
                <Card.Body>
                  <ModelGroup
                    emptyText={t("models.noTts")}
                    icon={<Volume2 size={16} className="text-accent-blue" />}
                    models={catalog.classified.tts}
                    badgeVariant="accent-blue"
                    controls={{ service: "tts", lifecycle }}
                  />
                </Card.Body>
              </Card>
            </div>

            <div className="grid grid-cols-2 gap-md models-pair-row">
              <Card>
                <Card.Header eyebrow={t("models.classified")} title={t("models.ocrModels")} />
                <Card.Body>
                  <ModelGroup
                    emptyText={t("models.noOcr")}
                    icon={<ScanText size={16} className="text-accent" />}
                    models={catalog.classified.ocrStandard}
                    badgeVariant="accent"
                    controls={{ service: "ocr", lifecycle }}
                  />
                </Card.Body>
              </Card>

              <Card>
                <Card.Header eyebrow={t("models.classified")} title={t("models.ocrVlModels")} />
                <Card.Body>
                  <ModelGroup
                    emptyText={t("models.noOcrVl")}
                    icon={<ScanText size={16} className="text-accent" />}
                    models={catalog.classified.ocrVl}
                    badgeVariant="accent"
                    controls={{ service: "ocr_vl", lifecycle }}
                  />
                </Card.Body>
              </Card>
            </div>

          </div>

          <Card>
            <Card.Header eyebrow={t("models.classified")} title={t("models.otherModels")} />
            <Card.Body>
              <ModelGroup
                emptyText={t("models.noOther")}
                icon={<Cpu size={16} className="text-muted" />}
                models={catalog.classified.other}
                badgeVariant="default"
              />
            </Card.Body>
          </Card>

          <Card>
            <Card.Header eyebrow={t("models.technical")} title={t("models.rawTitle")} />
            <Card.Body>
              <div className="result-block stack gap-sm">
                <span className="card-eyebrow">{t("models.rawIds")}</span>
                <pre className="code-preview">
                  <code>{JSON.stringify(allModelIds, null, 2)}</code>
                </pre>
              </div>
            </Card.Body>
          </Card>
        </>
      )}
    </div>
  );
}

interface ModelGroupProps {
  emptyText: string;
  icon: React.ReactNode;
  models: ModelObject[];
  badgeVariant: "default" | "accent" | "accent-blue";
  controls?: { service: ModelService; lifecycle: ModelLifecycle };
}

function ModelGroup({ emptyText, icon, models, badgeVariant, controls }: ModelGroupProps) {
  const { t } = useTranslation();
  if (models.length === 0) {
    return <p className="text-sm text-muted">{emptyText}</p>;
  }

  return (
    <div className="stack gap-sm">
      {models.map((model) => (
        <div key={model.id} className="model-row">
          <div className="model-row-details stack gap-sm">
            <div className="hstack gap-sm">
              <span className="model-row-icon">{icon}</span>
              <div className="model-row-name stack gap-xs">
                <span className="text-sm text-bold">{model.name ?? model.id}</span>
                {model.name && <span className="text-xs text-muted text-mono">{model.id}</span>}
              </div>
            </div>
            <div className="hstack gap-xs flex-wrap">
              {model.capabilities.map((capability) => (
                <Badge key={capability} variant={badgeVariant}>{t(`models.capabilities.${capability}`)}</Badge>
              ))}
            </div>
          </div>
          {controls && <ModelControls model={model} {...controls} />}
        </div>
      ))}
    </div>
  );
}

function ModelControls({ model, service, lifecycle }: {
  model: ModelObject;
  service: ModelService;
  lifecycle: ModelLifecycle;
}) {
  const { t } = useTranslation();
  const { statuses, pending, operationError, isRefreshing, change } = lifecycle;
  const modelStatus = statuses.find((entry) => entry.id === model.id && entry.service === service);
  const pendingAction = pending?.model === model.id && pending.service === service ? pending.action : undefined;
  const error = operationError?.model === model.id && operationError.service === service ? operationError : null;
  const action = pendingAction ?? (
    modelStatus?.status === "loaded" ? "unload" : modelStatus?.status === "unloaded" ? "load" : undefined
  );
  const status = pendingAction
    ? (pendingAction === "load" ? "loading" : "unloading")
    : modelStatus?.status ?? "unknown";

  return (
    <>
      <div className="model-row-controls hstack gap-sm flex-wrap">
        <span role="status">
          <Badge variant={status === "loaded" ? "success" : status === "unknown" ? "warning" : "default"} dot>
            {t(`models.residency.${status}`)}
          </Badge>
        </span>
        {action && (
          <Button
            size="sm"
            type="button"
            loading={Boolean(pendingAction)}
            disabled={Boolean(pending) || isRefreshing}
            aria-label={t(`models.controlLabel.${action}`, { model: model.name ?? model.id })}
            onClick={() => void change(action, { model: model.id, service })}
          >
            {t(`models.control.${action}`)}
          </Button>
        )}
      </div>
      {error && (
        <p className="model-row-error text-sm text-danger" role="alert">
          {t(`models.controlError.${error.action}`, { message: error.message })}
        </p>
      )}
    </>
  );
}
