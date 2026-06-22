import { useState, startTransition } from "react";
import { useTranslation } from "react-i18next";
import { modelIds } from "@/features/models/modelCatalog";
import { useModels } from "@/features/models/useModels";
import type { ModelObject } from "@/shared/api/types";
import { loadPersistentState } from "@/shared/storage/persistentState";
import { Card, Button, Badge, Alert, StateView } from "@/shared/ui";
import { Mic, Volume2, Cpu, RefreshCw, ScanText } from "lucide-react";

export function ModelsPage() {
  const { t } = useTranslation();
  const [settings] = useState(() => loadPersistentState().settings);
  const catalog = useModels(settings);
  const allModelIds = modelIds(catalog.models);

  const handleReload = () => {
    startTransition(() => {
      catalog.reload();
    });
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
            loading={catalog.isLoading}
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
          <div className="grid grid-cols-2 gap-md">
            <Card>
              <Card.Header eyebrow={t("models.classified")} title={t("models.asrModels")} />
              <Card.Body>
                <ModelGroup
                  emptyText={t("models.noAsr")}
                  icon={<Mic size={16} className="text-accent" />}
                  models={catalog.classified.asr}
                  badgeVariant="accent"
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
                />
              </Card.Body>
            </Card>

            <Card>
              <Card.Header eyebrow={t("models.classified")} title={t("models.ocrModels")} />
              <Card.Body>
                <ModelGroup
                  emptyText={t("models.noOcr")}
                  icon={<ScanText size={16} className="text-accent" />}
                  models={catalog.classified.ocr}
                  badgeVariant="accent"
                />
              </Card.Body>
            </Card>
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
}

function ModelGroup({ emptyText, icon, models, badgeVariant }: ModelGroupProps) {
  if (models.length === 0) {
    return <p className="text-sm text-muted">{emptyText}</p>;
  }

  return (
    <div className="stack gap-sm">
      {models.map((model) => (
        <div
          key={model.id}
          className="hstack justify-between p-3 rounded-md border border-subtle bg-sunken"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "var(--space-2-5) var(--space-3)",
            background: "var(--color-bg-sunken)",
            border: "1px solid var(--color-border-subtle)",
            borderRadius: "var(--radius-md)"
          }}
        >
          <div className="hstack gap-sm">
            {icon}
            <span className="text-sm font-semibold text-mono truncate" style={{ maxWidth: "300px" }}>
              {model.id}
            </span>
          </div>
          <Badge variant={badgeVariant}>{model.owned_by ? String(model.owned_by) : "system"}</Badge>
        </div>
      ))}
    </div>
  );
}
