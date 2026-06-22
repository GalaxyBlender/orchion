import React from "react";
import { useTranslation } from "react-i18next";
import { Loader2, AlertCircle } from "lucide-react";

export interface ModelStatusProps {
  models: string[];
  isLoading: boolean;
  error: Error | null;
  kind: "ASR" | "TTS" | "OCR";
  listId: string;
}

export const ModelStatus: React.FC<ModelStatusProps> = ({
  models,
  isLoading,
  error,
  kind,
  listId
}) => {
  const { t } = useTranslation();

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="hstack gap-sm text-xs text-muted">
          <Loader2 size={12} className="animate-spin text-accent" />
          <span>{t("common.loadingModels", "Loading model suggestions...")}</span>
        </div>
      );
    }

    if (error) {
      return (
        <div className="hstack gap-sm text-xs text-warning">
          <AlertCircle size={12} />
          <span>
            {t("common.modelSuggestionsUnavailable", { message: error.message })}
          </span>
        </div>
      );
    }

    if (models.length === 0) {
      return (
        <div className="text-xs text-muted">
          {t("common.noModelSuggestions", { kind })}
        </div>
      );
    }

    return (
      <div className="text-xs text-muted">
        {t("common.modelSuggestions", { kind, models: models.join(", ") })}
      </div>
    );
  };

  return (
    <div className="stack gap-xs">
      {renderContent()}
      <datalist id={listId}>
        {models.map((id) => (
          <option key={id} value={id} />
        ))}
      </datalist>
    </div>
  );
};
