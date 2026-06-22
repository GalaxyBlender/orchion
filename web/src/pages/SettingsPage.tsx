import { type ChangeEvent, useState } from "react";
import { useTranslation } from "react-i18next";
import { currentLanguageSelection, setLanguageSelection, type LanguageSelection } from "@/shared/i18n";
import {
  cloneDefaultPersistentState,
  defaultPersistentState,
  loadPersistentState,
  resetPersistentState,
  savePersistentState,
  type PersistentSettings,
  type PersistentState,
} from "@/shared/storage/persistentState";
import { Card, FormField, Input, Select, Button, Alert, useToast } from "@/shared/ui";

export function SettingsPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const [persistentState, setPersistentState] = useState<PersistentState>(() => loadPersistentState());
  const [languageSelection, setLanguageSelectionState] = useState<LanguageSelection>(() => persistentState.ui.language ?? currentLanguageSelection());
  const settings = persistentState.settings;

  const updateSettings = (event: ChangeEvent<HTMLInputElement>) => {
    const field = event.target.name as keyof PersistentSettings;
    const value = event.target.value;

    setPersistentState((currentState) => {
      const nextState: PersistentState = {
        ...currentState,
        settings: { ...currentState.settings, [field]: value },
      };
      savePersistentState(nextState);
      return nextState;
    });
  };

  const updateLanguage = (event: ChangeEvent<HTMLSelectElement>) => {
    const selection = event.target.value as LanguageSelection;
    setLanguageSelection(selection);
    setLanguageSelectionState(selection);
    setPersistentState((currentState) => {
      const nextUi = { ...currentState.ui };
      if (selection === "auto") {
        delete nextUi.language;
      } else {
        nextUi.language = selection;
      }

      const nextState: PersistentState = { ...currentState, ui: nextUi };
      savePersistentState(nextState);
      return nextState;
    });
    toast.success(t("settings.languageLabel", "Language updated"));
  };

  const resetAsr = () => {
    resetSection({ asr: cloneDefaultPersistentState().asr });
    toast.success(t("settings.resetAsr", "ASR settings reset"));
  };

  const resetTts = () => {
    resetSection({ tts: cloneDefaultPersistentState().tts });
    toast.success(t("settings.resetTts", "TTS settings reset"));
  };

  const resetOcr = () => {
    resetSection({ ocr: cloneDefaultPersistentState().ocr });
    toast.success(t("settings.resetOcr", "OCR settings reset"));
  };

  const resetUi = () => {
    setLanguageSelection("auto");
    setLanguageSelectionState("auto");
    resetSection({ ui: cloneDefaultPersistentState().ui });
    toast.success(t("settings.resetUi", "UI preferences reset"));
  };

  const resetAll = () => {
    setLanguageSelection("auto");
    setLanguageSelectionState("auto");
    setPersistentState(resetPersistentState());
    toast.success(t("settings.resetAll", "Reset all settings"));
    setTimeout(() => window.location.reload(), 500);
  };

  function resetSection(section: Pick<Partial<PersistentState>, "asr" | "tts" | "ocr" | "ui">): void {
    setPersistentState((currentState) => {
      const nextState: PersistentState = { ...currentState, ...section };
      savePersistentState(nextState);
      return nextState;
    });
  }

  return (
    <div className="page animate-fade-in">
      <header className="page-header">
        <p className="card-eyebrow">{t("settings.kicker")}</p>
        <h2 className="page-title">{t("settings.title")}</h2>
        <p className="page-description">{t("settings.subtitle")}</p>
      </header>

      <Card variant="glass">
        <Card.Header eyebrow={t("settings.connectionEyebrow")} title={t("settings.connectionTitle")} />
        <Card.Body className="stack gap-md">
          <Alert variant="warning" title={t("settings.warningTitle")}>
            {t("settings.warning")}
          </Alert>

          <div className="grid grid-cols-2 gap-md">
            <FormField label={t("settings.apiKeyLabel")} description={t("settings.apiKeyDescription")}>
              <Input
                autoComplete="off"
                id="settings-api-key"
                name="apiKey"
                onChange={updateSettings}
                type="password"
                value={settings.apiKey}
              />
            </FormField>

            <FormField label={t("settings.languageLabel")} description={t("settings.languageDescription")}>
              <Select id="settings-language" name="language" onChange={updateLanguage} value={languageSelection}>
                <option value="auto">{t("settings.languageAuto")}</option>
                <option value="en">{t("settings.languageEnglish")}</option>
                <option value="zh-CN">{t("settings.languageChinese")}</option>
                <option value="zh-TW">{t("settings.languageChineseTW")}</option>
              </Select>
            </FormField>
          </div>
        </Card.Body>
      </Card>

      <div className="grid grid-cols-2 gap-md">
        <Card>
          <Card.Header eyebrow={t("settings.summaryEyebrow")} title={t("settings.summaryTitle")} />
          <Card.Body>
            <div className="result-block stack gap-sm">
              <span className="card-eyebrow">{t("settings.summaryLabel")}</span>
              <ul className="stack gap-xs text-sm list-disc pl-4 text-muted">
                <li>{t("settings.server", { value: currentServerOrigin() })}</li>
                <li>{t("settings.apiKey", { value: settings.apiKey ? t("settings.apiKeyConfigured") : t("settings.apiKeyNotConfigured") })}</li>
                <li>{t("settings.asrModel", { value: persistentState.asr.model || t("common.blank") })}</li>
                <li>{t("settings.ttsModel", { value: persistentState.tts.model || t("common.blank") })}</li>
                <li>{t("settings.ocrModel", { value: persistentState.ocr.model || t("common.blank") })}</li>
                <li>{t("settings.activePage", { value: persistentState.ui.activePage })}</li>
                <li>{t("settings.languageSummary", { value: languageSelection === "auto" ? t("settings.languageAuto") : languageSelection })}</li>
              </ul>
            </div>
          </Card.Body>
        </Card>

        <Card>
          <Card.Header eyebrow={t("settings.resetEyebrow")} title={t("settings.resetTitle")} />
          <Card.Body className="stack gap-md">
            <p className="text-sm text-muted">{t("settings.resetText")}</p>
            <div className="hstack gap-sm flex-wrap">
              <Button variant="secondary" size="sm" onClick={resetAsr}>{t("settings.resetAsr")}</Button>
              <Button variant="secondary" size="sm" onClick={resetTts}>{t("settings.resetTts")}</Button>
              <Button variant="secondary" size="sm" onClick={resetOcr}>{t("settings.resetOcr")}</Button>
              <Button variant="secondary" size="sm" onClick={resetUi}>{t("settings.resetUi")}</Button>
              <Button variant="danger" size="sm" onClick={resetAll}>{t("settings.resetAll")}</Button>
            </div>
          </Card.Body>
        </Card>
      </div>

      <Card>
        <Card.Header eyebrow={t("settings.defaultsEyebrow")} title={t("settings.defaultsTitle")} />
        <Card.Body>
          <div className="result-block stack gap-sm">
            <span className="card-eyebrow">{t("settings.defaultsLabel")}</span>
            <ul className="stack gap-xs text-sm list-disc pl-4 text-muted">
              <li>{t("settings.apiKey", { value: defaultPersistentState.settings.apiKey || t("common.blank") })}</li>
              <li>{t("settings.defaultAsrFormat", { value: defaultPersistentState.asr.responseFormat })}</li>
              <li>{t("settings.defaultTtsVoice", { value: defaultPersistentState.tts.speaker })}</li>
              <li>{t("settings.defaultOcrFormat", { value: defaultPersistentState.ocr.responseFormat })}</li>
            </ul>
          </div>
        </Card.Body>
      </Card>
    </div>
  );
}

function currentServerOrigin(): string {
  if (typeof window === "undefined") {
    return "/";
  }

  return window.location.origin;
}
