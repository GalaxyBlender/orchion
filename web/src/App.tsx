import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { currentLanguageSelection, detectBrowserLanguage, syncLanguageFromSelection } from "@/shared/i18n";
import { AppLayout, ToastProvider } from "@/shared/ui";
import { AsrPage } from "./pages/AsrPage";
import { ModelsPage } from "./pages/ModelsPage";
import { OcrPage } from "./pages/OcrPage";
import { SettingsPage } from "./pages/SettingsPage";
import { TtsPage } from "./pages/TtsPage";
import "@/styles/index.css";

export function App() {
  const { i18n } = useTranslation();
  const [, setLanguageVersion] = useState(0);
  const languageSelection = currentLanguageSelection();
  const selectedLanguage = languageSelection === "auto" ? detectBrowserLanguage() : languageSelection;

  if (i18n.language !== selectedLanguage && i18n.resolvedLanguage !== selectedLanguage) {
    void i18n.changeLanguage(selectedLanguage);
  }

  useEffect(() => {
    void syncLanguageFromSelection(languageSelection).then(() => {
      setLanguageVersion((version) => version + 1);
    });
  }, [i18n.language, languageSelection]);

  return (
    <ToastProvider>
      <BrowserRouter basename="/ui">
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Navigate to="/asr" replace />} />
            <Route path="/asr" element={<AsrPage />} />
            <Route path="/tts" element={<TtsPage />} />
            <Route path="/ocr" element={<OcrPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/asr" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  );
}
