import React from "react";
import { useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Menu, Search, Keyboard } from "lucide-react";
import { Button } from "./Button";

interface TopBarProps {
  onToggleMobileMenu: () => void;
  onOpenCommandPalette: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  onToggleMobileMenu,
  onOpenCommandPalette
}) => {
  const location = useLocation();
  const { t } = useTranslation();

  const getPageTitle = () => {
    const path = location.pathname;
    if (path.includes("/asr")) return t("shell.nav.asr.label", "ASR");
    if (path.includes("/tts")) return t("shell.nav.tts.label", "TTS");
    if (path.includes("/ocr")) return t("shell.nav.ocr.label", "OCR");
    if (path.includes("/models")) return t("shell.nav.models.label", "Models");
    if (path.includes("/settings")) return t("shell.nav.settings.label", "Settings");
    return "Orchion";
  };

  return (
    <div
      className="topbar"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "var(--space-3) var(--space-4)",
        background: "var(--color-surface-primary)",
        borderBottom: "1px solid var(--color-border-subtle)"
      }}
    >
      <div className="hstack gap-sm">
        <Button
          variant="ghost"
          size="sm"
          className="btn-icon-only text-muted hover:text-white"
          onClick={onToggleMobileMenu}
        >
          <Menu size={20} />
        </Button>
        <span className="text-md font-bold text-primary">{getPageTitle()}</span>
      </div>

      <div className="hstack gap-sm">
        <Button
          variant="ghost"
          size="sm"
          className="btn-icon-only text-muted hover:text-white"
          onClick={onOpenCommandPalette}
          title="Search / Command Palette"
        >
          <Search size={18} />
        </Button>
      </div>
    </div>
  );
};
