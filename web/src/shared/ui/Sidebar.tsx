import React from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Mic, Volume2, Database, Settings, Globe, ChevronLeft, ChevronRight, ScanText, FileText } from "lucide-react";
import { Button } from "./Button";
import { setLanguageSelection, currentLanguageSelection } from "@/shared/i18n";

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onToggleCollapse,
  mobileOpen = false
}) => {
  const { t } = useTranslation();
  const location = useLocation();

  const navItems = [
    {
      to: "/asr",
      label: t("shell.nav.asr.label", "ASR"),
      meta: t("shell.nav.asr.meta", "speech to text"),
      icon: <Mic size={18} />
    },
    {
      to: "/tts",
      label: t("shell.nav.tts.label", "TTS"),
      meta: t("shell.nav.tts.meta", "voice synthesis"),
      icon: <Volume2 size={18} />
    },
    {
      to: "/ocr",
      label: t("shell.nav.ocr.label", "OCR"),
      meta: t("shell.nav.ocr.meta", "document vision"),
      icon: <ScanText size={18} />
    },
    {
      to: "/pdf",
      label: t("shell.nav.pdf.label", "PDF"),
      meta: t("shell.nav.pdf.meta", "page images"),
      icon: <FileText size={18} />
    },
    {
      to: "/models",
      label: t("shell.nav.models.label", "Models"),
      meta: t("shell.nav.models.meta", "catalog"),
      icon: <Database size={18} />
    },
    {
      to: "/settings",
      label: t("shell.nav.settings.label", "Settings"),
      meta: t("shell.nav.settings.meta", "runtime"),
      icon: <Settings size={18} />
    }
  ];

  const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setLanguageSelection(e.target.value as any);
  };

  return (
    <div className={`sidebar-inner ${collapsed ? "sidebar-inner-collapsed" : ""}`}>
      <div className="sidebar-main stack gap-lg">
        <div className={`sidebar-brand ${collapsed ? "sidebar-brand-collapsed" : ""}`}>
          <div className="sidebar-brand-mark">
            <div className={`brand-icon ${collapsed ? "brand-icon-collapsed" : ""}`}>
              <TerminalIcon size={18} />
            </div>
            {!collapsed && (
              <div className="sidebar-brand-copy stack gap-0">
                <span className="sidebar-brand-title">Orchion</span>
                <span className="sidebar-brand-version">v0.2.0</span>
              </div>
            )}
          </div>
          {!collapsed && (
            <Button
              variant="ghost"
              size="sm"
              className="btn-icon-only text-muted hover:text-white"
              onClick={onToggleCollapse}
              title={t("shell.toggleSidebar", "Toggle sidebar collapsed state")}
            >
              <ChevronLeft size={16} />
            </Button>
          )}
        </div>

        <nav className="sidebar-nav stack gap-sm" aria-label={t("shell.primaryNav", "Primary navigation")}>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => 
                `nav-link ${collapsed ? "nav-link-collapsed" : ""} ${isActive ? "active" : ""}`
              }
              aria-label={collapsed ? `${item.label}, ${item.meta}` : undefined}
              title={collapsed ? `${item.label} - ${item.meta}` : undefined}
            >
              <span className="nav-icon">{item.icon}</span>
              {!collapsed && (
                <div className="nav-link-copy stack gap-0">
                  <span className="nav-link-label">{item.label}</span>
                  <span className="nav-link-meta">{item.meta}</span>
                </div>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className={`sidebar-footer stack gap-md ${collapsed ? "sidebar-footer-collapsed" : ""}`}>
        {!collapsed && (
          <div className="sidebar-language">
            <Globe size={14} />
            <select
              className="sidebar-language-select"
              value={currentLanguageSelection()}
              onChange={handleLanguageChange}
            >
              <option value="auto" style={{ background: "var(--color-surface-primary)" }}>{t("settings.languageAuto", "Auto")}</option>
              <option value="en" style={{ background: "var(--color-surface-primary)" }}>English</option>
              <option value="zh-CN" style={{ background: "var(--color-surface-primary)" }}>简体中文</option>
              <option value="zh-TW" style={{ background: "var(--color-surface-primary)" }}>繁體中文</option>
            </select>
          </div>
        )}

        {collapsed && (
          <Button
            variant="ghost"
            size="sm"
            className="btn-icon-only sidebar-expand-button text-muted hover:text-white mx-auto"
            onClick={onToggleCollapse}
            title={t("shell.toggleSidebar", "Toggle sidebar collapsed state")}
          >
            <ChevronRight size={16} />
          </Button>
        )}
      </div>
    </div>
  );
};

// Internal mini TerminalIcon since we don't import everything
const TerminalIcon = ({ size }: { size: number }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="4 17 10 11 4 5" />
    <line x1="12" y1="19" x2="20" y2="19" />
  </svg>
);
