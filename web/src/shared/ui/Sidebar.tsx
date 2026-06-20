import React, { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Mic, Volume2, Database, Settings, Globe, ChevronLeft, ChevronRight, Wifi, WifiOff } from "lucide-react";
import { Badge } from "./Badge";
import { Button } from "./Button";
import { setLanguageSelection, currentLanguageSelection } from "@/shared/i18n";
import { loadPersistentState } from "@/shared/storage/persistentState";
import { fetchModels } from "@/shared/api/client";

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
  const [latency, setLatency] = useState<number | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(true);

  // Check backend health status periodically
  useEffect(() => {
    const checkServer = async () => {
      const state = loadPersistentState();
      const start = Date.now();
      try {
        await fetchModels(state.settings);
        setLatency(Date.now() - start);
        setIsOnline(true);
      } catch {
        setIsOnline(false);
      }
    };

    checkServer();
    const interval = setInterval(checkServer, 10000);
    return () => clearInterval(interval);
  }, []);

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
    <div className="flex flex-col h-full justify-between p-4" style={{ height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
      <div className="stack gap-lg">
        {/* Brand Header */}
        <div className="hstack gap-sm justify-between">
          <div className="hstack gap-sm">
            <div className="brand-icon flex items-center justify-center bg-accent text-base rounded p-2" style={{ background: "var(--color-accent)", color: "var(--color-bg-sunken)", borderRadius: "var(--radius-sm)", padding: "var(--space-2)" }}>
              <TerminalIcon size={18} />
            </div>
            {!collapsed && (
              <div className="stack gap-0">
                <span className="font-bold text-md text-primary tracking-tight">Orchion</span>
                <span className="text-xs text-muted">v0.2.0</span>
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

        {/* Navigation items */}
        <nav className="stack gap-sm" aria-label={t("shell.primaryNav", "Primary navigation")}>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => 
                `nav-link hstack gap-md p-3 rounded-md transition-colors ${isActive ? "active" : ""}`
              }
              style={({ isActive }) => ({
                display: "flex",
                alignItems: "center",
                gap: "var(--space-3)",
                padding: "var(--space-2-5) var(--space-3)",
                borderRadius: "var(--radius-md)",
                color: isActive ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                background: isActive ? "var(--color-surface-active)" : "transparent",
                borderLeft: isActive ? "3px solid var(--color-accent)" : "3px solid transparent",
                textDecoration: "none",
                transition: "var(--transition-colors)"
              })}
            >
              <span className="nav-icon">{item.icon}</span>
              {!collapsed && (
                <div className="stack gap-0" style={{ display: "flex", flexDirection: "column", gap: "0" }}>
                  <span className="text-sm font-semibold">{item.label}</span>
                  <span className="text-xs text-tertiary">{item.meta}</span>
                </div>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Footer controls */}
      <div className="stack gap-md mt-auto">
        {/* Language switch */}
        {!collapsed && (
          <div className="hstack gap-xs text-xs text-muted" style={{ display: "flex", alignItems: "center", gap: "var(--space-1)" }}>
            <Globe size={14} className="text-tertiary" />
            <select
              style={{
                background: "transparent",
                border: "none",
                color: "var(--color-text-secondary)",
                fontSize: "var(--text-xs)",
                cursor: "pointer",
                padding: "2px 4px",
                outline: "none",
              }}
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

        {/* Server Connection status */}
        <div className="hstack gap-sm text-xs border-t border-subtle pt-3" style={{ borderTop: "1px solid var(--color-border-subtle)", paddingTop: "var(--space-3)", display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          {isOnline ? (
            <>
              <Wifi size={14} className="text-success" />
              {!collapsed && (
                <span className="text-success font-semibold">
                  {t("common.online", "Online")}
                  {latency !== null && <span className="text-tertiary text-mono" style={{ marginLeft: "var(--space-1)" }}>({latency}ms)</span>}
                </span>
              )}
            </>
          ) : (
            <>
              <WifiOff size={14} className="text-danger" />
              {!collapsed && (
                <span className="text-danger font-semibold">{t("common.offline", "Offline")}</span>
              )}
            </>
          )}
        </div>

        {collapsed && (
          <Button
            variant="ghost"
            size="sm"
            className="btn-icon-only text-muted hover:text-white mx-auto"
            onClick={onToggleCollapse}
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
