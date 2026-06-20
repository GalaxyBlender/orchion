import React, { useState, useEffect, useRef, useMemo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Search, Navigation, Settings, Play, ShieldAlert, Sparkles, Terminal } from "lucide-react";
import { resetPersistentState } from "@/shared/storage/persistentState";
import { useToast } from "./Toast";

export interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

interface CommandItem {
  id: string;
  title: string;
  icon: React.ReactNode;
  category: "nav" | "action";
  action: () => void;
  shortcut?: string;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation();
  const toast = useToast();
  
  const [search, setSearch] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      setSearch("");
      setSelectedIndex(0);
      // Wait for modal animation
      setTimeout(() => searchInputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  const commands = useMemo<CommandItem[]>(() => {
    return [
      {
        id: "nav-asr",
        title: t("shell.commandPalette.navAsr", "Go to ASR (Speech to Text)"),
        icon: <Sparkles size={16} />,
        category: "nav",
        action: () => { navigate("/asr"); onClose(); },
        shortcut: "⌘1"
      },
      {
        id: "nav-tts",
        title: t("shell.commandPalette.navTts", "Go to TTS (Voice Synthesis)"),
        icon: <Sparkles size={16} />,
        category: "nav",
        action: () => { navigate("/tts"); onClose(); },
        shortcut: "⌘2"
      },
      {
        id: "nav-models",
        title: t("shell.commandPalette.navModels", "Go to Model Catalog"),
        icon: <Navigation size={16} />,
        category: "nav",
        action: () => { navigate("/models"); onClose(); },
        shortcut: "⌘3"
      },
      {
        id: "nav-settings",
        title: t("shell.commandPalette.navSettings", "Go to Settings"),
        icon: <Settings size={16} />,
        category: "nav",
        action: () => { navigate("/settings"); onClose(); },
        shortcut: "⌘4"
      },
      {
        id: "action-reset",
        title: t("shell.commandPalette.actionResetAll", "Reset all local state"),
        icon: <ShieldAlert size={16} className="text-danger" />,
        category: "action",
        action: () => {
          resetPersistentState();
          toast.success(t("settings.resetAll", "Reset all state"));
          onClose();
          window.location.reload();
        }
      }
    ];
  }, [t, navigate, onClose, toast]);

  const filteredCommands = useMemo(() => {
    if (!search) return commands;
    const lowerSearch = search.toLowerCase();
    return commands.filter(cmd => cmd.title.toLowerCase().includes(lowerSearch));
  }, [search, commands]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % filteredCommands.length);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + filteredCommands.length) % filteredCommands.length);
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          filteredCommands[selectedIndex].action();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, filteredCommands, selectedIndex, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="command-palette-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="command-palette" ref={containerRef}>
        <div className="command-palette-search-wrapper">
          <Search size={18} className="text-muted" />
          <input
            ref={searchInputRef}
            type="text"
            className="command-palette-search"
            placeholder={t("shell.commandPalette.search", "Type a command or search page...")}
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setSelectedIndex(0);
            }}
          />
        </div>
        <div className="command-palette-results">
          {filteredCommands.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted">
              {t("shell.commandPalette.noResults", "No commands found.")}
            </div>
          ) : (
            filteredCommands.map((cmd, idx) => (
              <div
                key={cmd.id}
                className={`command-palette-item ${idx === selectedIndex ? "active" : ""}`}
                onClick={cmd.action}
              >
                <div className="command-palette-item-left">
                  <span className="text-muted">{cmd.icon}</span>
                  <span className="text-sm font-semibold">{cmd.title}</span>
                </div>
                {cmd.shortcut && (
                  <span className="command-palette-shortcut">{cmd.shortcut}</span>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
