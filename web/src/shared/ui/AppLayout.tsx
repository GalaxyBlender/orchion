import React, { useState, useEffect } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { CommandPalette } from "./CommandPalette";

export const AppLayout: React.FC = () => {
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem("orchion.webui.sidebar.collapsed") === "true";
    } catch {
      return false;
    }
  });
  const [mobileOpen, setMobileOpen] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  const handleToggleCollapse = () => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem("orchion.webui.sidebar.collapsed", String(next));
      } catch {}
      return next;
    });
  };

  const handleToggleMobileMenu = () => {
    setMobileOpen((prev) => !prev);
  };

  const handleOpenCommandPalette = () => {
    setCommandPaletteOpen(true);
  };

  const handleCloseCommandPalette = () => {
    setCommandPaletteOpen(false);
  };

  const sidebarCollapsed = mobileOpen ? false : collapsed;

  // Keyboard Hotkeys listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Toggle Command Palette: Cmd/Ctrl + K
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
      
      // Page shortcuts: Cmd/Ctrl + 1/2/3/4/5/6
      if ((e.metaKey || e.ctrlKey) && ["1", "2", "3", "4", "5", "6"].includes(e.key)) {
        e.preventDefault();
        if (e.key === "1") navigate("/asr");
        if (e.key === "2") navigate("/tts");
        if (e.key === "3") navigate("/ocr");
        if (e.key === "4") navigate("/pdf");
        if (e.key === "5") navigate("/models");
        if (e.key === "6") navigate("/settings");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigate]);

  return (
    <div className={`app-layout ${collapsed ? "collapsed" : ""}`}>
      {/* Mobile Topbar */}
      <div className="topbar-wrapper">
        <TopBar
          onToggleMobileMenu={handleToggleMobileMenu}
          onOpenCommandPalette={handleOpenCommandPalette}
        />
      </div>

      {/* Backdrop overlay for mobile drawer */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30"
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0, 0, 0, 0.5)",
            zIndex: 199
          }}
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar navigation */}
      <div className={`sidebar-wrapper ${mobileOpen ? "mobile-open" : ""}`} style={{ zIndex: 200 }}>
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={handleToggleCollapse}
          mobileOpen={mobileOpen}
        />
      </div>

      {/* Main content viewport */}
      <main className="app-content" id="main-content" tabIndex={-1}>
        <Outlet />
      </main>

      {/* Global Command Palette */}
      <CommandPalette isOpen={commandPaletteOpen} onClose={handleCloseCommandPalette} />
    </div>
  );
};
