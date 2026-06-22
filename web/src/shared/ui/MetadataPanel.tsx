import React, { useState, useMemo } from "react";
import { Badge } from "./Badge";
import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronRight, Search } from "lucide-react";

export interface ParameterMetadata {
  name: string;
  label: string;
  defaultValue?: string;
  description: string;
  required: boolean;
  supported: boolean;
  notice?: string;
  options?: readonly string[] | string[];
}

export interface MetadataItemProps {
  metadata: ParameterMetadata;
  isOpen: boolean;
  onToggle: () => void;
}

export const MetadataItem: React.FC<MetadataItemProps> = ({ metadata, isOpen, onToggle }) => {
  const { t } = useTranslation();

  return (
    <div className="border-b border-subtle last:border-b-0" style={{ borderColor: "var(--color-border-subtle)" }}>
      {/* Header trigger */}
      <button
        type="button"
        className="w-full hstack justify-between p-4 text-left hover:bg-hover transition-colors"
        style={{
          display: "flex",
          width: "100%",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "var(--space-3) var(--space-4)",
          background: isOpen ? "var(--color-surface-hover)" : "transparent",
          border: "none",
          cursor: "pointer",
          textAlign: "left"
        }}
        onClick={onToggle}
      >
        <div className="hstack gap-sm flex-wrap" style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <span className="font-semibold text-sm text-primary">{metadata.label}</span>
          <span className="text-xs text-tertiary text-mono">{metadata.name}</span>
          {metadata.required && (
            <Badge variant="danger" style={{ padding: "1px 6px", fontSize: "10px" }}>
              {t("common.required", "Req")}
            </Badge>
          )}
        </div>
        <div className="hstack gap-sm" style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          {isOpen ? <ChevronDown size={16} className="text-muted" /> : <ChevronRight size={16} className="text-muted" />}
        </div>
      </button>

      {/* Collapsible Content */}
      {isOpen && (
        <div
          className="stack gap-sm p-4 bg-sunken animate-fade-in"
          style={{
            background: "var(--color-bg-sunken)",
            padding: "var(--space-4)",
            display: "flex",
            flexDirection: "column",
            gap: "var(--space-2)"
          }}
        >
          <div className="hstack gap-sm flex-wrap" style={{ display: "flex", gap: "var(--space-2)" }}>
            {!metadata.required && (
              <Badge variant="default">{t("common.optional", "Optional")}</Badge>
            )}
            {metadata.supported ? (
              <Badge variant="success">{t("common.supported", "Supported")}</Badge>
            ) : (
              <Badge variant="warning">{t("common.unsupported", "Unsupported")}</Badge>
            )}
          </div>
          
          <p className="text-sm text-muted" style={{ lineHeight: "var(--leading-relaxed)" }}>{metadata.description}</p>
          
          {metadata.notice && (
            <p
              className="text-xs text-warning pl-2"
              style={{
                borderLeft: "2px solid var(--color-warning)",
                paddingLeft: "var(--space-2)",
                marginTop: "4px"
              }}
            >
              {metadata.notice}
            </p>
          )}
          
          {(metadata.defaultValue !== undefined || metadata.options) && (
            <div
              className="hstack gap-lg flex-wrap text-xs text-tertiary text-mono border-t border-subtle pt-2"
              style={{
                display: "flex",
                gap: "var(--space-4)",
                flexWrap: "wrap",
                borderTop: "1px solid var(--color-border-subtle)",
                paddingTop: "var(--space-2)",
                marginTop: "4px"
              }}
            >
              {metadata.defaultValue !== undefined && (
                <span>
                  {t("common.default", "Default")}: {metadata.defaultValue || t("common.blank", "blank")}
                </span>
              )}
              {metadata.options && (
                <span>
                  {t("common.options", "Options")}: {metadata.options.join(", ")}
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export interface MetadataPanelProps {
  metadataList: readonly ParameterMetadata[] | ParameterMetadata[];
}

export const MetadataPanel: React.FC<MetadataPanelProps> = ({ metadataList }) => {
  const { t } = useTranslation();
  const [search, setSearch] = useState("");
  const [openItem, setOpenItem] = useState<string | null>(null);

  const filteredList = useMemo(() => {
    if (!search) return metadataList;
    const lower = search.toLowerCase();
    return metadataList.filter(
      (item) =>
        item.label.toLowerCase().includes(lower) || item.name.toLowerCase().includes(lower)
    );
  }, [search, metadataList]);

  const toggleItem = (name: string) => {
    setOpenItem((prev) => (prev === name ? null : name));
  };

  return (
    <div className="card card-elevated overflow-hidden">
      <div className="card-header stack gap-sm" style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
        <h3 className="card-title">{t("common.parameterNotes", "Parameter Notes")}</h3>
        
        {/* Search bar inside header */}
        <div
          className="hstack gap-xs rounded bg-sunken border border-subtle"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--space-2)",
            background: "var(--color-bg-sunken)",
            border: "1px solid var(--color-border-subtle)",
            borderRadius: "var(--radius-md)",
            padding: "4px 8px",
            width: "100%"
          }}
        >
          <Search size={14} className="text-tertiary" />
          <input
            type="text"
            className="text-xs"
            placeholder={t("common.parameterSearchPlaceholder", "Search parameters...")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              background: "transparent",
              border: "none",
              outline: "none",
              color: "var(--color-text-primary)",
              width: "100%",
              fontSize: "var(--text-xs)"
            }}
          />
        </div>
      </div>
      
      <div className="stack gap-0" style={{ display: "flex", flexDirection: "column", gap: "0" }}>
        {filteredList.length === 0 ? (
          <div className="p-4 text-center text-sm text-muted">{t("common.noParametersFound", "No parameters found.")}</div>
        ) : (
          filteredList.map((meta) => (
            <MetadataItem
              key={meta.name}
              metadata={meta}
              isOpen={openItem === meta.name}
              onToggle={() => toggleItem(meta.name)}
            />
          ))
        )}
      </div>
    </div>
  );
};
