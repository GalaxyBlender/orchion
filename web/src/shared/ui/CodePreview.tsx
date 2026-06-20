import React, { useState } from "react";
import { Copy, Check } from "lucide-react";
import { Button } from "./Button";
import { useToast } from "./Toast";
import { useTranslation } from "react-i18next";

export interface CodePreviewProps {
  children: string;
  label?: string;
}

export const CodePreview: React.FC<CodePreviewProps> = ({
  children,
  label
}) => {
  const [copied, setCopied] = useState(false);
  const toast = useToast();
  const { t } = useTranslation();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      toast.success(t("common.copied", "Copied to clipboard!"));
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error(t("common.error", "Failed to copy"));
    }
  };

  return (
    <div className="stack gap-sm">
      <div className="hstack justify-between">
        {label && <span className="card-eyebrow">{label}</span>}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          icon={copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
          iconPosition="left"
        >
          {copied ? t("common.copied", "Copied") : t("common.copy", "Copy")}
        </Button>
      </div>
      <pre className="code-preview">
        <code>{children}</code>
      </pre>
    </div>
  );
};
