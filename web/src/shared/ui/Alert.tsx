import React, { ReactNode } from "react";
import { Info, CheckCircle2, AlertTriangle, AlertCircle, X } from "lucide-react";
import { Button } from "./Button";

export interface AlertProps {
  variant?: "info" | "success" | "warning" | "danger";
  title?: string;
  onDismiss?: () => void;
  className?: string;
  children: ReactNode;
}

export const Alert: React.FC<AlertProps> = ({
  variant = "info",
  title,
  onDismiss,
  className = "",
  children
}) => {
  const getIcon = () => {
    switch (variant) {
      case "success":
        return <CheckCircle2 size={18} className="text-success" />;
      case "warning":
        return <AlertTriangle size={18} className="text-warning" />;
      case "danger":
        return <AlertCircle size={18} className="text-danger" />;
      case "info":
      default:
        return <Info size={18} className="text-info" />;
    }
  };

  return (
    <div className={`alert alert-${variant} ${className}`}>
      <div className="alert-icon-wrapper">{getIcon()}</div>
      <div className="alert-content">
        {title && <h4 className="alert-title">{title}</h4>}
        <div className="alert-desc">{children}</div>
      </div>
      {onDismiss && (
        <Button
          variant="ghost"
          size="sm"
          className="btn-icon-only text-muted hover:text-white"
          onClick={onDismiss}
        >
          <X size={16} />
        </Button>
      )}
    </div>
  );
};
