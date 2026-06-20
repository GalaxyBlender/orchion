import React, { ReactNode } from "react";
import { Loader2, Database, AlertCircle, WifiOff } from "lucide-react";

export type StateViewType = "loading" | "empty" | "error" | "offline";

export interface StateViewProps {
  type: StateViewType;
  title?: string;
  description?: string;
  message?: string; // fallback alias for description/loading text
  action?: ReactNode;
  className?: string;
}

export const StateView: React.FC<StateViewProps> = ({
  type,
  title,
  description,
  message,
  action,
  className = ""
}) => {
  const getIcon = () => {
    switch (type) {
      case "loading":
        return <Loader2 className="animate-spin text-accent" size={36} />;
      case "empty":
        return <Database className="text-muted" size={36} />;
      case "error":
        return <AlertCircle className="text-danger" size={36} />;
      case "offline":
        return <WifiOff className="text-warning" size={36} />;
    }
  };

  const getTitle = () => {
    if (title) return title;
    switch (type) {
      case "loading":
        return "Loading";
      case "empty":
        return "No Data Available";
      case "error":
        return "An Error Occurred";
      case "offline":
        return "Connection Offline";
    }
  };

  const getDesc = () => {
    if (description) return description;
    if (message) return message;
    switch (type) {
      case "loading":
        return "Please wait while we load your request...";
      case "empty":
        return "No records were found in this catalog.";
      case "error":
        return "Failed to fetch response. Check network configuration.";
      case "offline":
        return "Orchion backend server is currently offline or unreachable.";
    }
  };

  return (
    <div className={`state-view ${className}`}>
      <div className="state-view-icon-wrapper">{getIcon()}</div>
      <div className="stack gap-xs text-center">
        <h4 className="state-view-title">{getTitle()}</h4>
        <p className="state-view-desc">{getDesc()}</p>
      </div>
      {action && <div className="state-view-action">{action}</div>}
    </div>
  );
};
