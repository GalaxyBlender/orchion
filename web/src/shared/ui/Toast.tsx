import React, { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { CheckCircle2, AlertCircle, AlertTriangle, Info, X } from "lucide-react";
import { Button } from "./Button";

export type ToastType = "info" | "success" | "warning" | "error";

export interface ToastItem {
  id: string;
  message: string;
  type?: ToastType;
  duration?: number;
}

interface ToastContextType {
  showToast: (message: string, type?: ToastType, duration?: number) => void;
  success: (message: string, duration?: number) => void;
  error: (message: string, duration?: number) => void;
  warning: (message: string, duration?: number) => void;
  info: (message: string, duration?: number) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
};

export const ToastProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback(
    (message: string, type: ToastType = "info", duration = 3000) => {
      const id = Math.random().toString(36).substring(2, 9);
      setToasts((prev) => [...prev, { id, message, type, duration }]);

      if (duration > 0) {
        setTimeout(() => {
          removeToast(id);
        }, duration);
      }
    },
    [removeToast]
  );

  const success = useCallback((msg: string, dur?: number) => showToast(msg, "success", dur), [showToast]);
  const error = useCallback((msg: string, dur?: number) => showToast(msg, "error", dur), [showToast]);
  const warning = useCallback((msg: string, dur?: number) => showToast(msg, "warning", dur), [showToast]);
  const info = useCallback((msg: string, dur?: number) => showToast(msg, "info", dur), [showToast]);

  const getIcon = (type?: ToastType) => {
    switch (type) {
      case "success":
        return <CheckCircle2 size={18} className="text-success" />;
      case "warning":
        return <AlertTriangle size={18} className="text-warning" />;
      case "error":
        return <AlertCircle size={18} className="text-danger" />;
      case "info":
      default:
        return <Info size={18} className="text-info" />;
    }
  };

  return (
    <ToastContext.Provider value={{ showToast, success, error, warning, info }}>
      {children}
      <div className="toast-container" role="live" aria-live="assertive">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast-${toast.type ?? "info"}`}>
            <div className="toast-icon">{getIcon(toast.type)}</div>
            <div className="toast-message text-sm">
              {toast.message}
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="btn-icon-only toast-close"
              onClick={() => removeToast(toast.id)}
            >
              <X size={14} />
            </Button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};
