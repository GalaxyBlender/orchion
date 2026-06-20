import React, { ReactNode, useEffect, useRef } from "react";
import { X } from "lucide-react";
import { Button } from "./Button";

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  size?: "sm" | "md" | "lg" | "fullscreen";
  children: ReactNode;
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  size = "md",
  children
}) => {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
    } else {
      if (dialog.open) {
        dialog.close();
      }
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <dialog
        ref={dialogRef}
        className={`modal-content size-${size}`}
        onClose={onClose}
        style={{ display: "block", position: "relative" }}
      >
        <div className="modal-header card-header justify-between items-center" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 className="card-title">{title}</h3>
          <Button
            variant="ghost"
            size="sm"
            className="btn-icon-only text-muted hover:text-white"
            onClick={onClose}
          >
            <X size={18} />
          </Button>
        </div>
        <div className="modal-body card-body">{children}</div>
      </dialog>
    </div>
  );
};
