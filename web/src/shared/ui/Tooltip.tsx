import React, { ReactNode } from "react";

export interface TooltipProps {
  content: string;
  position?: "top" | "bottom";
  children: ReactNode;
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  position = "top",
  children
}) => {
  return (
    <div className="tooltip-trigger">
      {children}
      <div className={`tooltip tooltip-${position}`}>
        {content}
      </div>
    </div>
  );
};
