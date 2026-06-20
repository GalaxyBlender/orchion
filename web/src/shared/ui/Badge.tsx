import React, { ReactNode, HTMLAttributes } from "react";

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "accent" | "accent-blue" | "success" | "warning" | "danger";
  pulse?: boolean;
  dot?: boolean;
  children?: ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = "default",
  pulse = false,
  dot = false,
  className = "",
  children,
  ...props
}) => {
  const badgeClasses = [
    "badge",
    `badge-${variant}`,
    pulse ? "badge-pulse" : "",
    className
  ].filter(Boolean).join(" ");

  return (
    <span className={badgeClasses} {...props}>
      {dot && <span className="badge-dot" />}
      {children}
    </span>
  );
};
