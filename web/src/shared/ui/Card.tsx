import React, { ReactNode, HTMLAttributes } from "react";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "elevated" | "glass";
  interactive?: boolean;
  children: ReactNode;
}

export const Card: React.FC<CardProps> & {
  Header: React.FC<CardHeaderProps>;
  Body: React.FC<CardBodyProps>;
  Footer: React.FC<CardFooterProps>;
} = ({
  variant = "default",
  interactive = false,
  className = "",
  children,
  ...props
}) => {
  const cardClasses = [
    "card",
    `card-${variant}`,
    interactive ? "card-interactive" : "",
    className
  ].filter(Boolean).join(" ");

  return (
    <div className={cardClasses} {...props}>
      {children}
    </div>
  );
};

export interface CardHeaderProps extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  eyebrow?: string;
  title?: ReactNode;
  children?: ReactNode;
}

const CardHeader: React.FC<CardHeaderProps> = ({
  eyebrow,
  title,
  className = "",
  children,
  ...props
}) => {
  return (
    <div className={`card-header ${className}`} {...props}>
      <div className="card-title-group">
        {eyebrow && <span className="card-eyebrow">{eyebrow}</span>}
        {title && <h3 className="card-title">{title}</h3>}
      </div>
      {children && <div className="card-header-actions">{children}</div>}
    </div>
  );
};

export interface CardBodyProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

const CardBody: React.FC<CardBodyProps> = ({
  className = "",
  children,
  ...props
}) => {
  return (
    <div className={`card-body ${className}`} {...props}>
      {children}
    </div>
  );
};

export interface CardFooterProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

const CardFooter: React.FC<CardFooterProps> = ({
  className = "",
  children,
  ...props
}) => {
  return (
    <div className={`card-footer ${className}`} {...props}>
      {children}
    </div>
  );
};

Card.Header = CardHeader;
Card.Body = CardBody;
Card.Footer = CardFooter;
