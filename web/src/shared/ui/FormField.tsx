import React, {
  InputHTMLAttributes,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
  ReactNode,
  useId,
  useState,
  useRef,
  useEffect
} from "react";
import { Upload, X, FileAudio } from "lucide-react";
import { Button } from "./Button";

// FormField Wrapper
export interface FormFieldProps {
  label: ReactNode;
  description?: string;
  error?: string;
  children: ReactNode;
  className?: string;
}

export const FormField: React.FC<FormFieldProps> = ({
  label,
  description,
  error,
  children,
  className = ""
}) => {
  return (
    <div className={`form-field ${className}`}>
      <label className="form-field-label">{label}</label>
      {description && <p className="form-field-desc">{description}</p>}
      <div className="form-field-control-wrapper">{children}</div>
      {error && <span className="form-field-error">{error}</span>}
    </div>
  );
};

// Input
export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  error?: boolean;
}

export const Input: React.FC<InputProps> = ({
  className = "",
  error,
  type = "text",
  ...props
}) => {
  return (
    <input
      type={type}
      className={`input ${error ? "has-error" : ""} ${className}`}
      {...props}
    />
  );
};

// Select
export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  error?: boolean;
  options?: readonly SelectOption[] | SelectOption[];
}

export const Select: React.FC<SelectProps> = ({
  className = "",
  error,
  options,
  children,
  ...props
}) => {
  return (
    <select
      className={`select ${error ? "has-error" : ""} ${className}`}
      {...props}
    >
      {options
        ? options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))
        : children}
    </select>
  );
};

// TextArea
export interface TextAreaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  error?: boolean;
}

export const TextArea: React.FC<TextAreaProps> = ({
  className = "",
  error,
  ...props
}) => {
  return (
    <textarea
      className={`textarea ${error ? "has-error" : ""} ${className}`}
      {...props}
    />
  );
};

// Slider
export interface SliderProps {
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (val: number) => void;
  disabled?: boolean;
  className?: string;
}

export const Slider: React.FC<SliderProps> = ({
  min,
  max,
  step = 1,
  value,
  onChange,
  disabled = false,
  className = ""
}) => {
  return (
    <div className={`slider-container ${className}`}>
      <div className="slider-wrapper">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className="slider"
        />
      </div>
      <span className="slider-value">{value.toFixed(1)}</span>
    </div>
  );
};

// FileDropZone
export interface FileDropZoneProps {
  accept?: string;
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
  dropZoneText: string;
  dropZoneActiveText: string;
  fileDescText?: string;
  className?: string;
}

export const FileDropZone: React.FC<FileDropZoneProps> = ({
  accept,
  selectedFile,
  onFileSelect,
  dropZoneText,
  dropZoneActiveText,
  fileDescText,
  className = ""
}) => {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    inputRef.current?.click();
  };

  return (
    <div className={`file-drop-zone-container ${className}`}>
      {!selectedFile ? (
        <div
          className={`file-drop-zone ${dragActive ? "drag-active" : ""}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={onButtonClick}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept={accept}
            onChange={handleChange}
          />
          <Upload className="file-drop-icon" size={28} />
          <p className="text-sm font-semibold">{dragActive ? dropZoneActiveText : dropZoneText}</p>
          {fileDescText && <p className="text-xs text-muted">{fileDescText}</p>}
        </div>
      ) : (
        <div className="file-info">
          <div className="hstack gap-sm">
            <FileAudio size={18} className="text-accent" />
            <div className="stack gap-xs">
              <span className="text-sm font-semibold truncate" style={{ maxWidth: "250px" }}>
                {selectedFile.name}
              </span>
              <span className="text-xs text-muted">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </span>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="btn-icon-only text-danger"
            onClick={() => onFileSelect(null)}
          >
            <X size={16} />
          </Button>
        </div>
      )}
    </div>
  );
};

// SuggestionInput with keyboard support and custom drop overlay
export interface SuggestionInputProps {
  id?: string;
  value: string;
  onChange: (val: string) => void;
  suggestions: string[];
  isSuggestionRecommended?: (suggestion: string) => boolean;
  placeholder?: string;
  error?: boolean;
  disabled?: boolean;
  className?: string;
}

export const SuggestionInput: React.FC<SuggestionInputProps> = ({
  id,
  value,
  onChange,
  suggestions,
  isSuggestionRecommended,
  placeholder,
  error,
  disabled,
  className = ""
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const [searchValue, setSearchValue] = useState("");
  const wrapperRef = useRef<HTMLDivElement>(null);
   
  const filteredSuggestions = React.useMemo(() => {
    if (!searchValue) return suggestions;
    const lower = searchValue.toLowerCase();
    return suggestions.filter(s => s.toLowerCase().includes(lower));
  }, [searchValue, suggestions]);

  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        setIsOpen(true);
      }
      return;
    }

    if (e.key === "Escape") {
      setIsOpen(false);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedIndex(prev => 
        prev === filteredSuggestions.length - 1 ? 0 : prev + 1
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedIndex(prev => 
        prev <= 0 ? filteredSuggestions.length - 1 : prev - 1
      );
    } else if (e.key === "Enter") {
      if (highlightedIndex >= 0 && filteredSuggestions[highlightedIndex]) {
        e.preventDefault();
        onChange(filteredSuggestions[highlightedIndex]);
        setSearchValue("");
        setIsOpen(false);
      }
    }
  };

  return (
    <div ref={wrapperRef} className={`suggestions-wrapper ${className}`}>
      <Input
        id={id}
        autoComplete="off"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          setSearchValue(e.target.value);
          setIsOpen(true);
          setHighlightedIndex(-1);
        }}
        onFocus={() => {
          setSearchValue("");
          setIsOpen(true);
          setHighlightedIndex(-1);
        }}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        error={error}
        disabled={disabled}
      />
      {isOpen && filteredSuggestions.length > 0 && (
        <div className="suggestions-dropdown">
          {filteredSuggestions.map((suggestion, idx) => (
            <div
              key={suggestion}
              className={`suggestions-item ${idx === highlightedIndex ? "active" : ""} ${isSuggestionRecommended?.(suggestion) ? "recommended" : ""}`}
              onClick={() => {
                onChange(suggestion);
                setSearchValue("");
                setIsOpen(false);
              }}
              onMouseEnter={() => setHighlightedIndex(idx)}
            >
              <span>{suggestion}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
