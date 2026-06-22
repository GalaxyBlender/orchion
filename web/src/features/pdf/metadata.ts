import type { PdfImageFormat, PdfParameterMetadata } from "./types";

export const pdfImageFormats: PdfImageFormat[] = ["png", "jpeg", "webp"];

export const pdfParameterMetadata: PdfParameterMetadata[] = [
  {
    name: "file",
    label: "PDF file",
    defaultValue: "",
    description: "PDF document to upload and convert into page images.",
    required: true,
    supported: true,
  },
  {
    name: "response_format",
    label: "Output image format",
    defaultValue: "png",
    description: "Image format for rendered PDF pages.",
    required: false,
    supported: true,
    options: pdfImageFormats,
  },
  {
    name: "pages",
    label: "Pages",
    defaultValue: "",
    description: "Optional 1-based page selectors such as 1,3-5. Leave blank to render all pages.",
    required: false,
    supported: true,
  },
  {
    name: "scale",
    label: "Render scale",
    defaultValue: "",
    description: "Optional render scale from 0.1 to 4.0. Leave blank to use the backend default.",
    required: false,
    supported: true,
  },
];
