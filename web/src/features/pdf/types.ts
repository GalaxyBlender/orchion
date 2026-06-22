export type PdfImageFormat = "png" | "jpeg" | "webp";

export type PdfFormState = {
  file: File | null;
  responseFormat: PdfImageFormat;
  pages: string;
  scale: string;
};

export type PdfRequestInput = {
  file: File;
  responseFormat: PdfImageFormat;
  pages?: string;
  scale?: string;
};

export type PdfResult = {
  blob: Blob;
  fileName: string;
  pageCount: string | null;
  imageCount: string | null;
  images: PdfPreviewImage[];
};

export type PdfPreviewImage = {
  name: string;
  url: string;
  size: number;
};

export type PdfParameterMetadata = {
  name: string;
  label: string;
  defaultValue: string;
  description: string;
  required: boolean;
  supported: boolean;
  options?: readonly string[];
};
