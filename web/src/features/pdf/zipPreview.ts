import { Unzip, UnzipInflate } from "fflate";
import type { PdfPreviewImage } from "./types";

type ZipPreviewResult = {
  blob: Blob;
  images: PdfPreviewImage[];
};

type ZipPreviewOptions = {
  contentType: string;
  onImage?: (image: PdfPreviewImage) => void;
};

export async function readZipWithImagePreviews(response: Response, options: ZipPreviewOptions): Promise<ZipPreviewResult> {
  const chunks: Uint8Array[] = [];
  const images: PdfPreviewImage[] = [];
  const unzip = new Unzip((file) => {
    const fileChunks: Uint8Array[] = [];

    file.ondata = (error, data, final) => {
      if (error) {
        throw error;
      }
      fileChunks.push(data);
      if (!final || !isPreviewImage(file.name)) {
        return;
      }

      const blob = blobFromChunks(fileChunks, contentTypeForFile(file.name));
      const image = {
        name: file.name,
        url: URL.createObjectURL(blob),
        size: blob.size,
      };
      images.push(image);
      options.onImage?.(image);
    };
    file.start();
  });
  unzip.register(UnzipInflate);

  if (response.body) {
    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      chunks.push(value);
      unzip.push(value);
    }
    unzip.push(new Uint8Array(), true);
  } else {
    const bytes = new Uint8Array(await response.arrayBuffer());
    chunks.push(bytes);
    unzip.push(bytes, true);
  }

  return {
    blob: blobFromChunks(chunks, options.contentType),
    images,
  };
}

export function revokePreviewImages(images: PdfPreviewImage[]): void {
  for (const image of images) {
    URL.revokeObjectURL(image.url);
  }
}

function isPreviewImage(fileName: string): boolean {
  return /\.(png|jpe?g|webp)$/i.test(fileName);
}

function contentTypeForFile(fileName: string): string {
  if (/\.png$/i.test(fileName)) {
    return "image/png";
  }
  if (/\.webp$/i.test(fileName)) {
    return "image/webp";
  }
  return "image/jpeg";
}

function blobFromChunks(chunks: Uint8Array[], contentType: string): Blob {
  return new Blob(chunks.map(copyChunkBuffer), { type: contentType });
}

function copyChunkBuffer(chunk: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(chunk.byteLength);
  copy.set(chunk);
  return copy.buffer;
}
