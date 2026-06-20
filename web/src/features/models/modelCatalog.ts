import type { ModelObject } from "@/shared/api/types";

export type ModelKind = "asr" | "tts" | "other";

export interface ClassifiedModels {
  asr: ModelObject[];
  tts: ModelObject[];
  other: ModelObject[];
  all: ModelObject[];
}

export function modelKind(model: ModelObject | string): ModelKind {
  const modelId = typeof model === "string" ? model : model.id;
  const normalizedModelId = modelId.toLowerCase();

  if (normalizedModelId.includes("asr")) {
    return "asr";
  }

  if (
    normalizedModelId.includes("tts") ||
    normalizedModelId.includes("speech") ||
    normalizedModelId.includes("voice")
  ) {
    return "tts";
  }

  return "other";
}

export function classifyModels(models: ModelObject[]): ClassifiedModels {
  const classified: ClassifiedModels = {
    asr: [],
    tts: [],
    other: [],
    all: [...models],
  };

  for (const model of models) {
    classified[modelKind(model)].push(model);
  }

  return classified;
}

export function modelIds(models: ModelObject[]): string[] {
  return models.map((model) => model.id);
}
