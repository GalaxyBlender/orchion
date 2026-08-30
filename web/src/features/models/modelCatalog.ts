import type { ModelCapability, ModelObject, ModelType } from "@/shared/api/types";

export type ModelKind = ModelType | "other";

export interface ClassifiedModels {
  asr: ModelObject[];
  llm: ModelObject[];
  tts: ModelObject[];
  ttsPresetVoice: ModelObject[];
  ttsVoiceClone: ModelObject[];
  ttsVoiceDesign: ModelObject[];
  ocr: ModelObject[];
  ocrStandard: ModelObject[];
  ocrVl: ModelObject[];
  other: ModelObject[];
  all: ModelObject[];
}

export function modelKind(model: ModelObject): ModelKind {
  switch (model.type) {
    case "asr":
    case "llm":
    case "tts":
    case "ocr":
      return model.type;
    default:
      return "other";
  }
}

export function hasCapability(model: ModelObject, capability: ModelCapability): boolean {
  return model.capabilities.includes(capability);
}

export function classifyModels(models: ModelObject[]): ClassifiedModels {
  const classified: ClassifiedModels = {
    asr: [],
    llm: [],
    tts: [],
    ttsPresetVoice: [],
    ttsVoiceClone: [],
    ttsVoiceDesign: [],
    ocr: [],
    ocrStandard: [],
    ocrVl: [],
    other: [],
    all: [...models],
  };

  for (const model of models) {
    switch (modelKind(model)) {
      case "asr":
        classified.asr.push(model);
        break;
      case "llm":
        classified.llm.push(model);
        break;
      case "tts":
        classified.tts.push(model);
        if (hasCapability(model, "tts_preset_speakers")) classified.ttsPresetVoice.push(model);
        if (hasCapability(model, "tts_voice_cloning")) classified.ttsVoiceClone.push(model);
        if (hasCapability(model, "tts_voice_design")) classified.ttsVoiceDesign.push(model);
        break;
      case "ocr":
        classified.ocr.push(model);
        if (hasCapability(model, "ocr_vision_language")) {
          classified.ocrVl.push(model);
        } else if (hasCapability(model, "ocr_text")) {
          classified.ocrStandard.push(model);
        }
        break;
      case "other":
        classified.other.push(model);
        break;
    }
  }

  return classified;
}

export function modelIds(models: ModelObject[]): string[] {
  return models.map((model) => model.id);
}
