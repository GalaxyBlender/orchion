import type { ModelObject, ModelSubtype, ModelType } from "@/shared/api/types";

export type ModelKind = ModelType | "other";

export interface ClassifiedModels {
  asr: ModelObject[];
  tts: ModelObject[];
  ttsPresetVoice: ModelObject[];
  ttsVoiceClone: ModelObject[];
  ttsVoiceDesign: ModelObject[];
  ocr: ModelObject[];
  ocrStandard: ModelObject[];
  ocrVl: ModelObject[];
  ocrLayout: ModelObject[];
  other: ModelObject[];
  all: ModelObject[];
}

export function modelKind(model: ModelObject): ModelKind {
  switch (model.type) {
    case "asr":
    case "tts":
    case "ocr":
      return model.type;
    default:
      return "other";
  }
}

export function modelSubtype(model: ModelObject): ModelSubtype | undefined {
  switch (model.subtype) {
    case "standard":
    case "vl":
    case "layout":
    case "preset_voice":
    case "voice_clone":
    case "voice_design":
      return model.subtype;
    default:
      return undefined;
  }
}

export function classifyModels(models: ModelObject[]): ClassifiedModels {
  const classified: ClassifiedModels = {
    asr: [],
    tts: [],
    ttsPresetVoice: [],
    ttsVoiceClone: [],
    ttsVoiceDesign: [],
    ocr: [],
    ocrStandard: [],
    ocrVl: [],
    ocrLayout: [],
    other: [],
    all: [...models],
  };

  for (const model of models) {
    switch (modelKind(model)) {
      case "asr":
        classified.asr.push(model);
        break;
      case "tts":
        classified.tts.push(model);
        switch (modelSubtype(model)) {
          case "preset_voice":
            classified.ttsPresetVoice.push(model);
            break;
          case "voice_clone":
            classified.ttsVoiceClone.push(model);
            break;
          case "voice_design":
            classified.ttsVoiceDesign.push(model);
            break;
        }
        break;
      case "ocr":
        classified.ocr.push(model);
        switch (modelSubtype(model)) {
          case "standard":
            classified.ocrStandard.push(model);
            break;
          case "vl":
            classified.ocrVl.push(model);
            break;
          case "layout":
            classified.ocrLayout.push(model);
            break;
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
