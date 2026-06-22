export const persistentStateKey = "orchion.webui.state.v1";
export const persistentStateVersion = 1;

export interface PersistentSettings {
  serverBaseUrl: string;
  apiKey: string;
}

export interface PersistentAsrState {
  model: string;
  language: string;
  responseFormat: "json" | "text" | "verbose_json" | "srt";
  prompt: string;
  temperature: string;
  timestampGranularities: string[];
}

export type TtsMode = "preset" | "clone" | "design";

export type PersistentTtsModels = Record<TtsMode, string>;

export interface PersistentTtsState {
  mode: TtsMode;
  model: string;
  models: PersistentTtsModels;
  input: string;
  language: string;
  responseFormat: "wav" | "mp3" | "aac" | "opus" | "flac" | "pcm";
  speaker: string;
  referenceText: string;
  voicePrompt: string;
  speed: string;
  seed: string;
  temperature: string;
  topK: string;
  topP: string;
  repetitionPenalty: string;
  maxLength: string;
}

export interface PersistentOcrState {
  model: string;
  responseFormat: "json" | "text" | "markdown" | "html";
  task: "ocr" | "table" | "formula" | "chart" | "spotting" | "seal";
  layoutModel: string;
  maxTokens: string;
}

export interface PersistentUiState {
  theme: "dark";
  activePage: "asr" | "tts" | "ocr" | "models" | "settings";
  language?: "en" | "zh-CN" | "zh-TW";
}

export interface PersistentState {
  version: 1;
  settings: PersistentSettings;
  asr: PersistentAsrState;
  tts: PersistentTtsState;
  ocr: PersistentOcrState;
  ui: PersistentUiState;
}

export const defaultPersistentState: PersistentState = createDefaultPersistentState();

export function cloneDefaultPersistentState(): PersistentState {
  return createDefaultPersistentState();
}

export function loadPersistentState(storage?: Storage): PersistentState {
  const selectedStorage = resolveStorage(storage);
  if (!selectedStorage) {
    return cloneDefaultPersistentState();
  }

  let storedState: string | null;
  try {
    storedState = selectedStorage.getItem(persistentStateKey);
  } catch {
    return cloneDefaultPersistentState();
  }

  if (storedState === null) {
    return cloneDefaultPersistentState();
  }

  try {
    return mergePersistentState(JSON.parse(storedState));
  } catch {
    return cloneDefaultPersistentState();
  }
}

export function savePersistentState(state: PersistentState, storage?: Storage): void {
  const selectedStorage = resolveStorage(storage);
  if (!selectedStorage || !isPersistentState(state)) {
    return;
  }

  try {
    selectedStorage.setItem(persistentStateKey, JSON.stringify(state));
  } catch {
    return;
  }
}

export function resetPersistentState(storage?: Storage): PersistentState {
  const selectedStorage = resolveStorage(storage);
  if (selectedStorage) {
    try {
      selectedStorage.removeItem(persistentStateKey);
    } catch {
      return cloneDefaultPersistentState();
    }
  }

  return cloneDefaultPersistentState();
}

type PartialPersistentState = {
  settings?: Partial<PersistentSettings>;
  asr?: Partial<PersistentAsrState>;
  tts?: Partial<Omit<PersistentTtsState, "models">> & { models?: PersistentTtsModels };
  ocr?: Partial<PersistentOcrState>;
  ui?: Partial<PersistentUiState>;
};

type StringField<T> = {
  [K in keyof T]: T[K] extends string ? K : never;
}[keyof T];

function createDefaultPersistentState(): PersistentState {
  return {
    version: persistentStateVersion,
    settings: {
      serverBaseUrl: "",
      apiKey: "",
    },
    asr: {
      model: "",
      language: "",
      responseFormat: "json",
      prompt: "",
      temperature: "",
      timestampGranularities: [],
    },
    tts: {
      mode: "preset",
      model: "",
      models: {
        preset: "",
        clone: "",
        design: "",
      },
      input: "",
      language: "",
      responseFormat: "wav",
      speaker: "Serena",
      referenceText: "",
      voicePrompt: "",
      speed: "1.0",
      seed: "42",
      temperature: "0.7",
      topK: "20",
      topP: "0.8",
      repetitionPenalty: "1.05",
      maxLength: "2048",
    },
    ocr: {
      model: "",
      responseFormat: "json",
      task: "ocr",
      layoutModel: "",
      maxTokens: "",
    },
    ui: {
      theme: "dark",
      activePage: "asr",
    },
  };
}

function resolveStorage(storage?: Storage): Storage | undefined {
  if (storage) {
    return storage;
  }

  if (typeof window === "undefined") {
    return undefined;
  }

  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}

function mergePersistentState(value: unknown): PersistentState {
  const parsedState = parsePartialPersistentState(value);
  if (!parsedState) {
    return cloneDefaultPersistentState();
  }

  const defaults = cloneDefaultPersistentState();
  return {
    version: persistentStateVersion,
    settings: { ...defaults.settings, ...parsedState.settings },
    asr: { ...defaults.asr, ...parsedState.asr },
    tts: {
      ...defaults.tts,
      ...parsedState.tts,
      models: {
        ...defaults.tts.models,
        ...parsedState.tts?.models,
      },
    },
    ocr: { ...defaults.ocr, ...parsedState.ocr },
    ui: { ...defaults.ui, ...parsedState.ui },
  };
}

function parsePartialPersistentState(value: unknown): PartialPersistentState | null {
  if (!isRecord(value) || value.version !== persistentStateVersion) {
    return null;
  }

  const settings = parseSettings(value.settings);
  const asr = parseAsrState(value.asr);
  const tts = parseTtsState(value.tts);
  const ocr = parseOcrState(value.ocr);
  const ui = parseUiState(value.ui);

  if (settings === null || asr === null || tts === null || ocr === null || ui === null) {
    return null;
  }

  return { settings, asr, tts, ocr, ui };
}

function parseSettings(value: unknown): Partial<PersistentSettings> | null {
  return parseStringFields<PersistentSettings>(value, ["serverBaseUrl", "apiKey"]);
}

function parseAsrState(value: unknown): Partial<PersistentAsrState> | null {
  const state = parseStringFields<PersistentAsrState>(value, ["model", "language", "prompt", "temperature"]);
  if (state === null || value === undefined) {
    return state;
  }

  if (!isRecord(value)) {
    return null;
  }

  if ("responseFormat" in value) {
    if (!isOneOf(value.responseFormat, ["json", "text", "verbose_json", "srt"])) {
      return null;
    }
    state.responseFormat = value.responseFormat;
  }

  if ("timestampGranularities" in value) {
    if (!isStringArray(value.timestampGranularities)) {
      return null;
    }
    state.timestampGranularities = [...value.timestampGranularities];
  }

  return state;
}

function parseTtsState(value: unknown): Partial<PersistentTtsState> | null {
  const state = parseStringFields<PersistentTtsState>(value, [
    "model",
    "input",
    "language",
    "speaker",
    "referenceText",
    "voicePrompt",
    "speed",
    "seed",
    "temperature",
    "topK",
    "topP",
    "repetitionPenalty",
    "maxLength",
  ]);
  if (state === null || value === undefined) {
    return state;
  }

  if (!isRecord(value)) {
    return null;
  }

  if ("mode" in value) {
    if (!isOneOf(value.mode, ["preset", "clone", "design"])) {
      return null;
    }
    state.mode = value.mode;
  }

  if ("models" in value) {
    const models = parseTtsModels(value.models);
    if (models === null) {
      return null;
    }
    state.models = models;
  }

  if ("responseFormat" in value) {
    if (!isOneOf(value.responseFormat, ["wav", "mp3", "aac", "opus", "flac", "pcm"])) {
      return null;
    }
    state.responseFormat = value.responseFormat;
  }

  return state;
}

function parseTtsModels(value: unknown): PersistentTtsModels | null {
  const models = parseStringFields<PersistentTtsModels>(value, ["preset", "clone", "design"]);
  if (models === null) {
    return null;
  }
  if (typeof models.preset !== "string" || typeof models.clone !== "string" || typeof models.design !== "string") {
    return null;
  }
  return {
    preset: models.preset,
    clone: models.clone,
    design: models.design,
  };
}

function parseOcrState(value: unknown): Partial<PersistentOcrState> | null {
  const state = parseStringFields<PersistentOcrState>(value, ["model", "layoutModel", "maxTokens"]);
  if (state === null || value === undefined) {
    return state;
  }

  if (!isRecord(value)) {
    return null;
  }

  if ("responseFormat" in value) {
    if (!isOneOf(value.responseFormat, ["json", "text", "markdown", "html"])) {
      return null;
    }
    state.responseFormat = value.responseFormat;
  }

  if ("task" in value) {
    if (!isOneOf(value.task, ["ocr", "table", "formula", "chart", "spotting", "seal"])) {
      return null;
    }
    state.task = value.task;
  }

  return state;
}

function parseUiState(value: unknown): Partial<PersistentUiState> | null {
  if (value === undefined) {
    return {};
  }

  if (!isRecord(value)) {
    return null;
  }

  const state: Partial<PersistentUiState> = {};
  if ("theme" in value) {
    if (value.theme !== "dark") {
      return null;
    }
    state.theme = value.theme;
  }

  if ("activePage" in value) {
    if (!isOneOf(value.activePage, ["asr", "tts", "ocr", "models", "settings"])) {
      return null;
    }
    state.activePage = value.activePage;
  }

  if ("language" in value) {
    if (!isOneOf(value.language, ["en", "zh-CN", "zh-TW"])) {
      return null;
    }
    state.language = value.language;
  }

  return state;
}

function parseStringFields<T extends object>(
  value: unknown,
  fields: ReadonlyArray<StringField<T>>,
): Partial<T> | null {
  if (value === undefined) {
    return {};
  }

  if (!isRecord(value)) {
    return null;
  }

  const parsedValue: Partial<T> = {};
  for (const field of fields) {
    if (!(field in value)) {
      continue;
    }

    const fieldValue = value[field as string];
    if (typeof fieldValue !== "string") {
      return null;
    }

    parsedValue[field] = fieldValue as T[typeof field];
  }

  return parsedValue;
}

function isPersistentState(value: unknown): value is PersistentState {
  if (!isRecord(value) || value.version !== persistentStateVersion) {
    return false;
  }

  const settings = parseSettings(value.settings);
  const asr = parseAsrState(value.asr);
  const tts = parseTtsState(value.tts);
  const ocr = parseOcrState(value.ocr);
  const ui = parseUiState(value.ui);

  return (
    isRecord(value.settings) &&
    isRecord(value.asr) &&
    isRecord(value.tts) &&
    isRecord(value.ocr) &&
    isRecord(value.ui) &&
    settings !== null &&
    asr !== null &&
    tts !== null &&
    ocr !== null &&
    ui !== null &&
    hasStringFields(settings, ["serverBaseUrl", "apiKey"]) &&
    hasStringFields(asr, ["model", "language", "responseFormat", "prompt", "temperature"]) &&
    Array.isArray(asr.timestampGranularities) &&
    hasStringFields(tts, [
      "mode",
      "model",
      "input",
      "language",
      "responseFormat",
      "speaker",
      "referenceText",
      "voicePrompt",
      "speed",
      "seed",
      "temperature",
      "topK",
      "topP",
      "repetitionPenalty",
      "maxLength",
    ]) &&
    (!("models" in value.tts) || isTtsModels(value.tts.models)) &&
    hasStringFields(ocr, ["model", "responseFormat", "task", "layoutModel", "maxTokens"]) &&
    hasStringFields(ui, ["theme", "activePage"]) &&
    (!("language" in value.ui) || isOneOf(value.ui.language, ["en", "zh-CN", "zh-TW"]))
  );
}

function isTtsModels(value: unknown): value is PersistentTtsModels {
  return isRecord(value) && hasStringFields(value, ["preset", "clone", "design"]);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isOneOf<const T extends string>(value: unknown, allowedValues: readonly T[]): value is T {
  return typeof value === "string" && allowedValues.includes(value as T);
}

function hasStringFields(value: Record<string, unknown>, fields: readonly string[]): boolean {
  return fields.every((field) => typeof value[field] === "string");
}
