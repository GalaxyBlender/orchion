import { apiUrl } from "@/shared/api/client";
import type { ApiSettings } from "@/shared/api/types";
import type { TtsJsonBody, TtsPayload, TtsRequestInput } from "./types";

const endpointPath = "/v1/audio/speech";

const numericFields = [
  ["speed", "speed"],
  ["seed", "seed"],
  ["temperature", "temperature"],
  ["top_k", "topK"],
  ["top_p", "topP"],
  ["repetition_penalty", "repetitionPenalty"],
  ["max_length", "maxLength"],
] as const;

type NumericApiName = (typeof numericFields)[number][0];
type NumericInputName = (typeof numericFields)[number][1];

export function buildTtsPayload(input: TtsRequestInput): TtsPayload {
  if (input.mode === "clone") {
    return buildClonePayload(input);
  }

  const body = buildJsonBody(input);
  if (input.mode === "design") {
    const voicePrompt = input.voicePrompt.trim();
    if (voicePrompt !== "") {
      body.voice_prompt = voicePrompt;
    }
  }

  return {
    kind: "json",
    body,
    headers: { "Content-Type": "application/json" },
  };
}

interface TtsSummaryText {
  modeClone: string;
  modeDesign: string;
  modePreset: string;
  model: (model: string) => string;
  format: (format: string) => string;
  language: (language: string) => string;
  speaker: (speaker: string) => string;
  referenceAudio: (value: string) => string;
  referenceText: (value: string) => string;
  voicePrompt: (value: string) => string;
  speed: (speed: string) => string;
  sampling: (seed: string, temperature: string, topK: string, topP: string, repetitionPenalty: string, maxLength: string) => string;
  constraints: string;
  notSelected: string;
  omitted: string;
  sent: string;
}

const defaultSummaryText: TtsSummaryText = {
  modeClone: "Mode: voice clone (multipart reference audio)",
  modeDesign: "Mode: voice design (JSON voice_prompt)",
  modePreset: "Mode: preset voice",
  model: (model) => `Model: ${model}`,
  format: (format) => `Format: ${format}`,
  language: (language) => `Language: ${language === "" ? "auto" : language}`,
  speaker: (speaker) => `Speaker: ${speaker}`,
  referenceAudio: (value) => `Reference audio: ${value}`,
  referenceText: (value) => `Reference text: ${value}`,
  voicePrompt: (value) => `Voice prompt: ${value}`,
  speed: (speed) => `Speed: ${speed} (server currently only accepts 1.0)`,
  sampling: (seed, temperature, topK, topP, repetitionPenalty, maxLength) =>
    `Sampling: seed ${seed}, temperature ${temperature}, top_k ${topK}, top_p ${topP}, repetition_penalty ${repetitionPenalty}, max_length ${maxLength}`,
  constraints: "Constraints: top_p must be between 0 and 1; top_k, temperature, repetition_penalty, and max_length must be positive.",
  notSelected: "not selected",
  omitted: "omitted",
  sent: "sent",
};

export function summarizeTtsRequest(input: TtsRequestInput, text: TtsSummaryText = defaultSummaryText): string[] {
  const lines = [
    modeSummary(input, text),
    text.model(input.model),
    text.format(input.responseFormat),
    text.language(input.language.trim()),
  ];

  if (input.mode === "preset") {
    lines.push(text.speaker(input.speaker.trim()));
  }
  if (input.mode === "clone") {
    const referenceAudioName = input.referenceAudio?.name ?? input.referenceAudioName.trim();
    const referenceText = input.referenceText.trim();
    lines.push(text.referenceAudio(referenceAudioName === "" ? text.notSelected : referenceAudioName));
    lines.push(text.referenceText(referenceText === "" ? text.omitted : text.sent));
  }
  if (input.mode === "design") {
    lines.push(text.voicePrompt(input.voicePrompt.trim() === "" ? text.omitted : text.sent));
  }

  lines.push(text.speed(input.speed.trim()));
  lines.push(text.sampling(input.seed.trim(), input.temperature.trim(), input.topK.trim(), input.topP.trim(), input.repetitionPenalty.trim(), input.maxLength.trim()));
  lines.push(text.constraints);

  return lines;
}

export function buildTtsCurl(settings: ApiSettings, input: TtsRequestInput): string {
  const payload = buildTtsPayload(input);
  const lines = [`curl -X POST ${quote(apiUrl(settings, endpointPath))}`];
  const apiKey = settings.apiKey.trim();

  if (apiKey !== "") {
    lines.push(`-H ${quote(`Authorization: Bearer ${apiKey}`)}`);
  }

  if (payload.kind === "json") {
    lines.push(`-H ${quote("Content-Type: application/json")}`);
    lines.push(`--data ${quote(JSON.stringify(payload.body))}`);
    return joinCurlLines(lines);
  }

  const referenceAudioName = input.referenceAudio?.name ?? input.referenceAudioName.trim();
  if (referenceAudioName !== "") {
    lines.push(`-F ${quote(`reference_audio=@${referenceAudioName}`)}`);
  }

  for (const [name, value] of cloneCurlFields(input)) {
    lines.push(`-F ${quote(`${name}=${value}`)}`);
  }

  return joinCurlLines(lines);
}

function joinCurlLines(lines: readonly string[]): string {
  return lines.map((line, index) => (index === lines.length - 1 ? line : `${line} \\`)).join("\n");
}

function buildClonePayload(input: TtsRequestInput): TtsPayload {
  const formData = new FormData();
  const referenceText = input.referenceText.trim();

  formData.append("model", input.model);
  formData.append("input", input.input);
  formData.append("voice", "clone");
  formData.append("response_format", input.responseFormat);
  appendNonblankStringField(formData, "language", input.language);
  appendNumericFields(formData, input);
  if (input.referenceAudio) {
    formData.append("reference_audio", input.referenceAudio);
  }
  if (referenceText !== "") {
    formData.append("reference_text", referenceText);
  }

  return {
    kind: "multipart",
    formData,
    headers: {},
  };
}

function buildJsonBody(input: TtsRequestInput): TtsJsonBody {
  const body: TtsJsonBody = {
    model: input.model,
    input: input.input,
    voice: input.mode === "design" ? "design" : input.speaker.trim(),
    response_format: input.responseFormat,
  };
  appendNonblankStringField(body, "language", input.language);

  appendNumericFields(body, input);

  return body;
}

function appendNumericFields(target: TtsJsonBody | FormData, input: TtsRequestInput): void {
  for (const [apiName, inputName] of numericFields) {
    appendNumericField(target, apiName, input[inputName]);
  }
}

function appendNumericField(target: TtsJsonBody | FormData, name: NumericApiName, value: string): void {
  const numericValue = parseOptionalNumber(value);
  if (numericValue === undefined) {
    return;
  }

  if (target instanceof FormData) {
    target.append(name, String(numericValue));
    return;
  }

  target[name] = numericValue;
}

function cloneCurlFields(input: TtsRequestInput): Array<[string, string]> {
  const fields: Array<[string, string]> = [
    ["model", input.model],
    ["input", input.input],
    ["voice", "clone"],
    ["response_format", input.responseFormat],
  ];
  const referenceText = input.referenceText.trim();

  pushOptionalStringField(fields, "language", input.language);

  if (referenceText !== "") {
    fields.push(["reference_text", referenceText]);
  }
  for (const [apiName, inputName] of numericFields) {
    pushNumericCurlField(fields, apiName, input[inputName]);
  }

  return fields;
}

function pushNumericCurlField(fields: Array<[string, string]>, name: NumericApiName, value: string): void {
  const numericValue = parseOptionalNumber(value);
  if (numericValue !== undefined) {
    fields.push([name, String(numericValue)]);
  }
}

function appendNonblankStringField(target: TtsJsonBody | FormData, name: "language", value: string): void {
  const trimmedValue = value.trim();
  if (trimmedValue === "") {
    return;
  }
  if (target instanceof FormData) {
    target.append(name, trimmedValue);
    return;
  }
  target[name] = trimmedValue;
}

function pushOptionalStringField(fields: Array<[string, string]>, name: string, value: string): void {
  const trimmedValue = value.trim();
  if (trimmedValue !== "") {
    fields.push([name, trimmedValue]);
  }
}

function parseOptionalNumber(value: string): number | undefined {
  const trimmedValue = value.trim();
  return trimmedValue === "" ? undefined : Number(trimmedValue);
}

function modeSummary(input: TtsRequestInput, text: TtsSummaryText): string {
  if (input.mode === "clone") {
    return text.modeClone;
  }
  if (input.mode === "design") {
    return text.modeDesign;
  }
  return text.modePreset;
}

function quote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}
