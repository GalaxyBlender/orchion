import { expect, test } from "bun:test";
import { loadPersistentState, persistentStateKey } from "../shared/storage/persistentState";

test("adds LLM defaults when loading state saved before LLM support", () => {
  const storage = new MemoryStorage();
  storage.setItem(persistentStateKey, JSON.stringify({
    version: 1,
    settings: { serverBaseUrl: "", apiKey: "key" },
    asr: {
      model: "asr",
      language: "",
      responseFormat: "json",
      prompt: "",
      temperature: "",
      timestampGranularities: [],
    },
    tts: {
      mode: "preset",
      model: "tts",
      models: { preset: "tts", clone: "", design: "" },
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
    ocr: { model: "ocr", responseFormat: "json", task: "ocr", maxTokens: "" },
    ui: { theme: "dark", activePage: "activity" },
  }));

  const state = loadPersistentState(storage);

  expect(state.settings.apiKey).toBe("key");
  expect(state.llm).toEqual({
    model: "",
    systemPrompt: "",
    temperature: "0.7",
    topP: "0.9",
    maxCompletionTokens: "512",
  });
});

test("drops a saved LLM seed while preserving supported settings", () => {
  const storage = new MemoryStorage();
  storage.setItem(persistentStateKey, JSON.stringify({
    version: 1,
    llm: {
      model: "local/llm",
      systemPrompt: "Be concise.",
      temperature: "1",
      topP: "0.8",
      maxCompletionTokens: "256",
      seed: "42",
    },
  }));

  expect(loadPersistentState(storage).llm).toEqual({
    model: "local/llm",
    systemPrompt: "Be concise.",
    temperature: "1",
    topP: "0.8",
    maxCompletionTokens: "256",
  });
});

class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();

  get length(): number {
    return this.values.size;
  }

  clear(): void {
    this.values.clear();
  }

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  key(index: number): string | null {
    return [...this.values.keys()][index] ?? null;
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}
