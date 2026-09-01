import { expect, test } from "bun:test";
import i18n, { supportedLanguages } from "../shared/i18n";

const llmKeys = [
  "shell.nav.llm.label",
  "shell.commandPalette.navLlm",
  "llm.title",
  "llm.modelPlaceholder",
  "llm.requestPreviewEyebrow",
  "llm.incomplete.stopped",
  "models.llmModels",
  "models.capabilities.llm_chat",
  "models.capabilities.llm_embeddings",
  "models.capabilities.llm_resumable_streaming",
  "activity.operations.chat",
  "activity.operations.responses",
  "activity.columns.prefill",
  "activity.columns.decode",
  "settings.llmModel",
  "settings.resetLlm",
  "settings.defaultLlmTemperature",
];

test("defines the LLM UI translations in every supported language", () => {
  for (const language of supportedLanguages) {
    for (const key of llmKeys) {
      expect(i18n.getResource(language, "translation", key), `${language}: ${key}`).toBeDefined();
    }
  }
});
