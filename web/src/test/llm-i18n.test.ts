import { expect, test } from "bun:test";
import i18n, { supportedLanguages } from "../shared/i18n";

const llmKeys = [
  "shell.nav.llm.label",
  "shell.commandPalette.navLlm",
  "llm.title",
  "llm.validation.seed",
  "llm.incomplete.stopped",
  "models.llmModels",
  "models.capabilities.llm_chat",
  "activity.operations.chat",
  "activity.operations.responses",
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
