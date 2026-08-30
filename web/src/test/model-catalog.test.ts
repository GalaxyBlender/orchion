import { expect, test } from "bun:test";
import { classifyModels, modelKind } from "../features/models/modelCatalog";
import type { ModelObject } from "../shared/api/types";

test("classifies LLM models independently from other models", () => {
  const llm: ModelObject = {
    id: "local/llm",
    type: "llm",
    capabilities: ["llm_chat", "llm_streaming"],
  };

  expect(modelKind(llm)).toBe("llm");
  expect(classifyModels([llm]).llm).toEqual([llm]);
  expect(classifyModels([llm]).other).toEqual([]);
});
