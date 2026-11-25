import { expect, test } from "bun:test";
import { changeModelResidency, fetchModelStatuses } from "../shared/api/client";
import type { ModelStatus } from "../shared/api/types";

test("reads residency and sends model controls with the exact service and API key", async () => {
  const settings = { serverBaseUrl: "", apiKey: "test-key" };
  const model = "example/ocr-vl";
  const unloaded: ModelStatus = { object: "model_status", id: model, service: "ocr_vl", status: "unloaded" };
  const loaded: ModelStatus = { ...unloaded, status: "loaded" };
  const requests: { path: string; method?: string; body: unknown; authorization: string | null; contentType: string | null }[] = [];
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (input, init) => {
    const headers = new Headers(init?.headers);
    requests.push({
      path: String(input),
      method: init?.method,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : null,
      authorization: headers.get("authorization"),
      contentType: headers.get("content-type"),
    });
    switch (String(input)) {
      case "/api/models/status": return Response.json({ object: "list", data: [unloaded] });
      case "/api/models/load": return Response.json(loaded);
      case "/api/models/unload": return Response.json(unloaded);
      default: return new Response(null, { status: 404 });
    }
  };

  try {
    expect((await fetchModelStatuses(settings)).data).toEqual([unloaded]);
    expect(await changeModelResidency(settings, "load", { model, service: unloaded.service })).toEqual(loaded);
    expect(await changeModelResidency(settings, "unload", { model, service: loaded.service })).toEqual(unloaded);
    expect(requests).toEqual([
      { path: "/api/models/status", method: "GET", body: null, authorization: "Bearer test-key", contentType: null },
      { path: "/api/models/load", method: "POST", body: { model, service: "ocr_vl" }, authorization: "Bearer test-key", contentType: "application/json" },
      { path: "/api/models/unload", method: "POST", body: { model, service: "ocr_vl" }, authorization: "Bearer test-key", contentType: "application/json" },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("returns the server's model control error instead of reporting success", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json(
    { error: { message: "Model is not available", code: "model_not_available" } },
    { status: 404 },
  );

  try {
    await expect(changeModelResidency(
      { serverBaseUrl: "", apiKey: "" },
      "load",
      { model: "example/tts", service: "tts" },
    )).rejects.toMatchObject({
      message: "Model is not available",
      detail: { status: 404, code: "model_not_available" },
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
