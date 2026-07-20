import { describe, expect, test } from "bun:test";
import { readResponsePayload } from "../shared/api/apiHelpers";
import { parseApiError } from "../shared/api/errors";

describe("API error responses", () => {
  test("preserves OpenAI-style JSON details across the response helpers", async () => {
    const response = new Response(JSON.stringify({
      error: {
        message: "invalid API key",
        type: "invalid_request_error",
        param: null,
        code: "invalid_api_key",
      },
    }), {
      status: 401,
      headers: { "content-type": "application/json" },
    });

    const payload = await readResponsePayload(response);
    const error = parseApiError(response, payload);

    expect(payload).toEqual({
      error: {
        message: "invalid API key",
        type: "invalid_request_error",
        param: null,
        code: "invalid_api_key",
      },
    });
    expect(error.detail).toMatchObject({
      status: 401,
      message: "invalid API key",
      type: "invalid_request_error",
      param: null,
      code: "invalid_api_key",
    });
  });
});
