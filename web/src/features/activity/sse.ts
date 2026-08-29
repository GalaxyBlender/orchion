export function createSseParser(onEvent: (eventType: string, data: string) => void) {
  let buffer = "";

  const consumeBlock = (block: string) => {
    let eventType = "message";
    const data: string[] = [];
    for (const rawLine of block.split(/\r?\n/)) {
      if (rawLine === "" || rawLine.startsWith(":")) continue;
      const separator = rawLine.indexOf(":");
      const field = separator < 0 ? rawLine : rawLine.slice(0, separator);
      let value = separator < 0 ? "" : rawLine.slice(separator + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") eventType = value;
      if (field === "data") data.push(value);
    }
    if (data.length > 0) onEvent(eventType, data.join("\n"));
  };

  const drain = () => {
    while (true) {
      const match = /\r?\n\r?\n/.exec(buffer);
      if (!match || match.index === undefined) return;
      const block = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      consumeBlock(block);
    }
  };

  return {
    push(chunk: string) {
      buffer += chunk;
      drain();
    },
    finish() {
      if (buffer.trim() !== "") consumeBlock(buffer);
      buffer = "";
    },
  };
}
