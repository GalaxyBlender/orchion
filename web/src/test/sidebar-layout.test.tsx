import { expect, mock, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";

mock.module("@/shared/i18n", () => ({
  currentLanguageSelection: () => "auto",
  setLanguageSelection: () => {},
}));

mock.module("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}));

async function renderSidebar(collapsed: boolean): Promise<string> {
  const { Sidebar } = await import("../shared/ui/Sidebar");

  return renderToStaticMarkup(
    <MemoryRouter>
      <Sidebar collapsed={collapsed} onToggleCollapse={() => {}} />
    </MemoryRouter>,
  );
}

test("collapsed sidebar uses compact layout classes", async () => {
  const html = await renderSidebar(true);

  expect(html).toContain("sidebar-inner");
  expect(html).toContain("sidebar-inner-collapsed");
  expect(html).toContain("brand-icon-collapsed");
  expect(html).toContain("nav-link-collapsed");
});
