import { defineConfig } from "vitest/config";
import { textImports } from "../../tooling/vite-text.ts";

export default defineConfig({
  plugins: [textImports()],
  test: { name: "gpu-cite", include: ["test/**/*.test.ts"] },
});
