import { readFileSync } from "node:fs";
import type { Plugin } from "vite";

/** Lets vitest import `.wgsl` and `.txt` files as strings, matching tooling/build.ts. */
export function textImports(): Plugin {
  return {
    name: "gpu-utils:text-imports",
    enforce: "pre",
    load(id) {
      if (/\.(wgsl|txt)$/.test(id)) {
        return `export default ${JSON.stringify(readFileSync(id, "utf8"))};`;
      }
      return null;
    },
  };
}
