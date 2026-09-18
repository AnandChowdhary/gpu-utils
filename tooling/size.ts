/**
 * Measures the minified + Brotli size of each package's dist entry (weights included)
 * and compares it to the "gpuUtils.sizeBudget" (bytes) declared in package.json.
 *
 *   node --experimental-strip-types tooling/size.ts            # current package
 *   node --experimental-strip-types tooling/size.ts --all      # every package
 *   node --experimental-strip-types tooling/size.ts --all --markdown
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { brotliCompressSync, constants } from "node:zlib";
import { transformSync } from "esbuild";

type Pkg = {
  name: string;
  private?: boolean;
  gpuUtils?: { sizeBudget?: number };
};

export function measure(dir: string): { name: string; bytes: number; budget?: number } | null {
  const pkgPath = resolve(dir, "package.json");
  if (!existsSync(pkgPath)) return null;
  const pkg = JSON.parse(readFileSync(pkgPath, "utf8")) as Pkg;
  if (pkg.private) return null;
  const entry = resolve(dir, "dist/index.js");
  if (!existsSync(entry)) throw new Error(`${pkg.name}: dist/index.js missing, run build first`);
  const minified = transformSync(readFileSync(entry, "utf8"), {
    minify: true,
    format: "esm",
  }).code;
  const bytes = brotliCompressSync(Buffer.from(minified), {
    params: { [constants.BROTLI_PARAM_QUALITY]: 11 },
  }).length;
  const result: { name: string; bytes: number; budget?: number } = {
    name: pkg.name,
    bytes,
  };
  if (pkg.gpuUtils?.sizeBudget !== undefined) result.budget = pkg.gpuUtils.sizeBudget;
  return result;
}

const fmt = (b: number) => `${(b / 1024).toFixed(1)} KiB`;

if (import.meta.url === `file://${process.argv[1]}`) {
  const all = process.argv.includes("--all");
  const markdown = process.argv.includes("--markdown");
  const root = resolve(import.meta.dirname, "..");
  const dirs = all
    ? readdirSync(resolve(root, "packages")).map((d) => resolve(root, "packages", d))
    : [process.cwd()];
  const rows = dirs.map(measure).filter((r): r is NonNullable<typeof r> => r !== null);
  let failed = false;
  if (markdown) console.log("| Package | Size | Budget |\n|---|---|---|");
  for (const r of rows) {
    const over = r.budget !== undefined && r.bytes > r.budget;
    failed ||= over;
    if (markdown) {
      console.log(
        `| ${r.name} | ${fmt(r.bytes)} | ${r.budget ? fmt(r.budget) : "—"} ${over ? "❌" : ""} |`,
      );
    } else {
      console.log(
        `${r.name}: ${fmt(r.bytes)}${r.budget ? ` / budget ${fmt(r.budget)}` : ""}${over ? "  OVER BUDGET" : ""}`,
      );
    }
  }
  if (failed) process.exit(1);
}
