/**
 * Runs `npm pack --dry-run --json` for the current package and asserts the tarball
 * contains only dist/, README.md, MODEL_CARD.md, LICENSE and package.json, and that
 * the package declares no runtime dependencies.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const cwd = process.cwd();
const pkg = JSON.parse(readFileSync(resolve(cwd, "package.json"), "utf8")) as {
  name: string;
  private?: boolean;
  dependencies?: Record<string, string>;
};
if (pkg.private) process.exit(0);

if (pkg.dependencies && Object.keys(pkg.dependencies).length > 0) {
  console.error(`${pkg.name}: published packages must have zero runtime dependencies`);
  process.exit(1);
}

const out = execFileSync("npm", ["pack", "--dry-run", "--json"], { cwd, encoding: "utf8" });
const [info] = JSON.parse(out) as [{ files: { path: string; size: number }[]; size: number }];
const allowed = /^(dist\/[^/]+\.(js|d\.ts)|README\.md|MODEL_CARD\.md|LICENSE|package\.json)$/;
const bad = info!.files.filter((f) => !allowed.test(f.path));
for (const f of info!.files) console.log(`  ${f.path} (${f.size} B)`);
if (bad.length) {
  console.error(`${pkg.name}: unexpected files in tarball: ${bad.map((f) => f.path).join(", ")}`);
  process.exit(1);
}
for (const required of ["README.md", "MODEL_CARD.md", "LICENSE"]) {
  if (!info!.files.some((f) => f.path === required)) {
    console.error(`${pkg.name}: ${required} missing from tarball`);
    process.exit(1);
  }
}
console.log(`${pkg.name}: ${info!.files.length} files, ${info!.size} B packed`);
