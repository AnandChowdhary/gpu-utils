/**
 * Builds a publishable gpu-utils package.
 *
 *   node --experimental-strip-types ../../tooling/build.ts
 *
 * - Bundles src/index.ts (and any extra entries in package.json "gpuUtils.entries")
 *   into dist/*.js as a single dependency-free ES module.
 * - `.wgsl` imports are inlined as minified strings (via wgslender when available).
 * - Workspace packages such as @gpu-utils/runtime are bundled in, so published
 *   packages have zero runtime dependencies.
 * - Emits .d.ts files with tsc.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { build, type Plugin } from "esbuild";

const cwd = process.cwd();
const pkg = JSON.parse(readFileSync(resolve(cwd, "package.json"), "utf8")) as {
	name: string;
	gpuUtils?: { entries?: Record<string, string> };
};

const entries: Record<string, string> = {
	index: "src/index.ts",
	...pkg.gpuUtils?.entries,
};

let wgslenderReady: Promise<typeof import("wgslender") | null> | undefined;

async function minifyWgsl(source: string): Promise<string> {
	wgslenderReady ??= import("wgslender")
		.then(async (m) => {
			await m.initialize();
			return m;
		})
		.catch(() => null);
	const w = await wgslenderReady;
	if (!w) return source;
	const result = w.minify(source);
	if (result.errors.length > 0) {
		throw new Error(
			`WGSL minify failed: ${result.errors.map(String).join("; ")}`,
		);
	}
	return result.code;
}

const wgslPlugin: Plugin = {
	name: "wgsl-text",
	setup(b) {
		b.onLoad({ filter: /\.wgsl$/ }, async (args) => ({
			contents: `export default ${JSON.stringify(await minifyWgsl(readFileSync(args.path, "utf8")))};`,
			loader: "js",
		}));
	},
};

await build({
	entryPoints: Object.fromEntries(
		Object.entries(entries).map(([k, v]) => [k, resolve(cwd, v)]),
	),
	outdir: resolve(cwd, "dist"),
	bundle: true,
	format: "esm",
	platform: "browser",
	target: "es2022",
	sourcemap: false,
	minify: false,
	treeShaking: true,
	loader: { ".txt": "text" },
	plugins: [wgslPlugin],
	logLevel: "info",
});

execFileSync("npx", ["tsc", "-b", "tsconfig.json"], {
	cwd,
	stdio: "inherit",
});
