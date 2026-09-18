/**
 * Scaffolds a new package from tooling/template.
 *
 *   pnpm new gpu-view "Natural language to table view specs"
 */
import {
	cpSync,
	existsSync,
	readdirSync,
	readFileSync,
	renameSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { join, resolve } from "node:path";

const [, , name, description = "TODO"] = process.argv;
if (!name || !/^[a-z][a-z0-9-]*$/.test(name)) {
	console.error('Usage: pnpm new <package-name> "<description>"');
	process.exit(1);
}
const root = resolve(import.meta.dirname, "..");
const dest = resolve(root, "packages", name);
if (existsSync(dest)) {
	console.error(`${dest} already exists`);
	process.exit(1);
}
cpSync(resolve(root, "tooling/template"), dest, { recursive: true });

const pascal = name
	.split("-")
	.map((s) => s[0]!.toUpperCase() + s.slice(1))
	.join("");
const snake = name.replaceAll("-", "_");
const replace = (dir: string) => {
	for (const entry of readdirSync(dir)) {
		let p = join(dir, entry);
		if (entry.includes("__SNAKE__")) {
			const renamed = join(dir, entry.replaceAll("__SNAKE__", snake));
			renameSync(p, renamed);
			p = renamed;
		}
		if (statSync(p).isDirectory()) replace(p);
		else {
			const src = readFileSync(p, "utf8");
			writeFileSync(
				p,
				src
					.replaceAll("__NAME__", name)
					.replaceAll("__PASCAL__", pascal)
					.replaceAll("__SNAKE__", snake)
					.replaceAll("__DESCRIPTION__", description),
			);
		}
	}
};
replace(dest);

// Register in the root tsconfig solution.
const tsconfigPath = resolve(root, "tsconfig.json");
const tsconfig = JSON.parse(readFileSync(tsconfigPath, "utf8")) as {
	references: { path: string }[];
};
tsconfig.references.push({ path: `packages/${name}` });
tsconfig.references.sort((a, b) => a.path.localeCompare(b.path));
writeFileSync(tsconfigPath, `${JSON.stringify(tsconfig, null, 2)}\n`);

console.log(
	`Created packages/${name}. Next: pnpm install && pnpm --filter ${name} test`,
);
