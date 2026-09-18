/**
 * gpu-view: natural language to table view specs (filter, sort, group, aggregate,
 * limit, chart). Schema-blind: the model never sees your field names, only anonymous
 * "this token matched a field of kind X" features, so it transfers to any schema.
 *
 * Public API. The featurizer lives in ./features.ts, the reference forward pass in
 * ./cpu.ts, the WebGPU forward pass in ./gpu.ts, and the compiler in ./decode.ts.
 */
import { type Backend, hasWebGPU } from "@gpu-utils/runtime";
import { forwardCpu } from "./cpu.ts";
import { type CompileOptions, decode, type ViewSpec } from "./decode.ts";
import { featurize } from "./features.ts";
import { forwardGpuBatch } from "./gpu.ts";
import type { Schema } from "./match.ts";
import { MODEL } from "./model.ts";

export type {
  AggregateFn,
  Chart,
  CompileOptions,
  DiagnosticCode,
  FilterOp,
  FilterValue,
  Granularity,
  Role,
  SortDir,
  Span,
  ViewAggregate,
  ViewDiagnostic,
  ViewFilter,
  ViewGroup,
  ViewSort,
  ViewSpec,
  ViewToken,
} from "./decode.ts";
export type { FieldKind, Schema, SchemaField } from "./match.ts";
export { resolveTime } from "./time.ts";

export interface ViewOptions extends CompileOptions {
  /** The fields the phrase may refer to. Names, aliases and enum values are matched tolerantly. */
  schema: Schema;
  /** "auto" uses WebGPU for batches when available and the CPU reference otherwise. */
  backend?: Backend;
}

export class SchemaError extends Error {
  override name = "SchemaError";
}

function checkSchema(schema: Schema | undefined): Schema {
  if (!schema || !Array.isArray(schema.fields))
    throw new SchemaError("options.schema.fields is required");
  for (const f of schema.fields) {
    if (typeof f.name !== "string" || !f.name) throw new SchemaError("every field needs a name");
    if (!["text", "number", "date", "enum", "boolean"].includes(f.kind))
      throw new SchemaError(`field "${f.name}" has unknown kind "${f.kind}"`);
  }
  return schema;
}

/** Parses one phrase. Single phrases always run on the CPU reference path unless backend is "webgpu". */
export async function parse(text: string, options: ViewOptions): Promise<ViewSpec> {
  const [spec] = await parseMany([text], options);
  return spec!;
}

/** Inputs below this many total tokens run on the CPU under "auto": GPU readback dominates. */
const GPU_MIN_TOKENS = 256;

/** Parses many phrases against one schema; batches onto the GPU when it pays off. */
export async function parseMany(texts: string[], options: ViewOptions): Promise<ViewSpec[]> {
  const schema = checkSchema(options.schema);
  const features = texts.map((t) => featurize(t, schema));
  const total = features.reduce((s, f) => s + f.tokens.length, 0);
  const backend = options.backend ?? "auto";
  const useGpu =
    backend === "webgpu" || (backend === "auto" && hasWebGPU() && total >= GPU_MIN_TOKENS);
  const logits = useGpu
    ? await forwardGpuBatch(MODEL, features)
    : features.map((f) => forwardCpu(MODEL, f));
  const compileOptions: CompileOptions = {};
  if (options.now !== undefined) compileOptions.now = options.now;
  if (options.dateField !== undefined) compileOptions.dateField = options.dateField;
  return texts.map((text, i) =>
    decode(MODEL, features[i]!, logits[i]!, text, schema, compileOptions),
  );
}
