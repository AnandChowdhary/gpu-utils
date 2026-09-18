import { describe, expect, it } from "vitest";
import {
  applyVariants,
  emit,
  isValidClass,
  literalClass,
  parseValue,
  resolveVariant,
  standalone,
  wordsOf,
} from "../src/compile.ts";
import { decodeLabels } from "../src/decode.ts";
import { featurize } from "../src/features.ts";

/** Compile from hand-written gold tags: exercises the compiler independently of the model. */
function compile(text: string, tagged: string): string[] {
  // tagged: words with role markers, e.g. "P:rounded corners|V:blue|R:on hover|S:,|N:no"
  const f = featurize(text);
  const labels = f.tokens.map(() => "O");
  const boundary = f.tokens.map(() => false);
  let cursor = 0;
  for (const piece of tagged.split("|")) {
    const [role, phrase] = [piece.slice(0, 1), piece.slice(2)];
    const at = text.indexOf(phrase, cursor);
    if (at < 0) throw new Error(`"${phrase}" not in "${text}"`);
    cursor = at + phrase.length;
    let first = true;
    f.tokens.forEach((t, i) => {
      if (t.start >= at && t.end <= cursor) {
        const r = { P: "PROP", V: "VAL", R: "VAR" }[role];
        labels[i] = role === "S" ? "SEP" : role === "N" ? "NEG" : `${first ? "B" : "I"}-${r}`;
        first = false;
      }
    });
  }
  return decodeLabels(f, labels, boundary).classes;
}

describe("features", () => {
  it("emits 7 ids per token within the table", () => {
    const f = featurize("bold red text, 16px");
    expect(f.rows).toHaveLength(f.tokens.length);
    for (const row of f.rows) {
      expect(row).toHaveLength(7);
      for (const id of row) expect(id).toBeLessThan(1565);
    }
  });
});

describe("values", () => {
  it("parses colours, sizes, numbers and keywords", () => {
    expect(parseValue("light blue")).toMatchObject({ kind: "col", value: "blue-300" });
    expect(parseValue("blue 600")).toMatchObject({ kind: "col", value: "blue-600" });
    expect(parseValue("gray-500")).toMatchObject({ kind: "col", value: "gray-500" });
    expect(parseValue("grey")).toMatchObject({ kind: "col", value: "gray-500" });
    expect(parseValue("subtle")).toMatchObject({ kind: "sz", value: "sm" });
    expect(parseValue("very large")).toMatchObject({ kind: "sz", value: "xl" });
    expect(parseValue("16px")).toMatchObject({ kind: "unit", value: "16px" });
    expect(parseValue("1/2")).toMatchObject({ kind: "frac", value: "1/2" });
    expect(parseValue("four")).toMatchObject({ kind: "num", value: "4" });
    expect(parseValue("pill")).toMatchObject({ kind: "rad", value: "full" });
    expect(parseValue("muted")).toMatchObject({ kind: "col", value: "gray-500" });
    expect(parseValue("centred")).toMatchObject({ kind: "kw", value: "center" });
    expect(parseValue("blorp")).toBeNull();
  });
  it("emits classes for (property, value) pairs", () => {
    expect(emit("shadow", parseValue("subtle"), false)).toEqual(["shadow-sm"]);
    expect(emit("rounded", null, false)).toEqual(["rounded-lg"]);
    expect(emit("rounded", parseValue("pill"), false)).toEqual(["rounded-full"]);
    expect(emit("p", parseValue("16px"), false)).toEqual(["p-[16px]"]);
    expect(emit("w", parseValue("half"), false)).toEqual(["w-1/2"]);
    expect(emit("cols", parseValue("3"), false)).toEqual(["grid", "grid-cols-3"]);
    expect(emit("text", parseValue("bold"), false)).toEqual(["font-bold"]);
    expect(emit("border", parseValue("red"), false)).toEqual(["border", "border-red-500"]);
    expect(emit("shadow", null, true)).toEqual(["shadow-none"]);
    expect(emit("border", parseValue("32"), false)).toEqual([]);
    expect(standalone(parseValue("uppercase")!, false)).toEqual(["uppercase"]);
    expect(standalone(parseValue("muted")!, false)).toEqual(["bg-gray-500"]);
  });
});

describe("variants", () => {
  const v = (s: string) => resolveVariant(wordsOf(s));
  it("resolves states and breakpoints by keyword", () => {
    expect(v("on hover")).toBe("hover");
    expect(v("when the parent is hovered")).toBe("group-hover");
    expect(v("in dark mode")).toBe("dark");
    expect(v("on mobile")).toBe("max-sm");
    expect(v("sm and up")).toBe("sm");
    expect(v("sm")).toBe("sm");
    expect(v("on desktop")).toBe("lg");
    expect(v("below md")).toBe("max-md");
    expect(v("up to lg")).toBe("max-lg");
    expect(v("on very large screens")).toBe("xl");
    expect(v("only on mobile")).toBe("only-mobile");
    expect(v("keyboard focus")).toBe("focus-visible");
    expect(v("right to left")).toBe("rtl");
    expect(v("first child")).toBe("first");
    expect(v("zebra rows")).toBe("even");
    expect(v("banana")).toBeNull();
  });
  it("stacks variants in canonical order and handles only-*", () => {
    expect(applyVariants(["bg-blue-500"], ["hover", "dark"])).toEqual(["dark:hover:bg-blue-500"]);
    expect(applyVariants(["block"], ["only-mobile"])).toEqual(["sm:hidden"]);
    expect(applyVariants(["bg-red-500"], ["only-desktop"])).toEqual(["lg:bg-red-500"]);
  });
});

describe("vocabulary", () => {
  it("validates classes against the compiled Tailwind v4 table", () => {
    for (const c of [
      "p-4",
      "px-2.5",
      "w-1/2",
      "max-w-prose",
      "text-gray-500",
      "bg-black/50",
      "rounded-lg",
      "hover:bg-blue-600",
      "dark:md:text-white",
      "max-sm:hidden",
      "grid-cols-12",
      "duration-300",
      "p-[16px]",
      "text-[10px]",
      "shadow-xs",
      "ring-2",
      "outline-hidden",
      "-mt-4",
      "size-10",
    ]) {
      expect(isValidClass(c), c).toBe(true);
    }
    for (const c of [
      "p-",
      "border-32",
      "grid-cols-48",
      "z-1.5",
      "shadow-sm-",
      "hover:",
      "hover:nope",
      "text-blue-501",
      "bg-blueish",
      "rounded-huge",
      "py-[200s]",
    ]) {
      expect(isValidClass(c), c).toBe(false);
    }
    expect(literalClass("hover:bg-blue-500")).toBe("hover:bg-blue-500");
    expect(literalClass("blue")).toBeNull();
  });
});

describe("compiler on gold tags", () => {
  it("pairs values with the nearest compatible property", () => {
    expect(compile("bold red text", "V:bold|V:red|P:text")).toEqual(["font-bold", "text-red-500"]);
    expect(compile("padding 4 margin 2", "P:padding|V:4|P:margin|V:2")).toEqual(["p-4", "m-2"]);
    expect(compile("4 columns gap 2", "V:4|P:columns|P:gap|V:2")).toEqual([
      "grid",
      "grid-cols-4",
      "gap-2",
    ]);
    expect(compile("blue background white text", "V:blue|P:background|V:white|P:text")).toEqual([
      "bg-blue-500",
      "text-white",
    ]);
  });
  it("applies variants per segment and negation to the next span", () => {
    expect(compile("blue on hover, no shadow", "V:blue|R:on hover|S:,|N:no|P:shadow")).toEqual([
      "hover:bg-blue-500",
      "shadow-none",
    ]);
    expect(compile("hidden on mobile", "V:hidden|R:on mobile")).toEqual(["max-sm:hidden"]);
    expect(compile("not bold", "N:not|V:bold")).toEqual(["font-normal"]);
  });
  it("accepts literal classes and corrects single-character typos", () => {
    expect(compile("hover:bg-blue-500 and p-4", "V:hover:bg-blue-500|S:and|V:p-4")).toEqual([
      "hover:bg-blue-500",
      "p-4",
    ]);
    expect(compile("subtle shaddow", "V:subtle|P:shaddow")).toEqual(["shadow-sm"]);
    expect(compile("roundd corners", "P:roundd corners")).toEqual(["rounded-lg"]);
  });
  it("reports diagnostics instead of guessing", () => {
    const f = featurize("blorp 4");
    const r = decodeLabels(f, ["B-PROP", "O", "B-VAL"], [false, false, false]);
    expect(r.classes).toEqual([]);
    expect(r.diagnostics.map((d) => d.message)).toContain('unknown property "blorp"');
  });
  it("returns group spans in UTF-16 offsets", () => {
    const text = "red text, blue on hover";
    const f = featurize(text);
    const r = decodeLabels(
      f,
      ["B-VAL", "O", "B-PROP", "SEP", "O", "B-VAL", "O", "B-VAR", "I-VAR", "I-VAR"],
      f.tokens.map(() => false),
    );
    expect(r.groups).toHaveLength(2);
    expect(text.slice(r.groups[1]!.span.start, r.groups[1]!.span.end)).toBe("blue on hover");
    expect(r.groups[1]!.variant).toBe("hover");
  });
});
