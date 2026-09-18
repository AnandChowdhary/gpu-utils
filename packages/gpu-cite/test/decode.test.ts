import { describe, expect, it } from "vitest";
import { parsePages, trimSpan } from "../src/decode.ts";
import { findArxiv, findDois, findUrls } from "../src/features.ts";

describe("trimSpan", () => {
  const t = (s: string) => {
    const [a, b] = trimSpan(s, 0, s.length);
    return s.slice(a, b);
  };
  it("strips quotes and trailing punctuation", () => {
    expect(t("“Deep residual learning,”")).toBe("Deep residual learning");
    expect(t("'Title'.")).toBe("Title");
    expect(t("*Nature*,")).toBe("Nature");
  });
  it("drops unbalanced brackets but keeps balanced ones", () => {
    expect(t("(2019).")).toBe("2019");
    expect(t("GDP per capita (current US$).")).toBe("GDP per capita (current US$)");
    expect(t("[Online]")).toBe("Online");
  });
});

describe("parsePages", () => {
  it("parses ranges with any dash", () => {
    expect(parsePages("45–67")).toEqual({ from: "45", to: "67" });
    expect(parsePages("45-67")).toEqual({ from: "45", to: "67" });
    expect(parsePages("45—67")).toEqual({ from: "45", to: "67" });
    expect(parsePages("559 to 567")).toEqual({ from: "559", to: "567" });
  });
  it("expands abbreviated end pages", () => {
    expect(parsePages("1477-81")).toEqual({ from: "1477", to: "1481" });
    expect(parsePages("239–63")).toEqual({ from: "239", to: "263" });
  });
  it("keeps article numbers and single pages", () => {
    expect(parsePages("e12345")).toEqual({ from: "e12345", to: "e12345" });
    expect(parsePages("082002")).toEqual({ from: "082002", to: "082002" });
    expect(parsePages("e12345-e12346")).toEqual({ from: "e12345", to: "e12346" });
  });
});

describe("deterministic identifiers", () => {
  it("finds DOIs with balanced parentheses and strips trailing punctuation", () => {
    const s = "doi: 10.1016/S0140-6736(20)30925-9. PMID: 1";
    const [d] = findDois(s);
    expect(s.slice(d![0], d![1])).toBe("10.1016/S0140-6736(20)30925-9");
    const s2 = "(https://doi.org/10.1000/xyz123).";
    const [d2] = findDois(s2);
    expect(s2.slice(d2![0], d2![1])).toBe("10.1000/xyz123");
  });
  it("finds new- and old-style arXiv ids only with a cue", () => {
    const s = "arXiv:1706.03762v5 and hep-ph/9911415 and 2020.12345 alone";
    expect(findArxiv(s).map(([a, b]) => s.slice(a, b))).toEqual(["1706.03762v5", "hep-ph/9911415"]);
    const u = "https://arxiv.org/abs/2306.01228";
    expect(findArxiv(u).map(([a, b]) => u.slice(a, b))).toEqual(["2306.01228"]);
  });
  it("finds URLs", () => {
    const s = "Retrieved from www.example.org/x?y=1. See https://a.b/c).";
    expect(findUrls(s).map(([a, b]) => s.slice(a, b))).toEqual(["www.example.org/x?y=1", "https://a.b/c"]);
  });
});
