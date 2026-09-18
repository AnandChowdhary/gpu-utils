import { describe, expect, it } from "vitest";
import { parse, parseMany } from "../src/index.ts";

const GMAIL = `Hi Bob,

Thanks for the update, that works for me.

Best,
John Doe
CEO, Acme Inc
+1 (555) 123-4567
john@acme.com
https://www.acme.com

On Mon, Jan 5, 2024 at 3:14 PM Bob Smith <bob@example.com> wrote:
> Hi John,
>
> Can we move the meeting?
>
> Bob
`;

describe("gpu-email parse", () => {
  it("splits a Gmail-style top-posted reply", async () => {
    const r = await parse(GMAIL, { backend: "cpu" });
    expect(r.reply).toBe("Hi Bob,\n\nThanks for the update, that works for me.\n\nBest,");
    const kinds = r.segments.map((s) => s.kind);
    expect(kinds).toEqual(["greeting", "reply", "closing", "signature", "attribution", "quote"]);
    // Segments are ordered, non-overlapping, and quote lines are covered by the exact rule.
    for (let i = 1; i < r.segments.length; i++) {
      expect(r.segments[i]!.span[0]).toBeGreaterThanOrEqual(r.segments[i - 1]!.span[1]);
    }
    const quote = r.segments.find((s) => s.kind === "quote")!;
    expect(GMAIL.slice(quote.span[0], quote.span[1]).startsWith("> Hi John,")).toBe(true);
    expect(r.contact?.name).toBe("John Doe");
    expect(r.contact?.email).toEqual(["john@acme.com"]);
    expect(r.contact?.url).toEqual(["https://www.acme.com"]);
    expect(r.contact?.phone).toEqual(["+1 (555) 123-4567"]);
    expect(r.contact?.title).toBe("CEO");
    expect(r.contact?.company).toBe("Acme Inc");
    expect(r.diagnostics.backend).toBe("cpu");
    expect(r.diagnostics.lines).toBe(17);
  });

  it("treats '-- ' as an exact signature delimiter", async () => {
    const r = await parse("this is an email with a correct -- signature.\n\n-- \nrick\n", {
      backend: "cpu",
    });
    expect(r.reply).toBe("this is an email with a correct -- signature.");
    expect(r.segments.map((s) => s.kind)).toEqual(["reply", "signature"]);
    expect(r.segments[1]!.span).toEqual([47, 55]);
  });

  it("strips mobile signatures and Apple Mail attributions", async () => {
    const r = await parse(
      "Here is another email\n\nSent from my iPhone\n\nOn Apr 3, 2012, at 4:19 PM, bob <bob@example.com> wrote:\n\n> Hi\n",
      { backend: "cpu" },
    );
    expect(r.reply).toBe("Here is another email");
    expect(r.segments.map((s) => s.kind)).toEqual(["reply", "signature", "attribution", "quote"]);
    expect(r.contact).toBeUndefined();
  });

  it("handles Outlook header blocks without quote prefixes", async () => {
    const r = await parse(
      "Outlook with a reply directly above line\n________________________________________\nFrom: CRM Comments [crm-comment@example.com]\nSent: Friday, 23 March 2012 5:08 p.m.\nTo: John S. Greene\nSubject: [contact:106] John Greene\n\nA new comment has been added to the Contact named 'John Greene':\n\nI am replying to a comment.\n",
      { backend: "cpu" },
    );
    expect(r.reply).toBe("Outlook with a reply directly above line");
    expect(r.segments[0]!.kind).toBe("reply");
    expect(r.segments.some((s) => s.kind === "attribution")).toBe(true);
    expect(r.segments[r.segments.length - 1]!.kind).toBe("quote");
  });

  it("keeps inline (bottom-posted) replies", async () => {
    const r = await parse(
      "On Tue, Apr 29, 2014 at 4:22 PM, Example Dev <sugar@example.com>wrote:\n\n> okay. Well, here's some stuff I can write.\n>\n\nI will reply under this one\n\n>\n> okay?\n>\n\nand under this.\n\n--\nHey there, this is my signature\n",
      { backend: "cpu" },
    );
    expect(r.reply).toBe("I will reply under this one\n\nand under this.");
  });

  it("returns empty results for empty input and a plain word", async () => {
    const empty = await parse("", { backend: "cpu" });
    expect(empty.segments).toEqual([]);
    expect(empty.reply).toBe("");
    expect(empty.contact).toBeUndefined();
    const word = await parse("ok", { backend: "cpu" });
    expect(word.reply).toBe("ok");
    expect(word.segments).toEqual([{ kind: "reply", span: [0, 2] }]);
  });

  it("does not mistake 'sent from' in a sentence for a signature", async () => {
    const r = await parse(
      "Here is another email\n\nSent from my desk, is much easier then my mobile phone.\n",
      { backend: "cpu" },
    );
    expect(r.reply).toBe(
      "Here is another email\n\nSent from my desk, is much easier then my mobile phone.",
    );
  });

  it("parses French attributions and signatures", async () => {
    const r = await parse(
      "Bonjour Marie,\n\nMerci pour votre retour rapide.\n\nCordialement,\nPierre Dubois\nChef de Projet | Nova Solutions SAS\nTél. : +33 1 23 45 67 89\n\nLe 5 janv. 2024 à 15:14, Marie Martin <marie@example.fr> a écrit :\n> Bonjour Pierre,\n",
      { backend: "cpu" },
    );
    expect(r.reply).toBe("Bonjour Marie,\n\nMerci pour votre retour rapide.\n\nCordialement,");
    expect(r.contact?.name).toBe("Pierre Dubois");
    expect(r.contact?.phone).toEqual(["+33 1 23 45 67 89"]);
  });

  it("falls back to the CPU when WebGPU is unavailable under auto", async () => {
    const long = `${"Line of text that keeps going.\n".repeat(80)}\nOn Mon, Jan 5, 2024, Bob <bob@example.com> wrote:\n> hi\n`;
    const r = await parse(long);
    expect(r.diagnostics.backend).toBe("cpu");
    expect(r.diagnostics.tokens).toBeGreaterThan(512);
    expect(r.segments[r.segments.length - 1]!.kind).toBe("quote");
  });

  it("throws on a non-string input and parses many", async () => {
    await expect(parse(undefined as unknown as string)).rejects.toThrow("expects a string");
    const many = await parseMany(["ok", "thanks"], { backend: "cpu" });
    expect(many.map((m) => m.reply)).toEqual(["ok", "thanks"]);
  });
});
