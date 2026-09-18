import { describe, expect, it } from "vitest";
import { featurize, parse, parseMany } from "../src/index.ts";

const cpu = { backend: "cpu" as const };

describe("gpu-cite", () => {
	it("emits one feature row per token", () => {
		const f = featurize("hello world");
		expect(f.rows).toHaveLength(f.tokens.length);
	});

	it("parses an APA journal article", async () => {
		const r = await parse(
			"Hinton, G., Osindero, S., & Teh, Y.-W. (2006). A fast learning algorithm for deep belief nets. Neural Computation, 18(7), 1527–1554. https://doi.org/10.1162/neco.2006.18.7.1527",
			cpu,
		);
		expect(r.type).toBe("article");
		expect(r.authors.map((a) => a.family)).toEqual([
			"Hinton",
			"Osindero",
			"Teh",
		]);
		expect(r.authors[0]!.given).toBe("G.");
		expect(r.title).toBe("A fast learning algorithm for deep belief nets");
		expect(r.container).toBe("Neural Computation");
		expect(r.year).toBe(2006);
		expect(r.volume).toBe("18");
		expect(r.issue).toBe("7");
		expect(r.pages).toEqual({ from: "1527", to: "1554" });
		expect(r.doi).toBe("10.1162/neco.2006.18.7.1527");
		expect(r.url).toBeUndefined();
		expect(r.spans.title).toEqual([47, 93]);
	});

	it("parses an IEEE conference paper with et al.", async () => {
		const r = await parse(
			'[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Las Vegas, NV, USA, 2016, pp. 770–778.',
			cpu,
		);
		expect(r.type).toBe("conference");
		expect(r.authors).toHaveLength(4);
		expect(r.authors[0]).toMatchObject({ given: "K.", family: "He" });
		expect(r.title).toBe("Deep residual learning for image recognition");
		expect(r.year).toBe(2016);
		expect(r.pages).toEqual({ from: "770", to: "778" });
		expect(r.diagnostics.etAl).toBe(false);
	});

	it("extracts identifiers deterministically", async () => {
		const r = await parse(
			"Vaswani A, et al. Attention is all you need. arXiv preprint arXiv:1706.03762v5, 2017.",
			cpu,
		);
		expect(r.arxiv).toBe("1706.03762v5");
		expect(r.type).toBe("preprint");
		expect(r.diagnostics.etAl).toBe(true);
		expect(r.year).toBe(2017);
		const w = await parse(
			"NHS. Symptoms of flu. 2022. Available at: https://www.nhs.uk/conditions/flu/ [Accessed 14 February 2023].",
			cpu,
		);
		expect(w.url).toBe("https://www.nhs.uk/conditions/flu/");
		expect(w.accessed).toBe("14 February 2023");
		expect(w.type).toBe("web");
	});

	it("parses a book with edition, location and publisher", async () => {
		const r = await parse(
			"Knuth, D. E. (1997). The Art of Computer Programming, Vol. 1: Fundamental Algorithms (3rd ed.). Reading, MA: Addison-Wesley.",
			cpu,
		);
		expect(r.type).toBe("book");
		expect(r.edition).toBe("3rd");
		expect(r.publisher).toBe("Addison-Wesley");
		expect(r.location).toBe("Reading, MA");
	});

	it("handles empty and degenerate input without throwing", async () => {
		const empty = await parse("", cpu);
		expect(empty.type).toBe("unknown");
		expect(empty.authors).toEqual([]);
		expect(empty.diagnostics.warnings).toContain("empty input");
		const dots = await parse("...", cpu);
		expect(dots.diagnostics.tags).toHaveLength(3);
	});

	it("parseMany keeps absolute offsets per line", async () => {
		const text =
			"Smith, J. (2019). First paper. Journal A, 1(2), 3–4.\n\n  \nDoe, A. (2020). Second paper. Journal B, 5(6), 7–8. https://doi.org/10.1000/abc";
		const rs = await parseMany(text, cpu);
		expect(rs).toHaveLength(2);
		expect(rs[0]!.range).toEqual([0, text.indexOf("\n")]);
		expect(rs[1]!.doi).toBe("10.1000/abc");
		const [s, e] = rs[1]!.spans.doi!;
		expect(text.slice(s, e)).toBe("10.1000/abc");
		const [ts, te] = rs[1]!.spans.title!;
		expect(text.slice(ts, te)).toBe("Second paper");
		expect(rs[1]!.diagnostics.tokens[0]!.start).toBe(rs[1]!.range[0]);
	});

	it("treats newlines inside one reference as spaces", async () => {
		const r = await parse(
			"Smith, J. (2019). A wrapped\ntitle here. Journal A, 1(2), 3–4.",
			cpu,
		);
		expect(r.title).toBe("A wrapped title here");
	});
});
