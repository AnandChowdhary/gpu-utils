/** BIO helpers. Mirrors tooling/python/gpu_utils_training/decode.py and metrics.py. */

export const BIO_FORBID = -1e9;

/**
 * `[k, k]` transition table (`[from * k + to]`) forbidding `O → I-X`, `B-X → I-Y` and
 * `I-X → I-Y` (X ≠ Y). Labels are "O", "B-X" or "I-X"; other labels are unconstrained.
 */
export function bioTransitions(labels: string[], forbid = BIO_FORBID): Float32Array {
  const k = labels.length;
  const t = new Float32Array(k * k);
  for (let to = 0; to < k; to++) {
    const lt = labels[to]!;
    if (!lt.startsWith("I-")) continue;
    for (let from = 0; from < k; from++) {
      const lf = labels[from]!;
      const ok = (lf.startsWith("B-") || lf.startsWith("I-")) && lf.slice(2) === lt.slice(2);
      if (!ok) t[from * k + to] = forbid;
    }
  }
  return t;
}

/** Additive `[k]` penalty for the first token: a sequence may not start with `I-X`. */
export function bioStartMask(labels: string[], forbid = BIO_FORBID): Float32Array {
  return Float32Array.from(labels, (l) => (l.startsWith("I-") ? forbid : 0));
}

export interface BioSpan {
  label: string;
  /** Token index range, half-open. */
  startToken: number;
  endToken: number;
  /** UTF-16 offsets when `tokens` were given, half-open. */
  start: number;
  end: number;
}

/**
 * Decodes tag ids into spans. A stray `I-X` (after `O` or a different label) starts a new
 * span, the lenient conlleval convention also used by metrics.bio_to_spans.
 */
export function bioToSpans(
  tagIds: ArrayLike<number>,
  labels: string[],
  tokens?: ArrayLike<{ start: number; end: number }>,
): BioSpan[] {
  const spans: BioSpan[] = [];
  let current: { label: string; startToken: number } | undefined;
  const close = (endToken: number) => {
    if (!current) return;
    const first = tokens?.[current.startToken];
    const last = tokens?.[endToken - 1];
    spans.push({
      label: current.label,
      startToken: current.startToken,
      endToken,
      start: first?.start ?? current.startToken,
      end: last?.end ?? endToken,
    });
    current = undefined;
  };
  for (let i = 0; i < tagIds.length; i++) {
    const tag = labels[tagIds[i]!] ?? "O";
    if (tag.startsWith("B-") || (tag.startsWith("I-") && current?.label !== tag.slice(2))) {
      close(i);
      current = { label: tag.slice(2), startToken: i };
    } else if (!tag.startsWith("I-")) {
      close(i);
    }
  }
  close(tagIds.length);
  return spans;
}
