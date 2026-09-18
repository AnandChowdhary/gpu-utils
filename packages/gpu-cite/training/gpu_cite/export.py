"""Export the promoted checkpoint to ../model/{manifest.json,weights.txt,fixtures.json}.

    uv run python -m gpu_cite.export [--run default] [--fixtures 24]

``fixtures.json`` uses the canonical format ``{"cases": [{input, rows, logits, pooled}]}``
where ``logits`` has 31 BIO + 3 name-part columns and ``pooled`` the 8 type logits; both
are computed from the decoded int6 weights, so ``test/parity.test.ts`` and the WGSL
harness compare against exactly what the runtime loads. The manifest additionally carries
``roles``, ``types`` and ``nameparts`` for the decoder and the ``trans`` tensor.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from gpu_utils_training.export import export_package
from gpu_utils_training.features import tokenize
from gpu_utils_training.loop import load_checkpoint

from .data import CACHE, read_jsonl
from .features import WIDTH, featurize_tokens
from .labels import NAMEPARTS, ROLES, TAGS, TYPES
from .model import build

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
RUNS = Path(__file__).resolve().parents[1] / "runs"

HAND_WRITTEN = [
    "Smith, J., & Doe, A. B. (2019). A study of things. Journal of Stuff, 12(3), 45–67. https://doi.org/10.1000/xyz123",
    "[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. CVPR, 2016, pp. 770–778.",
    "Vaswani A, et al. Attention is all you need. arXiv preprint arXiv:1706.03762v5, 2017.",
    "van der Berg, P. (n.d.) Über Café. Retrieved March 3, 2021, from www.example.org/x?y=1.",
    "Knuth, D. E. (1997) The Art of Computer Programming, Vol. 1. 3rd edn. Reading, MA: Addison-Wesley.",
    "Müller, Hans (2005): Die Stadt als Text. In: Schmidt, Anna (Hrsg.): Urbane Räume. Bielefeld: transcript, S. 33–58.",
    "x",
    "😀 emoji\ttab  double  space",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--fixtures", type=int, default=24)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = build()
    ckpt = load_checkpoint(model, RUNS / args.run / "best.pt")
    history_path = RUNS / args.run / "history.json"
    best: dict[str, Any] = json.loads(history_path.read_text()).get("best") or {} if history_path.exists() else {}
    checkpoint = {
        "run": args.run,
        "epoch": ckpt.get("epoch"),
        "seed": ckpt.get("seed"),
        "date": datetime.now(tz=UTC).date().isoformat(),
        "heldout_exact_match": best.get("exact_match"),
        "heldout_micro_f1": best.get("micro_f1"),
    }
    texts = list(HAND_WRITTEN)
    for ex in read_jsonl(CACHE / "heldout.jsonl.gz")[: max(0, args.fixtures - len(texts))]:
        texts.append(ex["text"])
    fixtures = [{"input": text, "rows": featurize_tokens(text, tokenize(text))} for text in texts]
    manifest = export_package(
        model,
        MODEL_DIR,
        TAGS,
        {"name": "gpu-cite", "roles": ROLES, "types": TYPES, "nameparts": NAMEPARTS, "checkpoint": checkpoint},
        fixtures,
        slots=WIDTH,
    )
    print(f"exported {manifest['parameters']:,} parameters and {len(fixtures)} fixtures to {MODEL_DIR}")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("tensors", "labels")}, indent=1))


if __name__ == "__main__":
    main()
