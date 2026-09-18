"""Export the promoted checkpoint to ../model/{manifest.json,weights.txt,fixtures.json}.

``uv run python -m gpu_cite.export [--run default]``

Fixture logits are computed with the *exported* int6 weights (decoded back to float32),
so ``test/parity.test.ts`` compares the TypeScript CPU path against exactly the numbers
the runtime will load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gpu_utils_training.features import tokenize
from gpu_utils_training.quant import export, quantize

from .data import CACHE, read_jsonl
from .features import TABLE_ROWS, WIDTH, featurize_tokens
from .labels import NAMEPARTS, ROLES, TAGS, TYPES
from .model import CiteTagger

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
RUNS = Path(__file__).resolve().parents[1] / "runs"

EXTRA_FIXTURES = [
    "Smith, J., & Doe, A. B. (2019). A study of things. Journal of Stuff, 12(3), 45–67. https://doi.org/10.1000/xyz123",
    "[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. CVPR, 2016, pp. 770–778.",
    "Vaswani A, et al. Attention is all you need. arXiv preprint arXiv:1706.03762v5, 2017.",
    "van der Berg, P. (n.d.) Über Café. Retrieved March 3, 2021, from www.example.org/x?y=1.",
    "Knuth, D. E. (1997) The Art of Computer Programming, Vol. 1. 3rd edn. Reading, MA: Addison-Wesley.",
    "Müller, Hans (2005): Die Stadt als Text. In: Schmidt, Anna (Hrsg.): Urbane Räume. Bielefeld: transcript, S. 33–58.",
    "x",
    "😀 emoji\ttab  double  space",
]


def load_model(run: str) -> tuple[CiteTagger, dict[str, Any]]:
    ckpt = torch.load(RUNS / run / "best.pt", map_location="cpu", weights_only=False)
    model = CiteTagger(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--fixtures", type=int, default=24)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, ckpt = load_model(args.run)
    metrics_path = RUNS / args.run / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    tensors = {k: v.detach().numpy().astype(np.float32) for k, v in model.export_tensors().items()}
    tensors["emb"][0] = 0.0
    manifest = export(
        tensors,
        MODEL_DIR,
        {
            "name": "gpu-cite",
            "embed": model.embed_dim,
            "hidden": model.hidden,
            "head": model.head_dim,
            "tableRows": TABLE_ROWS,
            "featureWidth": WIDTH,
            "labels": TAGS,
            "roles": ROLES,
            "types": TYPES,
            "nameparts": NAMEPARTS,
            "checkpoint": {
                "run": args.run,
                "epoch": ckpt.get("epoch"),
                "seed": ckpt.get("seed"),
                "heldout_exact_match": metrics.get("exact_match"),
                "heldout_micro_f1": metrics.get("micro_f1"),
            },
        },
    )
    print(f"exported {manifest['parameters']} parameters to {MODEL_DIR}")

    # Reload the quantized values so fixtures match what the runtime decodes.
    with torch.no_grad():
        for name, param in model.export_tensors().items():
            q, scale = quantize(tensors[name])
            param.copy_(torch.from_numpy(q.astype(np.float32) * np.float32(scale)))
    model.quant = False

    texts = list(EXTRA_FIXTURES)
    held = read_jsonl(CACHE / "heldout.jsonl.gz")
    for ex in held[: max(0, args.fixtures - len(texts))]:
        texts.append(ex["text"])
    fixtures = []
    with torch.no_grad():
        for text in texts:
            toks = tokenize(text)
            rows = featurize_tokens(text, toks)
            r = torch.as_tensor(rows, dtype=torch.long).unsqueeze(0)
            m = torch.ones(1, len(rows))
            tags, parts, ty = model(r, m)
            fixtures.append(
                {
                    "text": text,
                    "rows": rows,
                    "tags": [round(float(v), 6) for v in tags[0].reshape(-1)],
                    "parts": [round(float(v), 6) for v in parts[0].reshape(-1)],
                    "type": [round(float(v), 6) for v in ty[0]],
                }
            )
    (MODEL_DIR / "fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False) + "\n")
    print(f"wrote {len(fixtures)} fixtures")


if __name__ == "__main__":
    main()
