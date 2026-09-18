"""Export runs/latest.pt to ../model/{manifest.json,weights.txt,fixtures.json}.

    uv run python -m gpu_email.export

fixtures.json holds >= 20 cases with the input text, the feature rows and the
PyTorch logits (int6-quantized weights) so test/parity.test.ts can assert that the
TypeScript reference forward pass reproduces them at 1e-4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.quant import export as quant_export

from gpu_email.data import EmailGen, render
from gpu_email.features import BIO_LABELS, LINE_KINDS, NUM_ROWS, NUM_SLOTS, featurize
from gpu_email.model import EmailTagger
from gpu_email.train import RUNS

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
TEST_FIXTURES = Path(__file__).resolve().parents[2] / "test" / "fixtures"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def row_hash(rows: list[list[int]]) -> int:
    """FNV-1a over the flat row ids; mirrors test/features.test.ts:rowHash."""
    h = 2166136261
    for row in rows:
        for v in row:
            h = ((h ^ v) * 16777619) & 0xFFFFFFFF
    return h


def write_feature_hashes() -> int:
    """Row hashes of the unfamiliar set + fixture texts for the TS featurizer parity test."""
    cases = json.loads((DATA_DIR / "unfamiliar.json").read_text())
    out = []
    for c in cases:
        nl = c.get("newline", "\n")
        text = nl.join(t for _, t in c["lines"]) + ("" if c.get("trailing_newline", True) is False else nl)
        rows = featurize(text)
        out.append({"name": c["name"], "text": text, "tokens": len(rows), "hash": row_hash(rows)})
    for i, text in enumerate(FIXTURE_TEXTS):
        rows = featurize(text)
        out.append({"name": f"fixture-{i}", "text": text, "tokens": len(rows), "hash": row_hash(rows)})
    TEST_FIXTURES.mkdir(parents=True, exist_ok=True)
    (TEST_FIXTURES / "features-hashes.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
    return len(out)

FIXTURE_TEXTS = [
    "Hi Bob,\n\nThanks for the update, that works for me.\n\nBest,\nJohn Doe\nCEO, Acme Inc\n+1 (555) 123-4567\njohn@acme.com\nhttps://www.acme.com\n\nOn Mon, Jan 5, 2024 at 3:14 PM Bob Smith <bob@example.com> wrote:\n> Hi John,\n>\n> Can we move the meeting?\n>\n> Bob\n",
    "Here is another email\n\nSent from my iPhone\n\nOn Apr 3, 2012, at 4:19 PM, bob <bob@example.com> wrote:\n\n> Hi\n",
    "Outlook with a reply\n\n\n ------------------------------\n\n*From:* Google Apps Sync Team [mailto:mail-noreply@google.com]\n*Sent:* Thursday, February 09, 2012 1:36 PM\n*To:* jow@xxxx.com\n*Subject:* Google Apps Sync was updated!\n\n\n\nDear Google Apps Sync user,\n\nGoogle Apps Sync for Microsoft Outlook was recently updated.\n\nSincerely,\n\nThe Google Apps Sync Team.\n",
    "Bonjour Marie,\n\nMerci pour votre retour rapide. Je reviens vers vous lundi.\n\nCordialement,\nPierre Dubois\nChef de Projet | Nova Solutions SAS\nTél. : +33 1 23 45 67 89\n\nLe 5 janv. 2024 à 15:14, Marie Martin <marie@example.fr> a écrit :\n> Bonjour Pierre,\n> Pouvez-vous m'envoyer le rapport ?\n",
    "Hallo Herr Müller,\n\ndas klingt gut. Anbei die aktuelle Version.\n\nMit freundlichen Grüßen\nAnna Schmidt\nGeschäftsführerin\nBergmann GmbH\nHauptstraße 12\n10115 Berlin\nTelefon: +49 30 1234 5678\n\n-----Ursprüngliche Nachricht-----\nVon: Hans Müller <hans@example.de>\nGesendet: Montag, 5. Februar 2024 09:12\nAn: Anna Schmidt\nBetreff: AW: Angebot\n\nHallo Frau Schmidt,\nkönnen Sie mir das Angebot schicken?\n",
    "Hola Carlos,\n\nMe parece bien. Nos vemos el viernes.\n\nSaludos,\nMaría García\nGerente de Ventas, Atlas Logistics S.L.\nMóvil: +34 612 345 678\n\nEl 5 ene 2024, a las 10:02, Carlos López <carlos@example.es> escribió:\n> ¿Te viene bien a las 10?\n",
    "this is an email with a correct -- signature.\n\n-- \nrick\n",
    "Thanks!\n\n---------- Forwarded message ---------\nFrom: Jane Roe <jane@example.org>\nDate: Tue, Mar 5, 2024 at 9:00 AM\nSubject: Invoice #4521\nTo: John Doe <john@acme.com>\n\nHi John,\n\nPlease find the invoice attached.\n\nJane\n",
    "I will reply under this one\n\n>\n> okay?\n>\n\nand under this.\n\n>\n> -- Tim\n>\n\n--\nHey there, this is my signature\n",
    "Awesome! I haven't had another problem with it.\r\n\r\nOn Aug 22, 2011, at 7:37 PM, defunkt<reply@reply.github.com> wrote:\r\n\r\n> Loader seems to be working well.\r\n",
    "お世話になっております。\n\n添付ファイルをご確認ください。\n\nよろしくお願いいたします。\n佐藤 健\n営業部長\n株式会社ノヴァ\n電話: 03-1234-5678\n\n2024年1月5日(金) 15:14 鈴木 花子 <suzuki@example.jp>:\n> ご確認のほどよろしくお願いいたします。\n",
    "您好，\n\n请查收附件。\n\n谢谢！\n王伟\n销售经理\n北京市朝阳区建国路88号\n手机：+86 138 0013 8000\n\n在 2024年1月5日，李娜 <lina@example.cn> 写道：\n> 没问题。\n",
    "Quick question about the roadmap.\n\nCheers,\nSam\n\nThis email and any attachments are confidential and intended solely for the use of the individual to whom they are addressed. If you have received this email in error please notify the sender.\n",
    "Hi folks\n\nWhat is the best way to clear a Riak bucket of all key, values after\nrunning a test?\nI am currently using the Java HTTP API.\n\n-Abhishek Kona\n\n\n_______________________________________________\nriak-users mailing list\nriak-users@lists.basho.com\nhttp://lists.basho.com/mailman/listinfo/riak-users_lists.basho.com\n",
    "Sounds good.\n\nOn Tue, Sep 25, 2012 at 8:59 AM, Chris Wanstrath\n<notifications@github.com>wrote:\n\n> Steps 0-2 are in prod.\n>\n",
    "Hi there!\n\nStuff happened.\n\nAnd here is a fix -- this is not a signature.\n\nkthxbai\n",
    "Thanks,\nJoe Smith | Director, Product Management | Initech\nM: 415-555-0199 | E: joe@initech.com\n",
    "",
    "Hello",
    "> quoted only\n> nothing new here\n",
    "Dear Dr. Patel,\n\nI hope this finds you well. Could you send over the latest version of the deck?\n\nKind regards,\nPriya Sharma\nSenior Product Manager\nMeridian Health Pvt. Ltd.\n+91 98765 43210 | priya.sharma@meridianhealth.in\nwww.meridianhealth.in\n\nSent from my Samsung Galaxy\n",
    "Outlook with a reply directly above line\n________________________________________\nFrom: CRM Comments [crm-comment@example.com]\nSent: Friday, 23 March 2012 5:08 p.m.\nTo: John S. Greene\nSubject: [contact:106] John Greene\n\nA new comment has been added to the Contact named 'John Greene':\n\nI am replying to a comment.\n",
]


def load_model() -> tuple[EmailTagger, dict]:
    ckpt = torch.load(RUNS / "latest.pt", map_location="cpu", weights_only=False)
    model = EmailTagger(ckpt["dim"], ckpt["hidden"], ckpt.get("dilations"))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def quantized_tensors(model: EmailTagger) -> dict[str, np.ndarray]:
    return {k: v.numpy().astype(np.float32) for k, v in model.export_tensors().items()}


def dequantized_model(model: EmailTagger) -> EmailTagger:
    """A copy whose weights are exactly the int6 values the runtime will decode."""
    from gpu_utils_training.quant import fake_quant

    copy = EmailTagger(model.dim, model.hidden, model.dilations)
    copy.load_state_dict(model.state_dict())
    with torch.no_grad():
        for p in copy.parameters():
            p.copy_(torch.from_numpy(fake_quant(p.numpy())))
    copy.eval()
    copy.qat = False
    return copy


def logits_for(model: EmailTagger, text: str) -> tuple[list[list[int]], np.ndarray]:
    rows = featurize(text)
    if not rows:
        return rows, np.zeros((0, len(LINE_KINDS) + len(BIO_LABELS)), dtype=np.float32)
    ids = torch.tensor(rows, dtype=torch.long).unsqueeze(0)
    mask = torch.ones(1, len(rows))
    with torch.no_grad():
        kind, bio = model(ids, mask)
    return rows, torch.cat([kind, bio], dim=-1)[0].numpy()


def main() -> None:
    model, ckpt = load_model()
    tensors = quantized_tensors(model)
    manifest = quant_export(
        tensors,
        MODEL_DIR,
        {
            "name": "gpu-email",
            "labels": LINE_KINDS,
            "fields": BIO_LABELS,
            "hidden": model.dim,
            "head": model.hidden,
            "dilations": model.dilations,
            "slots": NUM_SLOTS,
            "rows": NUM_ROWS,
            "checkpoint": {"seed": ckpt.get("seed"), "steps": ckpt.get("steps"), "epochs": ckpt.get("epochs"), "metrics": ckpt.get("metrics")},
        },
    )
    q = dequantized_model(model)
    gen = EmailGen(4242)
    texts = list(FIXTURE_TEXTS)
    while len(texts) < 28:
        lines, _ = gen.email()
        texts.append(render(lines, gen).text)
    fixtures = []
    for text in texts:
        rows, logits = logits_for(q, text)
        fixtures.append({"text": text, "features": rows, "logits": [[round(float(v), 5) for v in r] for r in logits]})
    (MODEL_DIR / "fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False))
    size = (MODEL_DIR / "weights.txt").stat().st_size
    n_hashes = write_feature_hashes()
    print(f"exported {manifest['parameters']} params, weights.txt {size} B, {len(fixtures)} fixtures, {n_hashes} feature hashes", file=sys.stderr)


if __name__ == "__main__":
    main()
