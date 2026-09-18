"""Role labels (BIO for spans; SEP and NEG are single-token roles)."""

LABELS = ["O", "B-PROP", "I-PROP", "B-VAL", "I-VAL", "B-VAR", "I-VAR", "SEP", "NEG"]
LABEL_ID = {label: i for i, label in enumerate(LABELS)}
BOUNDARY = len(LABELS)  # index of the boundary logit column in the tag head
