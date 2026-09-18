"""Closed function-word lexicon shared byte-for-byte with src/features.ts.

The model never sees field names. The only word identities it is allowed to learn
exactly are these task words (operators, sort/group/aggregate/chart/limit words,
time words, carrier prose). Everything else reaches the model as a hashed bucket
plus its schema-membership features.
"""

from __future__ import annotations

# Ordered; the index is the feature id. Changing order changes the model.
KEYWORDS: list[str] = [
    # carrier prose
    "show", "me", "list", "find", "all", "get", "give", "display", "which", "what", "the", "a", "an",
    "of", "for", "to", "in", "on", "at", "from", "that", "are", "is", "was", "were", "have", "has",
    "having", "had", "with", "without", "only", "please", "by", "per", "as", "and", "or", "not",
    "no", "but", "where", "whose", "who", "it", "its", "their", "my", "our", "any", "every", "each",
    "be", "being", "been", "do", "does", "did", "i", "want", "need", "see", "view", "pull", "fetch",
    "filter", "search", "query", "look", "up", "looking", "results", "rows", "records", "entries",
    "items", "values", "value", "data",
    # comparison / operator words
    "equals", "equal", "than", "more", "greater", "over", "above", "exceeding", "exceeds", "bigger",
    "higher", "larger", "less", "fewer", "under", "below", "smaller", "lower", "least", "most",
    "minimum", "maximum", "min", "max", "between", "containing", "contains", "contain", "include",
    "includes", "including", "mentioning", "mentions", "matching", "matches", "like", "named",
    "called", "titled", "one", "either", "set", "marked", "empty", "blank", "null", "missing",
    "unset", "none", "filled", "present", "lacking", "assigned", "exactly", "before", "after",
    "since", "until", "till", "through", "prior", "earlier", "later", "newer", "older", "past",
    "within", "during", "ago", "true", "false", "yes", "off", "enabled", "disabled", "except",
    "excluding", "exclude", "other", "isn", "aren", "don", "doesn", "never", "non", "unknown",
    "starting", "ending", "starts", "ends", "beginning", "isnt", "arent", "dont", "doesnt", "to",
    # sort words
    "sort", "sorted", "order", "ordered", "arrange", "arranged", "rank", "ranked", "ascending",
    "asc", "descending", "desc", "high", "low", "newest", "oldest", "latest", "earliest", "recent",
    "first", "last", "top", "bottom", "highest", "lowest", "z", "reverse", "alphabetical",
    "alphabetically", "largest", "smallest", "biggest",
    # group words
    "group", "grouped", "grouping", "broken", "down", "split", "segmented", "segment", "across",
    "breakdown", "bucketed", "bucket", "aggregated", "aggregate", "rolled", "roll",
    # aggregate words
    "total", "totals", "sum", "summed", "average", "avg", "mean", "count", "number", "how", "many",
    "median",
    # limit words
    "limit", "limited", "just", "cap", "capped", "maximum", "show", "at",
    # chart words
    "chart", "graph", "bar", "bars", "column", "columns", "histogram", "line", "lines", "trend",
    "pie", "donut", "doughnut", "table", "tabular", "grid", "kpi", "scorecard", "stat", "metric",
    "tile", "plot", "plotted", "visualize", "visualise", "visualization", "render", "rendered",
    "big", "single", "figure", "over", "time", "timeseries",
    # time words
    "today", "yesterday", "tomorrow", "this", "next", "previous", "current", "day", "days", "week",
    "weeks", "month", "months", "quarter", "quarters", "year", "years", "ytd", "q", "h", "half",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec", "hour", "hours", "now", "date", "dates",
    # numeric suffixes
    "k", "m", "b", "bn", "mm", "thousand", "million", "billion", "percent", "pct", "usd", "eur",
    # symbols
    ">", "<", "=", ":", ",", "-", "!", "%", "$", "€", "£", "\"", "'", "(", ")", ".", "?", "/",
    "#", "+", "&", "*", ";", "~", "_", "@", "|", "[", "]", "≥", "≤",
]
# Deduplicate while keeping first occurrence; a few words appear in two groups above.
_seen: set[str] = set()
KEYWORDS = [w for w in KEYWORDS if not (w in _seen or _seen.add(w))]
KEYWORD_ID: dict[str, int] = {w: i for i, w in enumerate(KEYWORDS)}
KEYWORD_COUNT = len(KEYWORDS)

MONTHS = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
]
MONTH_ABBREV = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
