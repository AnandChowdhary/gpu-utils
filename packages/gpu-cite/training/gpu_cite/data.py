"""Synthetic reference renderer: structured metadata → labelled reference strings.

Every rendered string is assembled from *pieces* ``(text, role, entity, namepart)`` so the
character span of every field, every individual author/editor and every name part is known
exactly. Tokens are then labelled BIO over roles (``labels.TAGS``), name parts over
``labels.NAMEPARTS`` and the whole string gets a document type.

Nineteen style families (APA, MLA, Chicago author-date and notes, IEEE, Vancouver, Harvard,
Nature, ACM, AMA, Elsevier numbered, Springer, plain BibTeX-ish, arXiv listing, Wikipedia
cite, CSE, terse physics, German, and a deliberately messy one) each carry internal random
variation: initials vs full names, ``et al.`` cut-offs, ``and``/``&``/``und``, quote styles,
volume(issue) forms, page-range dashes, numbering prefixes, dropped punctuation, typos.

``uv run python -m gpu_cite.data`` writes ``data/cache/{train,heldout}.jsonl``.
"""

from __future__ import annotations

import gzip
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpu_utils_training.features import tokenize

from .labels import NAMEPARTS, ROLES, TAG_INDEX, TYPE_INDEX, TYPES
from .sources import CACHE, load_metadata

Rng = random.Random

# ----------------------------------------------------------------------------- pools

CITIES = [
    "New York", "London", "Berlin", "Cambridge", "Oxford", "Paris", "Boston", "Chicago", "Heidelberg",
    "Amsterdam", "Tokyo", "Beijing", "Washington, DC", "San Francisco", "Los Angeles", "Philadelphia",
    "Dordrecht", "Hoboken, NJ", "Cham", "Singapore", "Sydney", "Toronto", "New Delhi", "Vienna", "Zurich",
    "Munich", "Milan", "Rome", "Madrid", "Barcelona", "Stockholm", "Copenhagen", "Edinburgh", "Dublin",
    "New York, NY", "Cambridge, MA", "Thousand Oaks, CA", "Upper Saddle River, NJ", "Reading, MA",
    "Princeton, NJ", "Ithaca, NY", "Basel", "Seoul", "Melbourne", "Montreal", "Vancouver", "Lisbon",
    "Prague", "Warsaw", "Helsinki", "Oslo", "Athens", "Istanbul", "Cairo", "Nairobi", "São Paulo",
    "Buenos Aires", "Mexico City", "Shanghai", "Hong Kong", "Taipei", "Bangalore", "Tehran", "Riyadh",
    "Cambridge, UK", "London, UK", "Berlin, Germany", "Paris, France", "Tokyo, Japan", "Boston, MA, USA",
]
PUBLISHERS = [
    "Springer", "Springer-Verlag", "Springer Nature", "Elsevier", "Wiley", "John Wiley & Sons", "Routledge",
    "SAGE", "SAGE Publications", "Oxford University Press", "Cambridge University Press", "MIT Press",
    "Pearson", "McGraw-Hill", "Penguin", "Random House", "Academic Press", "Kluwer Academic", "Plenum Press",
    "Prentice Hall", "Addison-Wesley", "Blackwell", "Macmillan", "Palgrave Macmillan", "Harvard University Press",
    "Princeton University Press", "Yale University Press", "University of Chicago Press", "Stanford University Press",
    "O'Reilly Media", "SIAM", "IEEE Press", "ACM Press", "CRC Press", "Taylor & Francis", "De Gruyter",
    "Mouton de Gruyter", "Brill", "Peter Lang", "Hanser", "Vieweg", "Teubner", "Duncker & Humblot",
    "Nova Science Publishers", "World Scientific", "Morgan Kaufmann", "No Starch Press", "Manning", "Apress",
    "Basic Books", "W. W. Norton", "Verso", "Polity", "Bloomsbury", "HarperCollins", "Simon & Schuster",
    "Cornell University Press", "Duke University Press", "Johns Hopkins University Press", "Columbia University Press",
]
UNIVERSITIES = [
    "Massachusetts Institute of Technology", "Stanford University", "University of California, Berkeley",
    "University of Oxford", "University of Cambridge", "ETH Zürich", "Carnegie Mellon University",
    "University of Toronto", "Technische Universität München", "University of Edinburgh", "Harvard University",
    "Princeton University", "University of Michigan", "Georgia Institute of Technology", "University of Washington",
    "Imperial College London", "University College London", "National University of Singapore", "Tsinghua University",
    "University of Tokyo", "Seoul National University", "Indian Institute of Technology Delhi",
    "Universidad de Buenos Aires", "Universität Heidelberg", "Université Paris-Saclay", "KU Leuven",
    "Delft University of Technology", "University of Melbourne", "McGill University", "Caltech", "MIT",
    "UC Berkeley", "Univ. of Illinois at Urbana-Champaign", "New York University", "Columbia University",
    "Cornell University", "Yale University", "Duke University", "Purdue University", "Ohio State University",
    "University of Texas at Austin", "University of Wisconsin–Madison", "Pennsylvania State University",
    "Institut Polytechnique de Paris", "Politecnico di Milano", "Sapienza Università di Roma",
]
DEPARTMENTS = [
    "Department of Computer Science", "Dept. of Electrical Engineering", "School of Medicine",
    "Department of Physics", "Faculty of Arts", "Department of Psychology", "School of Public Health",
    "Department of Economics", "Institute for Advanced Study", "Department of Mathematics",
]
INSTITUTIONS = [
    "National Bureau of Economic Research", "RAND Corporation", "World Health Organization", "OECD",
    "World Bank", "International Monetary Fund", "European Commission", "National Institutes of Health",
    "Centers for Disease Control and Prevention", "NASA", "Los Alamos National Laboratory",
    "Lawrence Berkeley National Laboratory", "Brookings Institution", "Pew Research Center",
    "Federal Reserve Bank of St. Louis", "Bank of England", "European Central Bank", "IEEE", "IETF",
    "W3C", "Internet Engineering Task Force", "Microsoft Research", "Google", "Google DeepMind",
    "OpenAI", "Meta AI", "IBM Research", "Bell Labs", "Xerox PARC", "SRI International", "Max-Planck-Institut",
    "Fraunhofer-Gesellschaft", "CNRS", "INRIA", "CERN", "United Nations", "UNESCO", "UNICEF",
]
SITES = [
    ("The New York Times", "nytimes.com"), ("BBC News", "bbc.co.uk"), ("The Guardian", "theguardian.com"),
    ("Wikipedia", "en.wikipedia.org"), ("GitHub", "github.com"), ("Medium", "medium.com"),
    ("TechCrunch", "techcrunch.com"), ("Nature News", "nature.com"), ("Reuters", "reuters.com"),
    ("Wired", "wired.com"), ("The Verge", "theverge.com"), ("Ars Technica", "arstechnica.com"),
    ("Stack Overflow", "stackoverflow.com"), ("MDN Web Docs", "developer.mozilla.org"),
    ("Python Documentation", "docs.python.org"), ("Khan Academy", "khanacademy.org"),
    ("Mayo Clinic", "mayoclinic.org"), ("WebMD", "webmd.com"), ("Harvard Business Review", "hbr.org"),
    ("Forbes", "forbes.com"), ("Bloomberg", "bloomberg.com"), ("The Economist", "economist.com"),
    ("Scientific American", "scientificamerican.com"), ("Quanta Magazine", "quantamagazine.org"),
    ("Smithsonian Magazine", "smithsonianmag.com"), ("NPR", "npr.org"), ("CNN", "cnn.com"),
    ("Al Jazeera", "aljazeera.com"), ("Der Spiegel", "spiegel.de"), ("Le Monde", "lemonde.fr"),
    ("El País", "elpais.com"), ("The Conversation", "theconversation.com"), ("Substack", "substack.com"),
    ("YouTube", "youtube.com"), ("Vox", "vox.com"), ("Slate", "slate.com"), ("Gov.uk", "gov.uk"),
    ("CDC", "cdc.gov"), ("NASA", "nasa.gov"), ("WHO", "who.int"), ("Our World in Data", "ourworldindata.org"),
]
ORGS = [
    "World Health Organization", "United Nations", "European Commission", "National Research Council",
    "American Psychological Association", "OECD", "World Bank", "Centers for Disease Control and Prevention",
    "U.S. Census Bureau", "Office for National Statistics", "Eurostat", "Statistics Canada", "IPCC",
    "Intergovernmental Panel on Climate Change", "Pew Research Center", "Gallup", "The Lancet Commission",
    "Royal Society", "National Academies of Sciences, Engineering, and Medicine", "International Energy Agency",
    "Amnesty International", "Human Rights Watch", "Greenpeace", "Wikipedia contributors", "Anonymous",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS_ABBR = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]
MONTHS_3 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "15th", "20th", "25th", "30th", "40th"]
ORDINAL_WORDS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
STOP = {"of", "the", "a", "an", "for", "on", "in", "and", "&", "to", "with", "by", "de", "la", "le", "et", "des", "du"}
ABBREV = {
    "journal": "J.", "international": "Int.", "proceedings": "Proc.", "conference": "Conf.", "review": "Rev.",
    "letters": "Lett.", "physics": "Phys.", "physical": "Phys.", "chemistry": "Chem.", "chemical": "Chem.",
    "biology": "Biol.", "biological": "Biol.", "medicine": "Med.", "medical": "Med.", "research": "Res.",
    "transactions": "Trans.", "annals": "Ann.", "american": "Am.", "european": "Eur.", "society": "Soc.",
    "science": "Sci.", "sciences": "Sci.", "engineering": "Eng.", "mathematics": "Math.", "mathematical": "Math.",
    "computer": "Comput.", "computing": "Comput.", "communications": "Commun.", "applied": "Appl.",
    "applications": "Appl.", "management": "Manag.", "psychology": "Psychol.", "psychological": "Psychol.",
    "education": "Educ.", "educational": "Educ.", "economics": "Econ.", "economic": "Econ.", "environmental": "Environ.",
    "environment": "Environ.", "materials": "Mater.", "molecular": "Mol.", "cellular": "Cell.", "clinical": "Clin.",
    "neuroscience": "Neurosci.", "quarterly": "Q.", "bulletin": "Bull.", "advances": "Adv.", "advanced": "Adv.",
    "systems": "Syst.", "system": "Syst.", "technology": "Technol.", "symposium": "Symp.", "workshop": "Wksp.",
    "national": "Natl.", "academy": "Acad.", "united": "U.", "states": "S.", "analysis": "Anal.", "theory": "Theory",
    "studies": "Stud.", "linguistics": "Ling.", "language": "Lang.", "history": "Hist.", "geology": "Geol.",
    "astronomy": "Astron.", "astrophysical": "Astrophys.", "nuclear": "Nucl.", "optics": "Opt.", "optical": "Opt.",
}

# --------------------------------------------------------------------------- builder


@dataclass
class Piece:
    text: str
    role: str | None = None  # one of ROLES or None (O)
    ent: int = -1  # entity index, distinguishes consecutive authors/editors
    part: int = 0  # NAMEPARTS index


@dataclass
class Builder:
    pieces: list[Piece] = field(default_factory=list)
    ent_counter: int = 0

    def o(self, text: str) -> None:
        if text:
            self.pieces.append(Piece(text))

    def f(self, text: str, role: str, part: int = 0, ent: int | None = None) -> int:
        """Add a field piece. Returns the entity index used."""
        if ent is None:
            self.ent_counter += 1
            ent = self.ent_counter
        if text:
            self.pieces.append(Piece(text, role, ent, part))
        return ent

    def text(self) -> str:
        return "".join(p.text for p in self.pieces)


# ---------------------------------------------------------------- name rendering


def _given_parts(given: str) -> list[str]:
    return [p for p in re.split(r"[ \-]+", given.strip()) if p]


def initials(given: str, style: str) -> str:
    parts = _given_parts(given)
    letters = [p[0].upper() for p in parts if p[0].isalpha()]
    if not letters:
        return given
    if style == "dot_space":
        return " ".join(f"{c}." for c in letters)
    if style == "dot":
        return "".join(f"{c}." for c in letters)
    if style == "plain":
        return "".join(letters)
    if style == "single":
        return f"{letters[0]}."
    if style == "space":
        return " ".join(letters)
    return " ".join(f"{c}." for c in letters)


def render_given(given: str, form: str, ini_style: str) -> str:
    if form == "full":
        return given
    if form == "first_full":
        parts = _given_parts(given)
        if len(parts) > 1:
            return parts[0] + " " + initials(" ".join(parts[1:]), ini_style)
        return given
    return initials(given, ini_style)


def add_person(
    B: Builder,
    person: dict[str, str],
    role: str,
    mode: str,
    form: str,
    ini_style: str,
) -> None:
    """Emit one author/editor as pieces of a single entity."""
    if "literal" in person:
        B.f(person["literal"], role, 0)
        return
    fam = person["family"]
    giv = render_given(person["given"], form, ini_style) if person.get("given") else ""
    ent = B.f("", role)  # reserve entity index
    if mode == "caps":
        fam = fam.upper()
        mode = "fam_giv"
    if not giv:
        B.f(fam, role, 2, ent)
        return
    if mode == "fam_giv":
        B.f(fam, role, 2, ent)
        B.f(", ", role, 0, ent)
        B.f(giv, role, 1, ent)
    elif mode == "giv_fam":
        B.f(giv, role, 1, ent)
        B.f(" ", role, 0, ent)
        B.f(fam, role, 2, ent)
    elif mode == "van":  # Vancouver: Smith JA
        B.f(fam, role, 2, ent)
        B.f(" ", role, 0, ent)
        B.f(initials(person["given"], "plain"), role, 1, ent)
    elif mode == "fam_giv_nocomma":  # Smith J. A.
        B.f(fam, role, 2, ent)
        B.f(" ", role, 0, ent)
        B.f(giv, role, 1, ent)
    else:
        raise ValueError(mode)


@dataclass
class NameStyle:
    mode: str = "fam_giv"  # first author
    rest_mode: str | None = None  # remaining authors (None = same)
    form: str = "initials"  # full | first_full | initials
    ini_style: str = "dot_space"
    sep: str = ", "
    last: str | None = " & "  # joiner before the final name; None = sep
    two: str | None = None  # joiner when exactly two names; None = last
    etal_after: int = 7  # show at most this many, then et al.
    etal_show: int = 1
    etal: str = " et al."
    editors_suffix: str = ""


def add_people(B: Builder, people: list[dict[str, str]], role: str, ns: NameStyle, rng: Rng) -> bool:
    """Render a name list. Returns True if 'et al.' truncation happened."""
    n = len(people)
    truncated = False
    shown = people
    if n > ns.etal_after:
        shown = people[: ns.etal_show]
        truncated = True
    for i, p in enumerate(shown):
        if i > 0:
            if truncated:
                B.o(ns.sep)
            elif i == len(shown) - 1:
                joiner = ns.last if ns.last is not None else ns.sep
                if len(shown) == 2 and ns.two is not None:
                    joiner = ns.two
                B.o(joiner)
            else:
                B.o(ns.sep)
        mode = ns.mode if i == 0 or ns.rest_mode is None else ns.rest_mode
        add_person(B, p, role, mode, ns.form, ns.ini_style)
    if truncated:
        B.o(ns.etal)
    return truncated


# ------------------------------------------------------------------- field helpers


def sentence_case(title: str) -> str:
    words = title.split(" ")
    out = []
    for i, w in enumerate(words):
        if i > 0 and len(w) > 1 and w[0].isupper() and w[1:].islower() and w.lower() not in {"i"}:
            out.append(w.lower())
        else:
            out.append(w)
    return " ".join(out)


def title_case(title: str) -> str:
    words = title.split(" ")
    out = []
    for i, w in enumerate(words):
        if w and w.islower() and (i == 0 or w not in STOP):
            out.append(w[0].upper() + w[1:])
        else:
            out.append(w)
    return " ".join(out)


def abbreviate(container: str, rng: Rng) -> str:
    words = container.split(" ")
    out = []
    for w in words:
        lw = w.lower().strip(",.:;")
        if lw in STOP:
            continue
        if lw in ABBREV:
            out.append(ABBREV[lw])
        elif len(lw) > 6 and rng.random() < 0.6 and w[0].isalpha():
            out.append(w[:4] + ".")
        else:
            out.append(w)
    return " ".join(out) if out else container


def maybe_typo(text: str, rng: Rng, p: float) -> str:
    """Length-preserving noise: swap or replace one character in a long word."""
    if rng.random() >= p or len(text) < 6:
        return text
    chars = list(text)
    idx = [i for i, c in enumerate(chars) if c.isalpha() and 0 < i < len(chars) - 1]
    if len(idx) < 2:
        return text
    i = rng.choice(idx)
    if rng.random() < 0.5 and chars[i + 1].isalpha():
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    else:
        chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def dash(rng: Rng) -> str:
    return rng.choices(["–", "-", "—", " - ", "‐"], weights=[45, 40, 8, 4, 3])[0]


def page_text(rec: dict[str, Any], rng: Rng, abbreviate_end: bool = False) -> str:
    p = rec["pages"]
    a, b = p["from"], p["to"]
    if not b or b == a:
        return a
    if abbreviate_end and a.isdigit() and b.isdigit() and len(a) == len(b) and len(a) >= 2 and rng.random() < 0.5:
        i = 0
        while i < len(a) - 1 and a[i] == b[i]:
            i += 1
        b = b[i:]
    return f"{a}{dash(rng)}{b}"


def add_pages(B: Builder, rec: dict[str, Any], rng: Rng, prefix: str | None = None) -> None:
    if "pages" not in rec:
        return
    text = page_text(rec, rng)
    single = not rec["pages"]["to"] or rec["pages"]["to"] == rec["pages"]["from"]
    if prefix is None:
        B.f(text, "PAGES")
        return
    if single and prefix in ("pp. ", "pp ", "pp.", "pages "):
        prefix = rng.choice(["p. ", "p ", "Article ", "art. no. ", "e"])
        if prefix == "e" and text.startswith("e"):
            prefix = "p. "
    B.o(prefix)
    B.f(text, "PAGES")


def add_volume_issue(B: Builder, rec: dict[str, Any], rng: Rng, style: str) -> None:
    vol, iss = rec.get("volume"), rec.get("issue")
    if not vol:
        if iss and style in ("paren", "no"):
            B.o(rng.choice(["no. ", "No. ", "(", "issue "]))
            B.f(iss, "ISSUE")
        return
    if style == "paren":  # 12(3)
        B.f(vol, "VOLUME")
        if iss:
            B.o(rng.choice(["(", "(", " ("]))
            B.f(iss, "ISSUE")
            B.o(")")
    elif style == "vol_no":  # vol. 12, no. 3
        B.o(rng.choice(["vol. ", "Vol. ", "vol ", "Volume ", "v. "]))
        B.f(vol, "VOLUME")
        if iss:
            B.o(rng.choice([", no. ", ", No. ", " no. ", ", number ", ", issue ", ", Issue "]))
            B.f(iss, "ISSUE")
    elif style == "bare":  # 12
        B.f(vol, "VOLUME")
    elif style == "colon":  # 12:3
        B.f(vol, "VOLUME")
        if iss:
            B.o(":")
            B.f(iss, "ISSUE")
    elif style == "bd":  # Bd. 12, H. 3 / Jg. 12
        B.o(rng.choice(["Bd. ", "Jg. ", "Band "]))
        B.f(vol, "VOLUME")
        if iss:
            B.o(rng.choice([", H. ", ", Heft ", ", Nr. "]))
            B.f(iss, "ISSUE")


def add_year(B: Builder, rec: dict[str, Any], rng: Rng, form: str = "year") -> None:
    y = rec.get("year")
    if y is None:
        B.f(rng.choice(["n.d.", "n. d.", "no date", "s.d."]), "YEAR")
        return
    if form == "year":
        B.f(str(y), "YEAR")
    elif form == "month_year":
        m = rec.get("month") or rng.randint(1, 12)
        B.f(f"{rng.choice([MONTHS, MONTHS_ABBR, MONTHS_3])[m - 1]} {y}", "YEAR")
    elif form == "apa_full":
        m = rec.get("month") or rng.randint(1, 12)
        B.f(f"{y}, {MONTHS[m - 1]} {rng.randint(1, 28)}", "YEAR")
    elif form == "dmy":
        m = rec.get("month") or rng.randint(1, 12)
        B.f(f"{rng.randint(1, 28)} {rng.choice([MONTHS, MONTHS_ABBR])[m - 1]} {y}", "YEAR")
    elif form == "ymd":
        m = rec.get("month") or rng.randint(1, 12)
        B.f(f"{y} {MONTHS_3[m - 1]} {rng.randint(1, 28)}", "YEAR")
    elif form == "iso":
        m = rec.get("month") or rng.randint(1, 12)
        B.f(f"{y}-{m:02d}-{rng.randint(1, 28):02d}", "YEAR")
    else:
        B.f(str(y), "YEAR")


def access_date(rng: Rng, form: str) -> str:
    y = rng.randint(2005, 2026)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    if form == "mdy":
        return f"{MONTHS[m - 1]} {d}, {y}"
    if form == "dmy":
        return f"{d} {MONTHS[m - 1]} {y}"
    if form == "ieee":
        return f"{d}-{MONTHS_3[m - 1]}-{y}"
    if form == "van":
        return f"{y} {MONTHS_3[m - 1]} {d}"
    if form == "iso":
        return f"{y}-{m:02d}-{d:02d}"
    return f"{d} {MONTHS_ABBR[m - 1]} {y}"


def add_doi(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    doi = rec.get("doi")
    if not doi:
        return
    if rng.random() < 0.15:
        doi = doi.upper() if rng.random() < 0.3 else doi
    form = rng.choices(["https", "doi:", "DOI: ", "doi.org", "bare", "dx"], weights=[40, 20, 15, 8, 7, 10])[0]
    prefix = {
        "https": "https://doi.org/",
        "doi:": "doi:",
        "DOI: ": rng.choice(["DOI: ", "DOI:", "doi: ", "DOI "]),
        "doi.org": "doi.org/",
        "bare": "",
        "dx": "http://dx.doi.org/",
    }[form]
    B.o(prefix)
    B.f(doi, "DOI")


def add_arxiv(B: Builder, rec: dict[str, Any], rng: Rng, form: str) -> None:
    aid = rec["arxiv"]
    if form == "prefix":
        B.o(rng.choice(["arXiv:", "arXiv: ", "arxiv:", "arXiv preprint arXiv:", "arXiv e-prints, arXiv:", "ArXiv:"]))
        B.f(aid, "ARXIV")
        if rec.get("category") and rng.random() < 0.5:
            B.o(f" [{rec['category']}]")
    elif form == "url":
        B.f(f"https://arxiv.org/abs/{aid}", "URL")
    elif form == "bracket":
        B.o("[")
        B.f(aid, "ARXIV")
        B.o("]")
    elif form == "eprint":
        B.o(rng.choice(["e-print ", "eprint ", "arXiv e-print "]))
        B.f(aid, "ARXIV")
    else:
        B.f(aid, "ARXIV")


def add_url(B: Builder, rec: dict[str, Any], rng: Rng, prefix: str = "") -> None:
    url = rec.get("url")
    if not url:
        return
    B.o(prefix)
    B.f(url, "URL")


def add_accessed(B: Builder, rng: Rng, form: str) -> None:
    if form == "apa6":
        B.o(rng.choice(["Retrieved ", "Retrieved on "]))
        B.f(access_date(rng, "mdy"), "ACCESSED")
        B.o(rng.choice([", from ", " from "]))
    elif form == "chicago":
        B.o(rng.choice(["Accessed ", "Accessed on ", "accessed "]))
        B.f(access_date(rng, "dmy"), "ACCESSED")
        B.o(".")
    elif form == "ieee":
        B.o(rng.choice(["[Accessed: ", "[accessed ", "(accessed "]))
        B.f(access_date(rng, "ieee" if rng.random() < 0.6 else "dmy"), "ACCESSED")
        B.o(rng.choice(["].", ")", "]"]))
    elif form == "harvard":
        B.o(rng.choice(["(Accessed: ", "(accessed ", "(Accessed "]))
        B.f(access_date(rng, "dmy"), "ACCESSED")
        B.o(rng.choice([").", ")"]))
    elif form == "van":
        B.o(rng.choice(["[cited ", "[accessed "]))
        B.f(access_date(rng, "van"), "ACCESSED")
        B.o("]")
    elif form == "viewed":
        B.o(rng.choice(["viewed ", "Viewed ", "last accessed ", "retrieved "]))
        B.f(access_date(rng, rng.choice(["dmy", "iso", "mdy"])), "ACCESSED")
        B.o(rng.choice([".", ",", ""]))


def add_edition(B: Builder, rec: dict[str, Any], rng: Rng, form: str = "paren") -> None:
    ed = rec.get("edition")
    if not ed:
        return
    if form == "paren":
        B.o(" (")
        B.f(ed, "EDITION")
        B.o(rng.choice([" ed.)", " edn.)", " edition)", " ed)"]))
    elif form == "plain":
        B.o(rng.choice([". ", ", "]))
        B.f(ed, "EDITION")
        B.o(rng.choice([" ed.", " edn.", " edition", " ed"]))
    elif form == "german":
        B.o(", ")
        B.f(ed, "EDITION")
        B.o(rng.choice([". Aufl.", ". Auflage"]))


def add_pub(B: Builder, rec: dict[str, Any], rng: Rng, form: str) -> None:
    pub, loc = rec.get("publisher"), rec.get("location")
    if form == "loc_colon_pub":
        if loc:
            B.f(loc, "LOCATION")
            B.o(rng.choice([": ", ": ", " : ", ":"]) if pub else "")
        if pub:
            B.f(pub, "PUBLISHER")
    elif form == "pub_comma_loc":
        if pub:
            B.f(pub, "PUBLISHER")
            if loc:
                B.o(", ")
        if loc:
            B.f(loc, "LOCATION")
    elif form == "pub_only":
        if pub:
            B.f(pub, "PUBLISHER")
    elif form == "loc_only":
        if loc:
            B.f(loc, "LOCATION")


def add_title(B: Builder, rec: dict[str, Any], rng: Rng, quote: str = "", case: str = "keep", end: str = "") -> None:
    t = rec["title"]
    if case == "sentence":
        t = sentence_case(t)
    elif case == "title":
        t = title_case(t)
    elif case == "upper":
        t = t.upper()
    open_q, close_q = {"": ("", ""), '"': ('"', '"'), "“": ("“", "”"), "‘": ("‘", "’"), "«": ("« ", " »"), "'": ("'", "'")}[quote]
    B.o(open_q)
    B.f(t, "TITLE")
    if end and rng.random() < 0.85:
        if quote and rng.random() < 0.6:
            B.o(end + close_q)
            return
    B.o(close_q)
    if end:
        B.o(end)


def add_container(B: Builder, rec: dict[str, Any], rng: Rng, abbrev_p: float = 0.0, markers: bool = False) -> None:
    c = rec.get("container")
    if not c:
        return
    if rec.get("container_short") and rng.random() < abbrev_p:
        c = rec["container_short"]
    elif rng.random() < abbrev_p * 0.6:
        c = abbreviate(c, rng)
    if markers and rng.random() < 0.08:
        m = rng.choice(["*", "_"])
        B.o(m)
        B.f(c, "CONTAINER")
        B.o(m)
    else:
        B.f(c, "CONTAINER")


def genre_text(rec: dict[str, Any], rng: Rng) -> str:
    if rec["type"] == "thesis":
        return rng.choice([
            "PhD thesis", "Ph.D. thesis", "Ph.D. dissertation", "Doctoral dissertation", "PhD dissertation",
            "Master's thesis", "MSc thesis", "M.S. thesis", "Diploma thesis", "Dissertation", "Thesis (PhD)",
            "Bachelor's thesis", "Habilitation thesis", "Doctoral thesis", "Tesis doctoral", "Thèse de doctorat",
        ])
    if rec["type"] == "report":
        num = rec.get("number", "")
        return rng.choice([
            f"Technical Report {num}", f"Tech. Rep. {num}", f"Technical report {num}", f"Report No. {num}",
            f"Working Paper {num}", f"Working Paper No. {num}", f"Discussion Paper {num}", f"White paper",
            f"Research Report {num}", f"Tech. Report {num}", f"Report {num}", f"Memo {num}", f"TR-{num}",
            f"NBER Working Paper {num}", f"Technical Memorandum {num}", f"Staff Report {num}",
        ]).strip()
    return ""


# ------------------------------------------------------------------------- styles
# Each style: fn(B, rec, rng) -> None. rec is a realized record (see realize()).


def _authors_or_title_first(B: Builder, rec: dict[str, Any], ns: NameStyle, rng: Rng) -> bool:
    if rec.get("authors"):
        return add_people(B, rec["authors"], "AUTHOR", ns, rng)
    return False


def style_apa(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="fam_giv", form="initials", ini_style=rng.choice(["dot_space", "dot", "dot_space"]),
                   sep=", ", last=rng.choice([", & ", " & ", ", and "]), etal_after=rng.choice([7, 20, 6]),
                   etal_show=rng.choice([1, 6]), etal=rng.choice([", et al.", " et al.", ", … "]))
    t = rec["type"]
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(" ") if not B.text().endswith(".") else B.o(" ")
    B.o("(")
    add_year(B, rec, rng, "apa_full" if t == "web" and rng.random() < 0.7 else "year")
    B.o("). ")
    if t == "article":
        add_title(B, rec, rng, case=rng.choice(["sentence", "keep"]), end=".")
        B.o(" ")
        add_container(B, rec, rng, abbrev_p=0.1, markers=True)
        if rec.get("volume"):
            B.o(", ")
            add_volume_issue(B, rec, rng, "paren")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng)
        B.o(". ")
        add_doi(B, rec, rng)
    elif t == "chapter":
        add_title(B, rec, rng, case="sentence", end=".")
        B.o(" In ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="initials", last=" & "), rng)
            B.o(rng.choice([" (Eds.), ", " (Ed.), ", " (eds.), "]))
        add_container(B, rec, rng, markers=True)
        if rec.get("pages"):
            B.o(" (")
            add_pages(B, rec, rng, prefix="pp. ")
            B.o(")")
        B.o(". ")
        add_pub(B, rec, rng, rng.choice(["pub_only", "loc_colon_pub"]))
        B.o(".")
        if rec.get("doi") and rng.random() < 0.5:
            B.o(" ")
            add_doi(B, rec, rng)
    elif t == "book":
        add_title(B, rec, rng, case="sentence")
        add_edition(B, rec, rng, "paren")
        B.o(". ")
        add_pub(B, rec, rng, rng.choice(["pub_only", "loc_colon_pub"]))
        B.o(".")
        if rec.get("doi") and rng.random() < 0.3:
            B.o(" ")
            add_doi(B, rec, rng)
    elif t == "conference":
        add_title(B, rec, rng, case="sentence", end=".")
        B.o(rng.choice([" In ", " In: ", " Paper presented at the ", " In "]))
        add_container(B, rec, rng, markers=True)
        if rec.get("pages"):
            B.o(" (")
            add_pages(B, rec, rng, prefix="pp. ")
            B.o(")")
        B.o(". ")
        if rec.get("location") and rng.random() < 0.5:
            B.f(rec["location"], "LOCATION")
            B.o(". ")
        add_doi(B, rec, rng)
    elif t == "thesis":
        add_title(B, rec, rng, case="sentence")
        B.o(" [" + genre_text(rec, rng) + ", ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o("].")
        if rec.get("url") and rng.random() < 0.5:
            B.o(" ")
            add_url(B, rec, rng)
    elif t == "report":
        add_title(B, rec, rng, case="sentence")
        B.o(" (" + genre_text(rec, rng) + "). ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o(".")
        if rec.get("url") and rng.random() < 0.5:
            B.o(" ")
            add_url(B, rec, rng)
    elif t == "web":
        add_title(B, rec, rng, case="sentence", end=".")
        B.o(" ")
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(". ")
        if rng.random() < 0.4:
            add_accessed(B, rng, "apa6")
        add_url(B, rec, rng)
    elif t == "preprint":
        add_title(B, rec, rng, case="sentence", end=".")
        B.o(" ")
        if rec.get("arxiv"):
            add_arxiv(B, rec, rng, rng.choice(["prefix", "url", "prefix"]))
        else:
            B.o(rng.choice(["bioRxiv. ", "PsyArXiv. ", "SSRN. ", "medRxiv. ", "Preprint. "]))
            add_doi(B, rec, rng)


def style_mla(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="fam_giv", rest_mode="giv_fam", form="full", sep=", ", last=", and ", two=", and ",
                   etal_after=2, etal_show=1, etal=", et al.")
    t = rec["type"]
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(". ")
    q = rng.choice(['"', "“", "“"])
    if t == "book":
        add_title(B, rec, rng, case="title")
        B.o(". ")
        if rec.get("edition"):
            B.f(rec["edition"], "EDITION")
            B.o(" ed., ")
        add_pub(B, rec, rng, "pub_only")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(".")
        return
    add_title(B, rec, rng, quote=q, case="title", end=".")
    B.o(" ")
    if t in ("article", "conference"):
        add_container(B, rec, rng, markers=True)
        if rec.get("volume"):
            B.o(", ")
            add_volume_issue(B, rec, rng, "vol_no")
        B.o(", ")
        add_year(B, rec, rng, rng.choice(["year", "month_year"]))
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix="pp. ")
        B.o(".")
        if rec.get("doi") and rng.random() < 0.6:
            B.o(" ")
            add_doi(B, rec, rng)
            B.o(".")
    elif t == "chapter":
        add_container(B, rec, rng, markers=True)
        if rec.get("editors"):
            B.o(", edited by ")
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="full", last=" and "), rng)
        B.o(", ")
        add_pub(B, rec, rng, "pub_only")
        B.o(", ")
        add_year(B, rec, rng)
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix="pp. ")
        B.o(".")
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(", ")
        add_year(B, rec, rng, "dmy")
        B.o(", ")
        add_url(B, rec, rng)
        B.o(". ")
        if rng.random() < 0.6:
            add_accessed(B, rng, "chicago")
    elif t == "thesis":
        B.o(genre_text(rec, rng) + ", ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "report":
        B.f(rec["publisher"], "PUBLISHER")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(". " + genre_text(rec, rng) + ".")
    elif t == "preprint":
        B.o(rng.choice(["arXiv", "ArXiv", "arXiv.org"]) + ", ")
        add_year(B, rec, rng)
        B.o(", ")
        add_arxiv(B, rec, rng, rng.choice(["url", "prefix"]))
        B.o(".")


def style_chicago_ad(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="fam_giv", rest_mode="giv_fam", form=rng.choice(["full", "first_full"]), sep=", ",
                   last=", and ", two=" and ", etal_after=10, etal_show=7, etal=", et al.")
    t = rec["type"]
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(". ")
    add_year(B, rec, rng)
    B.o(". ")
    if t == "article":
        add_title(B, rec, rng, quote=rng.choice(["“", '"']), case="title", end=".")
        B.o(" ")
        add_container(B, rec, rng, markers=True)
        if rec.get("volume"):
            B.o(" ")
            B.f(rec["volume"], "VOLUME")
            if rec.get("issue"):
                B.o(" (")
                B.f(rec["issue"], "ISSUE")
                B.o(")")
        if rec.get("pages"):
            B.o(": ")
            add_pages(B, rec, rng)
        B.o(".")
        if rec.get("doi") and rng.random() < 0.7:
            B.o(" ")
            add_doi(B, rec, rng)
            B.o(".")
    elif t == "book":
        add_title(B, rec, rng, case="title")
        B.o(". ")
        if rec.get("edition"):
            B.f(rec["edition"], "EDITION")
            B.o(" ed. ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(".")
    elif t == "chapter":
        add_title(B, rec, rng, quote="“", case="title", end=".")
        B.o(" In ")
        add_container(B, rec, rng)
        if rec.get("editors"):
            B.o(", edited by ")
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="full", last=" and ", two=" and "), rng)
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng)
        B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(".")
    elif t == "conference":
        add_title(B, rec, rng, quote="“", case="title", end=".")
        B.o(rng.choice([" In ", " Paper presented at ", " In "]))
        add_container(B, rec, rng)
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng)
        B.o(". ")
        if rec.get("location"):
            B.f(rec["location"], "LOCATION")
            B.o(rng.choice([": ", ". "]))
        if rec.get("publisher"):
            B.f(rec["publisher"], "PUBLISHER")
            B.o(".")
    elif t == "thesis":
        add_title(B, rec, rng, quote="“", case="title", end=".")
        B.o(" " + genre_text(rec, rng) + ", ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o(".")
    elif t == "report":
        add_title(B, rec, rng, case="title")
        B.o(". " + genre_text(rec, rng) + ". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(".")
    elif t == "web":
        add_title(B, rec, rng, quote="“", case="title", end=".")
        B.o(" ")
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(". ")
        if rng.random() < 0.5:
            add_accessed(B, rng, "chicago")
            B.o(" ")
        add_url(B, rec, rng)
        B.o(".")
    elif t == "preprint":
        add_title(B, rec, rng, quote="“", case="title", end=".")
        B.o(" ")
        if rec.get("arxiv"):
            B.o(rng.choice(["arXiv preprint. ", "Preprint, ", ""]))
            add_arxiv(B, rec, rng, rng.choice(["prefix", "url"]))
        else:
            B.o("Preprint. ")
            add_doi(B, rec, rng)
        B.o(".")


def style_chicago_note(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="giv_fam", form="full", sep=", ", last=" and ", two=" and ", etal_after=4, etal_show=1, etal=" et al.")
    t = rec["type"]
    if rng.random() < 0.5:
        B.o(f"{rng.randint(1, 60)}. ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(", ")
    if t == "book":
        add_title(B, rec, rng, case="title")
        B.o(" (")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(")")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng)
        B.o(".")
        return
    add_title(B, rec, rng, quote="“", case="title", end=",")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng)
        B.o(" ")
        if rec.get("volume"):
            B.f(rec["volume"], "VOLUME")
            if rec.get("issue"):
                B.o(", no. ")
                B.f(rec["issue"], "ISSUE")
            B.o(" ")
        B.o("(")
        add_year(B, rec, rng, rng.choice(["year", "month_year"]))
        B.o(")")
        if rec.get("pages"):
            B.o(": ")
            add_pages(B, rec, rng)
        B.o(".")
    elif t in ("chapter", "conference"):
        B.o("in ")
        add_container(B, rec, rng)
        if rec.get("editors"):
            B.o(", ed. ")
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="full", last=" and ", two=" and "), rng)
        B.o(" (")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(")")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng)
        B.o(".")
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(", ")
        add_year(B, rec, rng, "dmy")
        B.o(", ")
        add_url(B, rec, rng)
        B.o(".")
    elif t == "thesis":
        B.o("(" + genre_text(rec, rng) + ", ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(").")
    elif t == "report":
        B.o(genre_text(rec, rng) + " (")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(").")
    elif t == "preprint":
        B.o(rng.choice(["arXiv, ", "preprint, "]))
        add_year(B, rec, rng)
        B.o(", ")
        add_arxiv(B, rec, rng, rng.choice(["url", "prefix"]))
        B.o(".")


def style_ieee(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="giv_fam", form="initials", ini_style=rng.choice(["dot_space", "dot"]), sep=", ",
                   last=rng.choice([", and ", " and "]), two=" and ", etal_after=rng.choice([6, 3]), etal_show=1, etal=" et al.")
    t = rec["type"]
    if rng.random() < 0.7:
        B.o(f"[{rng.randint(1, 99)}] ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(", ")
    q = rng.choice(["“", '"', "“"])
    if t == "book":
        add_title(B, rec, rng, case="title")
        if rec.get("edition"):
            B.o(", ")
            B.f(rec["edition"], "EDITION")
            B.o(" ed")
        B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(".")
        return
    add_title(B, rec, rng, quote=q, case=rng.choice(["title", "keep"]), end=",")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, abbrev_p=0.6, markers=True)
        B.o(", ")
        if rec.get("volume"):
            add_volume_issue(B, rec, rng, "vol_no")
            B.o(", ")
        if rec.get("pages"):
            add_pages(B, rec, rng, prefix=rng.choice(["pp. ", "pp. ", "p. "]))
            B.o(", ")
        add_year(B, rec, rng, rng.choice(["year", "month_year"]))
        B.o(rng.choice([".", ", ", "."]))
        if rec.get("doi") and rng.random() < 0.6:
            B.o(" ")
            add_doi(B, rec, rng)
            B.o(".")
    elif t == "conference":
        B.o(rng.choice(["in ", "in ", "In ", "presented at the "]))
        add_container(B, rec, rng, abbrev_p=0.5)
        B.o(", ")
        if rec.get("location") and rng.random() < 0.6:
            B.f(rec["location"], "LOCATION")
            B.o(", ")
        add_year(B, rec, rng, rng.choice(["year", "month_year"]))
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix="pp. ")
        B.o(".")
        if rec.get("doi") and rng.random() < 0.4:
            B.o(" ")
            add_doi(B, rec, rng)
    elif t == "chapter":
        B.o("in ")
        add_container(B, rec, rng)
        if rec.get("editors"):
            B.o(", ")
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="initials", last=" and "), rng)
            B.o(rng.choice([", Eds. ", ", Ed. ", ", eds. "]))
        else:
            B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(", ")
        add_year(B, rec, rng)
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix="pp. ")
        B.o(".")
    elif t == "thesis":
        B.o(genre_text(rec, rng) + ", ")
        if rng.random() < 0.5:
            B.o(rng.choice(DEPARTMENTS) + ", ")
        B.f(rec["publisher"], "PUBLISHER")
        if rec.get("location"):
            B.o(", ")
            B.f(rec["location"], "LOCATION")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "report":
        B.f(rec["publisher"], "PUBLISHER")
        if rec.get("location"):
            B.o(", ")
            B.f(rec["location"], "LOCATION")
        B.o(", " + genre_text(rec, rng) + ", ")
        add_year(B, rec, rng, rng.choice(["year", "month_year"]))
        B.o(".")
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(". ")
        B.o(rng.choice(["[Online]. Available: ", "Accessed: ", "[Online]. Available at: "]))
        add_url(B, rec, rng)
        B.o(" ")
        add_accessed(B, rng, "ieee")
    elif t == "preprint":
        add_year(B, rec, rng)
        B.o(", ")
        add_arxiv(B, rec, rng, rng.choice(["prefix", "prefix", "url"]))
        B.o(".")


def style_vancouver(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="van", sep=", ", last=None, etal_after=6, etal_show=rng.choice([6, 3]), etal=rng.choice([", et al.", " et al."]))
    t = rec["type"]
    if rng.random() < 0.6:
        B.o(f"{rng.randint(1, 120)}. ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(". ")
    add_title(B, rec, rng, case="sentence", end=".")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, abbrev_p=0.7)
        B.o(rng.choice([". ", " ", ". "]))
        add_year(B, rec, rng, rng.choice(["year", "ymd", "year"]))
        if rec.get("volume"):
            B.o(rng.choice([";", "; "]))
            add_volume_issue(B, rec, rng, "paren")
        if rec.get("pages"):
            B.o(":")
            B.f(page_text(rec, rng, abbreviate_end=True), "PAGES")
        B.o(".")
        if rec.get("doi") and rng.random() < 0.5:
            B.o(" ")
            add_doi(B, rec, rng)
    elif t == "book":
        if rec.get("edition"):
            B.f(rec["edition"], "EDITION")
            B.o(" ed. ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o("; ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "chapter":
        B.o("In: ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="van", sep=", ", last=None), rng)
            B.o(", editors. ")
        add_container(B, rec, rng)
        B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o("; ")
        add_year(B, rec, rng)
        if rec.get("pages"):
            B.o(". p. ")
            B.f(page_text(rec, rng, abbreviate_end=True), "PAGES")
        B.o(".")
    elif t == "conference":
        B.o("In: ")
        add_container(B, rec, rng)
        B.o("; ")
        add_year(B, rec, rng, rng.choice(["year", "ymd"]))
        if rec.get("location"):
            B.o("; ")
            B.f(rec["location"], "LOCATION")
        B.o(". ")
        if rec.get("publisher"):
            B.f(rec["publisher"], "PUBLISHER")
            B.o("; ")
        if rec.get("pages"):
            B.o("p. ")
            B.f(page_text(rec, rng, abbreviate_end=True), "PAGES")
        B.o(".")
    elif t == "thesis":
        B.o("[" + genre_text(rec, rng).lower() + "]. ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o("; ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "report":
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o("; ")
        add_year(B, rec, rng)
        B.o(". " + genre_text(rec, rng) + ".")
    elif t == "web":
        B.o("[Internet]. ")
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o("; ")
        add_year(B, rec, rng, "ymd")
        B.o(" ")
        add_accessed(B, rng, "van")
        B.o(". Available from: ")
        add_url(B, rec, rng)
    elif t == "preprint":
        B.o(rng.choice(["arXiv [Preprint]. ", "bioRxiv [Preprint]. ", "Preprint. "]))
        add_year(B, rec, rng)
        B.o(". ")
        if rec.get("arxiv"):
            add_arxiv(B, rec, rng, rng.choice(["url", "prefix"]))
        else:
            add_doi(B, rec, rng)
        B.o(".")


def style_harvard(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode=rng.choice(["fam_giv", "fam_giv_nocomma"]), form="initials", ini_style=rng.choice(["dot", "dot_space", "plain"]),
                   sep=", ", last=rng.choice([" and ", " & ", ", and "]), etal_after=rng.choice([3, 4, 8]), etal_show=1, etal=" et al.")
    t = rec["type"]
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(rng.choice([" (", ", (", " "]))
    else:
        B.o("(")
    add_year(B, rec, rng)
    B.o(rng.choice([") ", "). ", ") "]))
    if t == "article":
        add_title(B, rec, rng, quote=rng.choice(["", "'", "‘"]), case="sentence", end=rng.choice([".", ","]))
        B.o(" ")
        add_container(B, rec, rng, abbrev_p=0.2, markers=True)
        if rec.get("volume"):
            B.o(", ")
            add_volume_issue(B, rec, rng, "paren")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix=rng.choice(["pp. ", "pp.", "pp "]))
        B.o(".")
        if rec.get("doi") and rng.random() < 0.5:
            B.o(rng.choice([" doi: ", " Available at: ", " "]))
            B.f(rec["doi"], "DOI")
            B.o(".")
    elif t == "book":
        add_title(B, rec, rng, case="sentence")
        B.o(". ")
        if rec.get("edition"):
            B.f(rec["edition"], "EDITION")
            B.o(" edn. ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(".")
    elif t == "chapter":
        add_title(B, rec, rng, quote=rng.choice(["", "'"]), case="sentence", end=",")
        B.o(" in ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="fam_giv", form="initials", ini_style="dot", last=" and "), rng)
            B.o(rng.choice([" (eds.) ", " (eds) ", " (ed.) "]))
        add_container(B, rec, rng)
        B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix="pp. ")
        B.o(".")
    elif t == "conference":
        add_title(B, rec, rng, quote=rng.choice(["", "'"]), case="sentence", end=",")
        B.o(" ")
        add_container(B, rec, rng)
        B.o(". ")
        if rec.get("location"):
            B.f(rec["location"], "LOCATION")
            B.o(", ")
        add_year(B, rec, rng, "dmy")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix="pp. ")
        B.o(".")
    elif t == "thesis":
        add_title(B, rec, rng, case="sentence")
        B.o(". " + genre_text(rec, rng) + ". ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o(".")
    elif t == "report":
        add_title(B, rec, rng, case="sentence")
        B.o(". " + genre_text(rec, rng) + ". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(".")
    elif t == "web":
        add_title(B, rec, rng, case="sentence")
        B.o(". ")
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(". ")
        B.o(rng.choice(["Available at: ", "[Online] Available at: ", "Available from: "]))
        add_url(B, rec, rng)
        B.o(" ")
        add_accessed(B, rng, "harvard")
    elif t == "preprint":
        add_title(B, rec, rng, case="sentence")
        B.o(". ")
        if rec.get("arxiv"):
            add_arxiv(B, rec, rng, rng.choice(["prefix", "url"]))
        else:
            B.o("Preprint. ")
            add_doi(B, rec, rng)
        B.o(".")


def style_nature(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="fam_giv", form="initials", ini_style=rng.choice(["dot_space", "dot"]), sep=", ", last=" & ",
                   etal_after=5, etal_show=1, etal=" et al.")
    t = rec["type"]
    if rng.random() < 0.5:
        B.o(f"{rng.randint(1, 80)}. ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(" ")
    if t == "article":
        add_title(B, rec, rng, case="sentence", end=".")
        B.o(" ")
        add_container(B, rec, rng, abbrev_p=0.7, markers=True)
        B.o(" ")
        if rec.get("volume"):
            B.f(rec["volume"], "VOLUME")
            B.o(", ")
        if rec.get("pages"):
            add_pages(B, rec, rng)
            B.o(" ")
        B.o("(")
        add_year(B, rec, rng)
        B.o(").")
        if rec.get("doi") and rng.random() < 0.3:
            B.o(" ")
            add_doi(B, rec, rng)
    elif t == "book":
        add_title(B, rec, rng, case="sentence")
        B.o(" (")
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(").")
    elif t in ("chapter", "conference"):
        add_title(B, rec, rng, case="sentence", end=".")
        B.o(rng.choice([" in ", " In "]))
        add_container(B, rec, rng)
        if rec.get("editors"):
            B.o(" (")
            B.o(rng.choice(["eds ", "ed. ", "eds. "]))
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="fam_giv", form="initials", last=" & "), rng)
            B.o(")")
        if rec.get("pages"):
            B.o(" ")
            add_pages(B, rec, rng)
        B.o(" (")
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(").")
    elif t == "thesis":
        add_title(B, rec, rng, case="sentence")
        B.o(". " + genre_text(rec, rng) + ", ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o(" (")
        add_year(B, rec, rng)
        B.o(").")
    elif t == "report":
        add_title(B, rec, rng, case="sentence")
        B.o(". " + genre_text(rec, rng) + " (")
        B.f(rec["publisher"], "PUBLISHER")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(").")
    elif t == "web":
        add_title(B, rec, rng, case="sentence")
        B.o(". ")
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(" ")
        add_url(B, rec, rng)
        B.o(" (")
        add_year(B, rec, rng)
        B.o(").")
    elif t == "preprint":
        add_title(B, rec, rng, case="sentence")
        B.o(". ")
        B.o(rng.choice(["Preprint at ", "Preprint at ", ""]))
        if rec.get("arxiv"):
            add_arxiv(B, rec, rng, rng.choice(["url", "url", "prefix"]))
        else:
            add_doi(B, rec, rng)
        B.o(" (")
        add_year(B, rec, rng)
        B.o(").")


def style_acm(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="giv_fam", form="full", sep=", ", last=", and ", two=" and ", etal_after=12, etal_show=1, etal=" et al.")
    t = rec["type"]
    if rng.random() < 0.6:
        B.o(f"[{rng.randint(1, 60)}] ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(". ")
    add_year(B, rec, rng)
    B.o(". ")
    add_title(B, rec, rng, case="title", end=".")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, markers=True)
        B.o(" ")
        if rec.get("volume"):
            B.f(rec["volume"], "VOLUME")
            if rec.get("issue"):
                B.o(", ")
                B.f(rec["issue"], "ISSUE")
            B.o(" ")
        B.o("(")
        add_year(B, rec, rng, rng.choice(["month_year", "year"]))
        B.o(")")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng)
        B.o(". ")
        add_doi(B, rec, rng)
    elif t == "conference":
        B.o("In ")
        add_container(B, rec, rng)
        B.o(rng.choice([" (", " ("]))
        if rec.get("event_short"):
            B.o(rec["event_short"] + " ")
        B.o("'" + str(rec["year"])[-2:] + "). ")
        if rec.get("publisher"):
            B.f(rec["publisher"], "PUBLISHER")
            B.o(", ")
        if rec.get("location"):
            B.f(rec["location"], "LOCATION")
            B.o(", ")
        if rec.get("pages"):
            add_pages(B, rec, rng)
            B.o(". ")
        add_doi(B, rec, rng)
    elif t == "chapter":
        B.o("In ")
        add_container(B, rec, rng)
        if rec.get("editors"):
            B.o(", ")
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="full", last=" and "), rng)
            B.o(" (Eds.)")
        B.o(". ")
        add_pub(B, rec, rng, "pub_comma_loc")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng)
        B.o(".")
    elif t == "book":
        add_edition(B, rec, rng, "paren") if rng.random() < 0.5 else None
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(".")
    elif t == "thesis":
        B.o(genre_text(rec, rng) + ". ")
        B.f(rec["publisher"], "PUBLISHER")
        if rec.get("location"):
            B.o(", ")
            B.f(rec["location"], "LOCATION")
        B.o(".")
    elif t == "report":
        B.o(genre_text(rec, rng) + ". ")
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(".")
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(". ")
        B.o(rng.choice(["Retrieved from ", "", "Retrieved from "]))
        add_url(B, rec, rng)
    elif t == "preprint":
        if rec.get("arxiv"):
            add_arxiv(B, rec, rng, rng.choice(["prefix", "url"]))
        else:
            add_doi(B, rec, rng)
        B.o(".")


def style_ama(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="van", sep=", ", last=None, etal_after=6, etal_show=3, etal=", et al")
    t = rec["type"]
    if rng.random() < 0.5:
        B.o(f"{rng.randint(1, 60)}. ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(". ")
    add_title(B, rec, rng, case="sentence", end=".")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, abbrev_p=0.7, markers=True)
        B.o(". ")
        add_year(B, rec, rng)
        if rec.get("volume"):
            B.o(";")
            add_volume_issue(B, rec, rng, "paren")
        if rec.get("pages"):
            B.o(":")
            add_pages(B, rec, rng)
        B.o(". ")
        add_doi(B, rec, rng)
    elif t == "book":
        if rec.get("edition"):
            B.f(rec["edition"], "EDITION")
            B.o(" ed. ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o("; ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "chapter":
        B.o("In: ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="van", sep=", ", last=None), rng)
            B.o(", eds. ")
        add_container(B, rec, rng)
        B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o("; ")
        add_year(B, rec, rng)
        if rec.get("pages"):
            B.o(":")
            add_pages(B, rec, rng)
        B.o(".")
    else:
        style_vancouver_tail(B, rec, rng)


def style_vancouver_tail(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    t = rec["type"]
    if t == "conference":
        B.o("In: ")
        add_container(B, rec, rng)
        B.o("; ")
        add_year(B, rec, rng)
        B.o("; ")
        if rec.get("location"):
            B.f(rec["location"], "LOCATION")
        B.o(".")
    elif t == "thesis":
        B.o(genre_text(rec, rng) + ". ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o("; ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "report":
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o("; ")
        add_year(B, rec, rng)
        B.o(". " + genre_text(rec, rng) + ".")
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(". ")
        add_url(B, rec, rng)
        B.o(". ")
        add_accessed(B, rng, "viewed")
    elif t == "preprint":
        add_year(B, rec, rng)
        B.o(". ")
        if rec.get("arxiv"):
            add_arxiv(B, rec, rng, "prefix")
        else:
            add_doi(B, rec, rng)


def style_elsevier(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="giv_fam", form="initials", ini_style=rng.choice(["dot", "dot_space"]), sep=", ", last=", ",
                   etal_after=rng.choice([3, 5]), etal_show=1, etal=" et al.")
    t = rec["type"]
    if rng.random() < 0.7:
        B.o(rng.choice([f"[{rng.randint(1, 60)}] ", f"{rng.randint(1, 60)}. ", f"({rng.randint(1, 60)}) "]))
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(", ")
    add_title(B, rec, rng, case="sentence", end=",")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, abbrev_p=0.6, markers=True)
        B.o(rng.choice([" ", ", ", " "]))
        if rec.get("volume"):
            B.f(rec["volume"], "VOLUME")
            if rec.get("issue") and rng.random() < 0.5:
                B.o(" (")
                B.f(rec["issue"], "ISSUE")
                B.o(")")
            B.o(" ")
        B.o("(")
        add_year(B, rec, rng)
        B.o(")")
        if rec.get("pages"):
            B.o(" ")
            add_pages(B, rec, rng)
        B.o(rng.choice([".", ", ", "."]))
        if rec.get("doi") and rng.random() < 0.6:
            B.o(" ")
            add_doi(B, rec, rng)
            B.o(".")
    elif t in ("chapter", "conference"):
        B.o("in: ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="initials", ini_style="dot", last=", "), rng)
            B.o(rng.choice([" (Eds.), ", " (Ed.), "]))
        add_container(B, rec, rng)
        B.o(", ")
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(", ")
        add_year(B, rec, rng)
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix="pp. ")
        B.o(".")
    elif t == "book":
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(".")
    else:
        style_vancouver_tail(B, rec, rng)


def style_springer(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="fam_giv_nocomma", form="initials", ini_style=rng.choice(["plain", "dot"]), sep=", ", last=", ",
                   etal_after=rng.choice([3, 6]), etal_show=rng.choice([1, 3]), etal=" et al")
    t = rec["type"]
    if rng.random() < 0.5:
        B.o(f"{rng.randint(1, 60)}. ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(" ")
    B.o("(")
    add_year(B, rec, rng)
    B.o(") ")
    add_title(B, rec, rng, case="sentence", end=".")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, abbrev_p=0.5)
        B.o(" ")
        if rec.get("volume"):
            add_volume_issue(B, rec, rng, rng.choice(["bare", "paren"]))
        if rec.get("pages"):
            B.o(":")
            add_pages(B, rec, rng)
        B.o(". ")
        add_doi(B, rec, rng)
    elif t in ("chapter", "conference"):
        B.o("In: ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="fam_giv_nocomma", form="initials", ini_style="plain", last=", "), rng)
            B.o(rng.choice([" (eds) ", " (ed) "]))
        add_container(B, rec, rng)
        B.o(". ")
        add_pub(B, rec, rng, "pub_comma_loc")
        if rec.get("pages"):
            B.o(", pp ")
            B.f(page_text(rec, rng), "PAGES")
        B.o(".")
    elif t == "book":
        if rec.get("edition"):
            B.f(rec["edition"], "EDITION")
            B.o(" edn. ")
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(".")
    else:
        style_vancouver_tail(B, rec, rng)


def style_plain(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    """BibTeX plain / abbrv."""
    ns = NameStyle(mode="giv_fam", form=rng.choice(["full", "initials"]), ini_style="dot_space", sep=", ", last=", and ", two=" and ",
                   etal_after=20)
    t = rec["type"]
    if rng.random() < 0.6:
        B.o(f"[{rng.randint(1, 40)}] ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(". ")
    add_title(B, rec, rng, case="keep", end=".")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, abbrev_p=0.3)
        B.o(", ")
        if rec.get("volume"):
            add_volume_issue(B, rec, rng, "paren")
            if rec.get("pages"):
                B.o(":")
                add_pages(B, rec, rng)
            B.o(", ")
        add_year(B, rec, rng, rng.choice(["year", "month_year"]))
        B.o(".")
    elif t in ("chapter", "conference"):
        B.o("In ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="full", last=" and "), rng)
            B.o(rng.choice([", editors, ", ", editor, "]))
        add_container(B, rec, rng)
        if rec.get("volume"):
            B.o(", volume ")
            B.f(rec["volume"], "VOLUME")
        if rec.get("pages"):
            B.o(", pages ")
            B.f(page_text(rec, rng), "PAGES")
        B.o(". ")
        if rec.get("publisher"):
            B.f(rec["publisher"], "PUBLISHER")
            B.o(", ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "book":
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(", ")
        if rec.get("edition"):
            B.f(rec["edition"], "EDITION")
            B.o(" edition, ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "thesis":
        B.o(genre_text(rec, rng) + ", ")
        B.f(rec["publisher"], "PUBLISHER")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "report":
        B.o(genre_text(rec, rng) + ", ")
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(", ")
        add_year(B, rec, rng)
        B.o(". ")
        add_url(B, rec, rng)
        B.o(".")
    elif t == "preprint":
        B.o(rng.choice(["arXiv preprint ", "CoRR, abs/", "arXiv preprint "]))
        if rec.get("arxiv"):
            if B.text().endswith("abs/"):
                B.f(rec["arxiv"], "ARXIV")
            else:
                add_arxiv(B, rec, rng, "prefix")
        else:
            add_doi(B, rec, rng)
        B.o(", ")
        add_year(B, rec, rng)
        B.o(".")


def style_arxiv_listing(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    """Single-line arXiv listing / Google Scholar / Semantic Scholar-ish forms."""
    ns = NameStyle(mode="giv_fam", form=rng.choice(["full", "initials"]), sep=", ", last=", ", etal_after=rng.choice([3, 10]), etal_show=1, etal=" et al.")
    t = rec["type"]
    form = rng.choice(["listing", "scholar", "bare", "semantic"])
    if form == "listing" and rec.get("arxiv"):
        B.o("[")
        B.f(rec["arxiv"], "ARXIV")
        B.o("] ")
        add_title(B, rec, rng, case="keep", end=".")
        B.o(" ")
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(". (")
        add_year(B, rec, rng, rng.choice(["year", "month_year"]))
        B.o(")")
        return
    if form == "scholar":
        add_people(B, rec["authors"], "AUTHOR", NameStyle(mode="van", sep=", ", last=None, etal_after=3, etal_show=1), rng)
        B.o(". ")
        add_title(B, rec, rng, case="keep", end=".")
        B.o(" ")
        if t == "preprint":
            B.o("arXiv preprint ")
            add_arxiv(B, rec, rng, "prefix")
        elif rec.get("container"):
            add_container(B, rec, rng, abbrev_p=0.3)
            if rec.get("volume"):
                B.o(". ")
                add_year(B, rec, rng)
                B.o(";")
                add_volume_issue(B, rec, rng, "paren")
                if rec.get("pages"):
                    B.o(":")
                    add_pages(B, rec, rng)
                B.o(".")
                return
        elif rec.get("publisher"):
            B.f(rec["publisher"], "PUBLISHER")
        B.o(rng.choice([". ", ", "]))
        add_year(B, rec, rng)
        B.o(".")
        return
    if form == "semantic":
        add_title(B, rec, rng, case="keep")
        B.o(" | ")
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(" | ")
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(" | ")
        add_year(B, rec, rng)
        if rec.get("doi"):
            B.o(" | ")
            add_doi(B, rec, rng)
        return
    # bare: minimal punctuation
    add_people(B, rec["authors"], "AUTHOR", ns, rng)
    B.o(" ")
    add_year(B, rec, rng)
    B.o(" ")
    add_title(B, rec, rng, case="keep")
    B.o(" ")
    if rec.get("container"):
        add_container(B, rec, rng, abbrev_p=0.4)
        B.o(" ")
    if rec.get("volume"):
        B.f(rec["volume"], "VOLUME")
        B.o(" ")
    if rec.get("pages"):
        add_pages(B, rec, rng)
    if rec.get("arxiv"):
        B.o(" ")
        add_arxiv(B, rec, rng, "prefix")


def style_wikipedia(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="fam_giv", form="full", sep="; ", last="; ", etal_after=6, etal_show=3, etal="; et al.")
    t = rec["type"]
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(" ")
    B.o("(")
    add_year(B, rec, rng, rng.choice(["year", "dmy", "month_year"]))
    B.o("). ")
    add_title(B, rec, rng, quote=rng.choice(['"', "“"]), case="keep", end=".")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, markers=True)
        B.o(". ")
        if rec.get("volume"):
            add_volume_issue(B, rec, rng, "paren")
            B.o(": ")
        if rec.get("pages"):
            add_pages(B, rec, rng)
            B.o(". ")
        if rec.get("doi"):
            B.o("doi:")
            B.f(rec["doi"], "DOI")
            B.o(". ")
        if rng.random() < 0.3:
            B.o(f"ISSN {rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}.")
    elif t in ("book", "chapter"):
        if t == "chapter" and rec.get("container"):
            B.o("In ")
            if rec.get("editors"):
                add_people(B, rec["editors"], "EDITOR", NameStyle(mode="fam_giv", form="full", sep="; ", last="; "), rng)
                B.o(" (eds.). ")
            add_container(B, rec, rng)
            B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(". ")
        if rec.get("pages"):
            B.o("pp. ")
            B.f(page_text(rec, rng), "PAGES")
            B.o(". ")
        if rng.random() < 0.5:
            B.o(f"ISBN 978-{rng.randint(0, 9)}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}-{rng.randint(0, 9)}.")
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(". ")
        add_url(B, rec, rng)
        B.o(". ")
        add_accessed(B, rng, "viewed")
    else:
        style_vancouver_tail(B, rec, rng)


def style_cse(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="van", sep=", ", last=None, etal_after=10, etal_show=10)
    t = rec["type"]
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(". ")
    add_year(B, rec, rng, rng.choice(["year", "ymd"]))
    B.o(". ")
    add_title(B, rec, rng, case="sentence", end=".")
    B.o(" ")
    if t == "article":
        add_container(B, rec, rng, abbrev_p=0.5)
        B.o(". ")
        if rec.get("volume"):
            add_volume_issue(B, rec, rng, "paren")
        if rec.get("pages"):
            B.o(":")
            add_pages(B, rec, rng)
        B.o(".")
        if rec.get("doi") and rng.random() < 0.5:
            B.o(" ")
            add_doi(B, rec, rng)
    elif t == "book":
        if rec.get("edition"):
            B.f(rec["edition"], "EDITION")
            B.o(" ed. ")
        add_pub(B, rec, rng, "loc_colon_pub")
        B.o(".")
    elif t == "chapter":
        B.o("In: ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="van", sep=", ", last=None), rng)
            B.o(", editors. ")
        add_container(B, rec, rng)
        B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        if rec.get("pages"):
            B.o(". p. ")
            B.f(page_text(rec, rng), "PAGES")
        B.o(".")
    else:
        style_vancouver_tail(B, rec, rng)


def style_physics(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    """Terse AIP/APS/JHEP style: A. Author et al., Phys. Rev. D 60, 082002 (1999)."""
    ns = NameStyle(mode="giv_fam", form="initials", ini_style=rng.choice(["dot_space", "dot", "space"]), sep=", ", last=rng.choice([", and ", " and ", ", "]),
                   two=" and ", etal_after=rng.choice([1, 2, 3]), etal_show=1, etal=rng.choice([" et al.", " et al", " et al.,"]))
    t = rec["type"]
    if rng.random() < 0.5:
        B.o(f"[{rng.randint(1, 80)}] ")
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(rng.choice([", ", " ", ", "]))
    if t == "article":
        if rng.random() < 0.3:
            add_title(B, rec, rng, case="keep", end=",")
            B.o(" ")
        add_container(B, rec, rng, abbrev_p=0.85)
        B.o(" ")
        if rec.get("volume"):
            B.f(rec["volume"], "VOLUME")
            B.o(rng.choice([", ", " (", ", "]))
            if B.text().endswith("("):
                add_year(B, rec, rng)
                B.o(") ")
                if rec.get("pages"):
                    add_pages(B, rec, rng)
                B.o(rng.choice([".", ";", ""]))
                return
        if rec.get("pages"):
            add_pages(B, rec, rng)
            B.o(" ")
        B.o("(")
        add_year(B, rec, rng)
        B.o(")")
        B.o(rng.choice([".", ";", "", "."]))
        if rec.get("arxiv") and rng.random() < 0.5:
            B.o(rng.choice([" [", ", "]))
            add_arxiv(B, rec, rng, "prefix")
            B.o("]" if B.text().count("[") > B.text().count("]") else "")
    elif t == "preprint":
        if rng.random() < 0.5:
            add_title(B, rec, rng, case="keep", end=",")
            B.o(" ")
        add_arxiv(B, rec, rng, rng.choice(["prefix", "bare", "eprint"]))
        B.o(rng.choice([" (", ", "]))
        add_year(B, rec, rng)
        B.o(")." if B.text().rstrip("0123456789").endswith("(") else ".")
    elif t in ("conference", "chapter"):
        if rng.random() < 0.5:
            add_title(B, rec, rng, case="keep", end=",")
            B.o(" ")
        B.o(rng.choice(["in ", "in ", "In: "]))
        add_container(B, rec, rng, abbrev_p=0.5)
        if rec.get("editors"):
            B.o(", edited by ")
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="giv_fam", form="initials", last=" and "), rng)
        B.o(" (")
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(")")
        if rec.get("pages"):
            B.o(", ")
            add_pages(B, rec, rng, prefix=rng.choice(["p. ", "pp. "]))
        B.o(".")
    elif t == "book":
        add_title(B, rec, rng, case="keep")
        B.o(" (")
        add_pub(B, rec, rng, "pub_comma_loc")
        B.o(", ")
        add_year(B, rec, rng)
        B.o(").")
    else:
        style_vancouver_tail(B, rec, rng)


def style_german(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    ns = NameStyle(mode="fam_giv", rest_mode=rng.choice(["fam_giv", "giv_fam"]), form=rng.choice(["full", "initials"]), sep="; ", last=rng.choice([" und ", "; ", " & "]),
                   etal_after=3, etal_show=1, etal=" u. a.")
    t = rec["type"]
    if rec.get("authors"):
        add_people(B, rec["authors"], "AUTHOR", ns, rng)
        B.o(rng.choice([" (", ": "]))
        if B.text().endswith("("):
            add_year(B, rec, rng)
            B.o("): ")
    add_title(B, rec, rng, case="keep", end=rng.choice([".", ","]))
    B.o(" ")
    if t == "article":
        B.o("In: ")
        add_container(B, rec, rng)
        B.o(" ")
        add_volume_issue(B, rec, rng, rng.choice(["bd", "paren", "bare"]))
        if not B.text().rstrip().endswith(")") and not B.text().endswith("): "):
            B.o(" (")
            add_year(B, rec, rng)
            B.o(")")
        if rec.get("pages"):
            B.o(", S. ")
            B.f(page_text(rec, rng), "PAGES")
        B.o(".")
    elif t in ("chapter", "conference"):
        B.o("In: ")
        if rec.get("editors"):
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode="fam_giv", form="full", sep="; ", last="; "), rng)
            B.o(" (Hrsg.): ")
        add_container(B, rec, rng)
        B.o(". ")
        add_pub(B, rec, rng, "loc_colon_pub")
        if not B.text().endswith("): "):
            B.o(" ")
            add_year(B, rec, rng)
        if rec.get("pages"):
            B.o(", S. ")
            B.f(page_text(rec, rng), "PAGES")
        B.o(".")
    elif t == "book":
        add_edition(B, rec, rng, "german")
        B.o(" ")
        add_pub(B, rec, rng, "loc_colon_pub")
        if not B.text().endswith("): "):
            B.o(" ")
            add_year(B, rec, rng)
        B.o(".")
    elif t == "thesis":
        B.o(rng.choice(["Dissertation, ", "Diss., ", "Masterarbeit, ", "Diplomarbeit, "]))
        B.f(rec["publisher"], "PUBLISHER")
        B.o(" ")
        add_year(B, rec, rng)
        B.o(".")
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(", ")
        B.o("URL: ")
        add_url(B, rec, rng)
        B.o(" (abgerufen am ")
        B.f(access_date(rng, "iso"), "ACCESSED")
        B.o(").")
    else:
        style_vancouver_tail(B, rec, rng)


def style_messy(B: Builder, rec: dict[str, Any], rng: Rng) -> None:
    """Copy-paste soup: minimal punctuation, odd orders, extra junk."""
    ns = NameStyle(mode=rng.choice(["fam_giv", "giv_fam", "van", "fam_giv_nocomma", "caps"]), form=rng.choice(["full", "initials", "first_full"]),
                   ini_style=rng.choice(["dot", "dot_space", "plain", "space"]), sep=rng.choice([", ", "; ", " / ", ",", " , "]),
                   last=rng.choice([" and ", " & ", ", ", " y ", " et ", None]), etal_after=rng.choice([2, 3, 5, 30]), etal_show=1,
                   etal=rng.choice([" et al", " et al.", " and others", " et al.,", " ..."]))
    t = rec["type"]
    order = rng.choice(["ayt", "aty", "tay", "yat"])
    sep = rng.choice([". ", ", ", " ", "; ", " - ", " | "])
    if rng.random() < 0.3:
        B.o(rng.choice(["- ", "* ", "• ", f"{rng.randint(1, 30)} ", f"[{rng.randint(1, 30)}]", "> "]))
    parts = {
        "a": lambda: add_people(B, rec["authors"], "AUTHOR", ns, rng) if rec.get("authors") else None,
        "y": lambda: (B.o("("), add_year(B, rec, rng, rng.choice(["year", "month_year"])), B.o(")")) if rng.random() < 0.5 else add_year(B, rec, rng),
        "t": lambda: add_title(B, rec, rng, quote=rng.choice(["", "", '"', "“", "'"]), case=rng.choice(["keep", "sentence", "title", "upper"])),
    }
    for i, key in enumerate(order):
        if i:
            B.o(sep)
        parts[key]()
    B.o(sep)
    if t == "article":
        add_container(B, rec, rng, abbrev_p=0.4, markers=True)
        if rec.get("volume"):
            B.o(rng.choice([" ", ", ", sep]))
            add_volume_issue(B, rec, rng, rng.choice(["paren", "vol_no", "bare", "colon"]))
        if rec.get("pages"):
            B.o(rng.choice([", ", ": ", " ", ", pp. ", " p. ", sep]))
            add_pages(B, rec, rng)
    elif t in ("chapter", "conference"):
        B.o(rng.choice(["In ", "in ", "In: ", ""]))
        add_container(B, rec, rng, abbrev_p=0.3)
        if rec.get("editors") and rng.random() < 0.6:
            B.o(rng.choice([", ed. ", " (eds) ", ", edited by ", " eds. "]))
            add_people(B, rec["editors"], "EDITOR", NameStyle(mode=rng.choice(["fam_giv", "giv_fam"]), form="initials", last=" and "), rng)
        if rec.get("pages"):
            B.o(rng.choice([", ", " ", ", pp. ", sep]))
            add_pages(B, rec, rng)
        if rec.get("publisher") or rec.get("location"):
            B.o(sep)
            add_pub(B, rec, rng, rng.choice(["loc_colon_pub", "pub_comma_loc", "pub_only"]))
    elif t == "book":
        add_edition(B, rec, rng, rng.choice(["paren", "plain"]))
        B.o(sep)
        add_pub(B, rec, rng, rng.choice(["loc_colon_pub", "pub_comma_loc", "pub_only"]))
    elif t == "thesis":
        B.o(genre_text(rec, rng) + sep)
        B.f(rec["publisher"], "PUBLISHER")
    elif t == "report":
        B.o(genre_text(rec, rng) + sep)
        add_pub(B, rec, rng, rng.choice(["loc_colon_pub", "pub_comma_loc", "pub_only"]))
    elif t == "web":
        if rec.get("container"):
            add_container(B, rec, rng)
            B.o(sep)
        add_url(B, rec, rng)
        if rng.random() < 0.5:
            B.o(" ")
            add_accessed(B, rng, rng.choice(["viewed", "harvard", "ieee", "apa6"]))
    elif t == "preprint":
        if rec.get("arxiv"):
            add_arxiv(B, rec, rng, rng.choice(["prefix", "url", "bare", "bracket"]))
        else:
            add_doi(B, rec, rng)
    if rec.get("doi") and t != "preprint" and rng.random() < 0.5:
        B.o(rng.choice([sep, " ", ". "]))
        add_doi(B, rec, rng)
    if rec.get("url") and t != "web" and rng.random() < 0.3:
        B.o(rng.choice([" ", sep, " Available at "]))
        add_url(B, rec, rng)
    if rng.random() < 0.15:
        B.o(rng.choice([" [CrossRef]", " [PubMed]", " Google Scholar", " [Google Scholar] [CrossRef]", " (in press)", " (forthcoming)", " PMID: 12345678", " (PDF)", " ↗"]))
    B.o(rng.choice([".", "", "", " ."]))


STYLES: list[tuple[str, Any, float]] = [
    ("apa", style_apa, 13),
    ("mla", style_mla, 6),
    ("chicago_ad", style_chicago_ad, 6),
    ("chicago_note", style_chicago_note, 4),
    ("ieee", style_ieee, 9),
    ("vancouver", style_vancouver, 8),
    ("harvard", style_harvard, 8),
    ("nature", style_nature, 6),
    ("acm", style_acm, 5),
    ("ama", style_ama, 4),
    ("elsevier", style_elsevier, 5),
    ("springer", style_springer, 5),
    ("plain", style_plain, 6),
    ("arxiv_listing", style_arxiv_listing, 4),
    ("wikipedia", style_wikipedia, 3),
    ("cse", style_cse, 3),
    ("physics", style_physics, 6),
    ("german", style_german, 3),
    ("messy", style_messy, 8),
]

# ------------------------------------------------------------------- realization


def _slug(title: str, rng: Rng) -> str:
    words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in title.split()[:6]]
    words = [w for w in words if w]
    return rng.choice(["-", "_", ""]).join(words) or "page"


def _clean_publisher(name: str) -> str:
    name = re.sub(r"\s*\(.*?\)\s*", " ", name)
    name = re.sub(r"\b(BV|B\.V\.|LLC|Inc\.?|Ltd\.?|GmbH|LLP|plc|AG|SA|S\.A\.)\b\.?", "", name)
    return re.sub(r"\s+", " ", name).strip(" ,")


class Pools:
    def __init__(self, records: list[dict[str, Any]], rng: Rng):
        self.given: list[str] = []
        self.family: list[str] = []
        self.containers: list[str] = []
        for r in records:
            for a in r["authors"]:
                if a.get("given"):
                    self.given.append(a["given"])
                self.family.append(a["family"])
            if r.get("container") and r["type"] == "article":
                self.containers.append(r["container"])
        rng.shuffle(self.given)
        rng.shuffle(self.family)

    def person(self, rng: Rng) -> dict[str, str]:
        return {"given": rng.choice(self.given), "family": rng.choice(self.family)}


def realize(rec: dict[str, Any], rng: Rng, pools: Pools) -> dict[str, Any] | None:
    """Turn a source record into a fully specified record of a (possibly converted) type."""
    r: dict[str, Any] = dict(rec)
    r["authors"] = [dict(a) for a in rec["authors"]]
    t = r["type"]
    # Rebalance types: many CrossRef articles become other genres with synthesized details.
    if t == "article" and rng.random() < 0.30:
        t = rng.choices(["book", "thesis", "report", "web", "chapter", "conference"], weights=[8, 5, 5, 7, 3, 3])[0]
    if t == "preprint" and rng.random() < 0.15 and r.get("doi"):
        t = "article"
        r["container"] = rng.choice(pools.containers)
        r["volume"] = str(rng.randint(1, 250))
        r["issue"] = str(rng.randint(1, 12))
        a = rng.randint(1, 900)
        r["pages"] = {"from": str(a), "to": str(a + rng.randint(1, 30))}
    r["type"] = t
    if t == "article":
        if not r.get("container"):
            return None
        if rng.random() < 0.1:
            r.pop("issue", None)
        if rng.random() < 0.1:
            r.pop("pages", None)
        if rng.random() < 0.3:
            r.pop("doi", None)
        if "arxiv" in r and rng.random() < 0.5:
            r.pop("arxiv", None)
    elif t == "book":
        for k in ("container", "container_short", "volume", "issue", "pages", "event"):
            r.pop(k, None)
        r["publisher"] = _clean_publisher(r["publisher"]) if r.get("publisher") and rng.random() < 0.4 else rng.choice(PUBLISHERS)
        if rng.random() < 0.6:
            r["location"] = rng.choice(CITIES)
        else:
            r.pop("location", None)
        if rng.random() < 0.25:
            r["edition"] = rng.choice(ORDINALS[:6] + ORDINAL_WORDS[:5] + ["Rev.", "2", "3"])
        if rng.random() < 0.6:
            r.pop("doi", None)
    elif t == "chapter":
        if not r.get("container"):
            r["container"] = rng.choice([x for x in pools.containers if len(x) > 12])
        for k in ("volume", "issue", "container_short"):
            r.pop(k, None) if rng.random() < 0.7 else None
        if not r.get("editors") or rng.random() < 0.2:
            r["editors"] = [pools.person(rng) for _ in range(rng.choice([1, 1, 2, 2, 3]))]
        if rng.random() < 0.3:
            r.pop("editors", None)
        r["publisher"] = _clean_publisher(r["publisher"]) if r.get("publisher") and rng.random() < 0.5 else rng.choice(PUBLISHERS)
        if rng.random() < 0.5:
            r["location"] = rng.choice(CITIES)
        else:
            r.pop("location", None)
        if not r.get("pages") and rng.random() < 0.7:
            a = rng.randint(1, 600)
            r["pages"] = {"from": str(a), "to": str(a + rng.randint(3, 40))}
        if rng.random() < 0.5:
            r.pop("doi", None)
    elif t == "conference":
        if not r.get("container"):
            n = rng.choice(ORDINALS)
            r["container"] = rng.choice([
                f"Proceedings of the {n} International Conference on {title_case(rec['title'].split(':')[0][:40])}",
                f"Proc. {n} Int. Conf. on {title_case(rec['title'].split(':')[0][:30])}",
                f"{n} Annual Meeting of the Association for {title_case(rec['title'].split(':')[0][:30])}",
                f"Proceedings of {rng.choice(['ICML', 'NeurIPS', 'CVPR', 'ACL', 'CHI', 'SIGIR', 'KDD', 'ICSE', 'AAAI', 'ISCA', 'ICRA', 'IROS'])} {rec['year']}",
                f"Advances in Neural Information Processing Systems {rng.randint(10, 37)}",
                f"{rng.choice(['IEEE', 'ACM', 'International'])} {rng.choice(['Conference', 'Symposium', 'Workshop'])} on {title_case(rec['title'].split(':')[0][:35])}",
            ])
        for k in ("volume", "issue", "container_short"):
            r.pop(k, None) if rng.random() < 0.8 else None
        if rng.random() < 0.5:
            r["location"] = rng.choice(CITIES)
        else:
            r.pop("location", None)
        if rng.random() < 0.4:
            r["publisher"] = rng.choice(["IEEE", "ACM", "Springer", "AAAI Press", "PMLR", "Curran Associates", "Association for Computational Linguistics", "IEEE Computer Society", "ACM Press", "Elsevier", "USENIX Association", "SIAM", "MIT Press", "Morgan Kaufmann"])
        else:
            r.pop("publisher", None)
        if not r.get("pages") and rng.random() < 0.6:
            a = rng.randint(1, 3000)
            r["pages"] = {"from": str(a), "to": str(a + rng.randint(3, 15))}
        if rng.random() < 0.5:
            r.pop("doi", None)
        if rng.random() < 0.3:
            r["event_short"] = rng.choice(["ICML", "NeurIPS", "CVPR", "ACL", "CHI", "SIGIR", "KDD", "ICSE", "WWW", "SIGMOD"])
    elif t == "thesis":
        for k in ("container", "container_short", "volume", "issue", "pages", "editors", "event"):
            r.pop(k, None)
        r["authors"] = r["authors"][:1]
        r["publisher"] = rng.choice(UNIVERSITIES)
        if rng.random() < 0.4:
            r["location"] = rng.choice(CITIES)
        else:
            r.pop("location", None)
        r.pop("doi", None) if rng.random() < 0.7 else None
        if rng.random() < 0.3:
            r["url"] = f"https://{rng.choice(['hdl.handle.net', 'dspace.mit.edu', 'ethos.bl.uk', 'theses.fr', 'repository.example.edu'])}/{rng.randint(1000, 99999)}/{rng.randint(100, 99999)}"
    elif t == "report":
        for k in ("container", "container_short", "volume", "issue", "pages", "editors", "event"):
            r.pop(k, None)
        if rng.random() < 0.25:
            r["authors"] = [{"literal": rng.choice(ORGS)}]
        r["publisher"] = rng.choice(INSTITUTIONS + UNIVERSITIES)
        if rng.random() < 0.4:
            r["location"] = rng.choice(CITIES)
        else:
            r.pop("location", None)
        r["number"] = rng.choice([f"{rng.randint(1, 999)}", f"TR-{rng.randint(1, 999)}", f"{rng.randint(2000, 2025)}-{rng.randint(1, 99):02d}", f"No. {rng.randint(1, 30000)}", f"w{rng.randint(1000, 32000)}", f"RFC {rng.randint(100, 9999)}", f"CS-{rng.randint(80, 99)}-{rng.randint(1, 99)}"])
        r.pop("doi", None) if rng.random() < 0.6 else None
        if rng.random() < 0.4:
            r["url"] = f"https://www.{rng.choice(['nber.org/papers', 'rand.org/pubs', 'oecd.org/reports', 'who.int/publications', 'ietf.org/rfc', 'brookings.edu/research'])}/{_slug(r['title'], rng)}.{rng.choice(['pdf', 'html', ''])}".rstrip(".")
    elif t == "web":
        for k in ("volume", "issue", "pages", "editors", "event", "container_short", "publisher", "location"):
            r.pop(k, None)
        site, domain = rng.choice(SITES)
        r["container"] = site if rng.random() < 0.8 else domain
        if rng.random() < 0.35:
            r["authors"] = [{"literal": rng.choice(ORGS + [site])}]
        elif rng.random() < 0.15:
            r["authors"] = []
        else:
            r["authors"] = r["authors"][: rng.choice([1, 1, 2])]
        path = _slug(r["title"], rng)
        r["url"] = rng.choice([
            f"https://{domain}/{path}",
            f"https://www.{domain}/{rng.randint(2010, 2026)}/{rng.randint(1, 12):02d}/{path}.html",
            f"http://{domain}/{path}?id={rng.randint(100, 99999)}",
            f"https://{domain}/wiki/{path}",
            f"https://{domain}/articles/{rng.randint(1000, 9999999)}",
            f"www.{domain}/{path}",
        ])
        r["month"] = rng.randint(1, 12)
        r.pop("doi", None)
        if rng.random() < 0.15:
            r.pop("year", None)
    elif t == "preprint":
        for k in ("container", "container_short", "volume", "issue", "pages", "editors", "publisher", "location"):
            r.pop(k, None)
        if not r.get("arxiv"):
            if rng.random() < 0.5:
                r["arxiv"] = f"{rng.randint(7, 25):02d}{rng.randint(1, 12):02d}.{rng.randint(0, 99999):05d}" + (f"v{rng.randint(1, 4)}" if rng.random() < 0.3 else "")
                r.pop("doi", None) if rng.random() < 0.7 else None
        elif rng.random() < 0.3:
            r["arxiv"] = r["arxiv"] + f"v{rng.randint(1, 5)}"
        if r.get("arxiv") and rng.random() < 0.15:
            cat = rng.choice(["hep-th", "hep-ph", "astro-ph", "cond-mat", "quant-ph", "math", "cs", "gr-qc"])
            r["arxiv"] = f"{cat}/{rng.randint(92, 99) if rng.random() < 0.5 else rng.randint(1, 7):02d}{rng.randint(1, 12):02d}{rng.randint(0, 999):03d}"
        if not r.get("arxiv"):
            if not r.get("doi"):
                return None
    return r


# ------------------------------------------------------------------- labelling


def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def label_pieces(pieces: list[Piece]) -> dict[str, Any]:
    """Assemble pieces into text and BIO / namepart labels per token."""
    text = "".join(p.text for p in pieces)
    # char (UTF-16) -> piece index
    owner: list[int] = []
    for i, p in enumerate(pieces):
        owner.extend([i] * _utf16_len(p.text))
    tokens = tokenize(text)
    tags: list[int] = []
    parts: list[int] = []
    prev_ent = -1
    prev_role: str | None = None
    for tok in tokens:
        # majority owner over the token's chars
        counts = Counter(owner[tok.start : tok.end])
        pi = counts.most_common(1)[0][0]
        p = pieces[pi]
        if p.role is None or tok.text.isspace() and (prev_role != p.role or prev_ent != p.ent) and False:
            tags.append(0)
            parts.append(0)
            prev_ent, prev_role = -1, None
            continue
        if p.role == prev_role and p.ent == prev_ent:
            tags.append(TAG_INDEX[f"I-{p.role}"])
        else:
            tags.append(TAG_INDEX[f"B-{p.role}"])
        parts.append(p.part if p.role in ("AUTHOR", "EDITOR") else 0)
        prev_ent, prev_role = p.ent, p.role
    return {"text": text, "tags": tags, "parts": parts}


def _apply_noise(pieces: list[Piece], rng: Rng, level: float) -> list[Piece]:
    out: list[Piece] = []
    for p in pieces:
        text = p.text
        if p.role in ("TITLE", "CONTAINER") and level > 0:
            text = " ".join(maybe_typo(w, rng, 0.03 * level) for w in text.split(" "))
        if p.role is None and level > 0:
            r = rng.random()
            if r < 0.06 * level and text.strip() in (",", ".", ";", ":"):
                text = text.strip()  # drop the space after punctuation
            elif r < 0.10 * level and text == " ":
                text = "  "
            elif r < 0.13 * level and text.strip() in (",", ".") and len(text) > 1:
                text = text[1:]  # drop the punctuation, keep the space
        out.append(Piece(text, p.role, p.ent, p.part))
    return out


def render(rec: dict[str, Any], rng: Rng, style_name: str | None = None) -> dict[str, Any]:
    names = [s[0] for s in STYLES]
    if style_name is None:
        style_name = rng.choices(names, weights=[s[2] for s in STYLES])[0]
    fn = next(s[1] for s in STYLES if s[0] == style_name)
    B = Builder()
    fn(B, rec, rng)
    pieces = B.pieces
    noise = 0.0 if rng.random() < 0.5 else rng.choice([0.5, 1.0, 2.0])
    pieces = _apply_noise(pieces, rng, noise)
    # Global casing / whitespace noise (does not move spans relative to pieces).
    if rng.random() < 0.02:
        pieces = [Piece(p.text.lower(), p.role, p.ent, p.part) for p in pieces]
    if rng.random() < 0.1:
        pieces = [Piece(rng.choice(["  ", "\t", " "]))] + pieces
    if rng.random() < 0.1:
        pieces = pieces + [Piece(rng.choice(["  ", " ", "\t"]))]
    ex = label_pieces(pieces)
    ex["type"] = TYPE_INDEX[rec["type"]]
    ex["style"] = style_name
    ex["etal"] = " et al" in ex["text"] or " u. a." in ex["text"] or " and others" in ex["text"]
    return ex


# --------------------------------------------------------------------- dataset


def build(n_train: int = 120_000, n_heldout: int = 4000, seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    records = load_metadata(seed)
    if not records:
        raise SystemExit("no metadata in cache; run `python -m gpu_cite.sources` first")
    pools = Pools(records, rng)
    # Held-out records are disjoint from training records at the *metadata* level.
    n_hold_rec = max(200, len(records) // 10)
    hold_recs, train_recs = records[:n_hold_rec], records[n_hold_rec:]

    def make(recs: list[dict[str, Any]], n: int, r: Rng) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        attempts = 0
        while len(out) < n and attempts < n * 3:
            attempts += 1
            rec = realize(r.choice(recs), r, pools)
            if rec is None:
                continue
            ex = render(rec, r)
            if len(ex["tags"]) > 240 or len(ex["text"]) < 15:
                continue
            out.append(ex)
        return out

    train = make(train_recs, n_train, random.Random(seed + 1))
    heldout = make(hold_recs, n_heldout, random.Random(seed + 2))
    return train, heldout


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=120_000)
    ap.add_argument("--heldout", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()
    train, heldout = build(args.train, args.heldout, args.seed)
    write_jsonl(CACHE / "train.jsonl.gz", train)
    write_jsonl(CACHE / "heldout.jsonl.gz", heldout)
    print(f"train {len(train)}  heldout {len(heldout)}")
    print("types:", Counter(TYPES[e["type"]] for e in train).most_common())
    print("styles:", Counter(e["style"] for e in train).most_common())
    rng = random.Random(args.seed)
    for ex in rng.sample(train, args.show):
        print(f"[{ex['style']}/{TYPES[ex['type']]}] {ex['text']}")
    # role coverage
    cov = Counter()
    for ex in train:
        for tag in set(ex["tags"]):
            if tag and tag <= len(ROLES):
                cov[ROLES[tag - 1]] += 1
    print("role coverage:", {k: round(v / len(train), 3) for k, v in cov.most_common()})
    print("nameparts:", Counter(NAMEPARTS[p] for ex in train[:5000] for p in ex["parts"]).most_common())


if __name__ == "__main__":
    main()
