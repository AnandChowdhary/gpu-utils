"""Domain field pools and random schema sampling.

A schema is `{ fields: [{ name, kind, aliases?, values? }] }`, exactly what the
TypeScript API takes. Training samples random subsets of each training domain's
pool, so the model sees hundreds of distinct schemas; four whole domains are held
out for the transfer evaluation and share no field word, alias or enum value with
training (enforced by `assert_disjoint`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

TEXT, NUMBER, DATE, ENUM, BOOL = "text", "number", "date", "enum", "boolean"
KINDS = [TEXT, NUMBER, DATE, ENUM, BOOL]


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    aliases: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    person: bool = False  # text field that can take "me" as a value
    adjectives: tuple[str, ...] = ()  # polarity adjectives usable as aliases ("cheap", "tall")
    year_like: bool = False  # number field whose values look like years (vintage, founded)
    unit: str = ""  # unit word that may follow a value ("cm", "kg")
    primary: bool = False  # date field used for bare time phrases

    def to_json(self) -> dict:
        out: dict = {"name": self.name, "kind": self.kind}
        aliases = list(self.aliases) + list(self.adjectives)
        if aliases:
            out["aliases"] = aliases
        if self.values:
            out["values"] = list(self.values)
        if self.primary:
            out["primary"] = True
        return out


@dataclass
class Schema:
    domain: str
    entity: str  # plural noun used as carrier prose ("orders")
    fields: list[Field] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"fields": [f.to_json() for f in self.fields]}

    def by_name(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def of_kind(self, kind: str) -> list[Field]:
        return [f for f in self.fields if f.kind == kind]


def F(name: str, kind: str, aliases: str = "", values: str = "", person: bool = False,
      adj: str = "", year_like: bool = False, unit: str = "") -> Field:
    return Field(
        name,
        kind,
        tuple(a.strip() for a in aliases.split("|") if a.strip()),
        tuple(v.strip() for v in values.split("|") if v.strip()),
        person,
        tuple(a.strip() for a in adj.split("|") if a.strip()),
        year_like,
        unit,
    )


# ---------------------------------------------------------------- training domains
POOLS: dict[str, tuple[list[str], list[Field]]] = {
    "ecommerce": (
        ["orders", "purchases", "carts"],
        [
            F("customer", TEXT, "buyer|shopper|client", person=True),
            F("total", NUMBER, "amount|order value|order total|spend", adj="expensive|cheap|pricey|big", unit="dollars"),
            F("status", ENUM, "state", "pending|paid|shipped|delivered|cancelled|refunded"),
            F("country", ENUM, "nation|ship to", "Germany|France|Spain|Italy|Japan|Brazil|Canada|Mexico"),
            F("created_at", DATE, "created|order date|placed|ordered"),
            F("items", NUMBER, "item count|quantity|qty", unit="units"),
            F("discount", NUMBER, "coupon|promo|markdown"),
            F("gift", BOOL, "is gift|gift wrapped"),
            F("payment_method", ENUM, "payment|paid with", "card|paypal|bank transfer|cash on delivery|crypto"),
            F("category", ENUM, "department|product type", "electronics|clothing|books|toys|home|garden|sports"),
            F("shipping_cost", NUMBER, "shipping|postage|delivery fee"),
            F("rating", NUMBER, "review score|review|rated", adj="good|popular", unit="stars"),
            F("weight_kg", NUMBER, "package weight|heaviness", adj="heavy|light", unit="kg"),
            F("sku", TEXT, "product code|part number"),
            F("returned", BOOL, "return|sent back"),
        ],
    ),
    "crm": (
        ["deals", "contacts", "leads", "accounts"],
        [
            F("company", TEXT, "organisation|organization|firm"),
            F("owner", TEXT, "rep|account manager|sales rep|salesperson", person=True),
            F("stage", ENUM, "pipeline stage|phase", "lead|qualified|proposal|negotiation|won|lost"),
            F("value", NUMBER, "deal size|deal value|arr|revenue", adj="big|large|small|rich", unit="dollars"),
            F("founded", NUMBER, "founded in|established|est", year_like=True),
            F("industry", ENUM, "sector|vertical", "software|healthcare|retail|banking|manufacturing|media|insurance"),
            F("region", ENUM, "territory|geo", "emea|apac|americas|latam|nordics|dach|benelux"),
            F("last_contacted", DATE, "last contact|contacted|last touch"),
            F("email", TEXT, "mail|email address"),
            F("phone", TEXT, "telephone|mobile number"),
            F("score", NUMBER, "lead score|fit"),
            F("closed_at", DATE, "close date|closed|closing"),
            F("newsletter", BOOL, "subscribed|opted in"),
            F("source", ENUM, "channel|lead source", "referral|webinar|ads|organic|conference|partner|cold call"),
            F("employees", NUMBER, "headcount|staff count", unit="people"),
            F("website", TEXT, "url|domain"),
        ],
    ),
    "issues": (
        ["issues", "tickets", "bugs", "tasks"],
        [
            F("title", TEXT, "summary|heading|headline"),
            F("status", ENUM, "state|workflow state", "open|in progress|closed|resolved|blocked|todo|done|backlog"),
            F("priority", ENUM, "severity|urgency|prio", "low|medium|high|urgent|critical"),
            F("assignee", TEXT, "assigned|assigned to|owner|assignee name", person=True),
            F("reporter", TEXT, "author|created by|reported by|creator", person=True),
            F("labels", ENUM, "label|tag|tags", "bug|feature|docs|question|chore|security|regression"),
            F("created", DATE, "opened|created at|filed"),
            F("updated", DATE, "updated at|modified|last updated|last activity"),
            F("due", DATE, "due date|deadline|target date"),
            F("estimate", NUMBER, "points|story points|effort|size", adj="big|small", unit="pts"),
            F("comments", NUMBER, "comment count|replies|discussion"),
            F("project", ENUM, "board|repo|repository", "web|mobile|api|infra|design|billing"),
            F("resolved", BOOL, "fixed|is resolved"),
            F("sprint", TEXT, "iteration|cycle"),
            F("watchers", NUMBER, "subscribers|followers"),
        ],
    ),
    "music": (
        ["tracks", "songs", "albums", "recordings"],
        [
            F("artist", TEXT, "performer|band|musician|singer"),
            F("album", TEXT, "record|release title"),
            F("genre", ENUM, "style", "rock|jazz|pop|hip hop|classical|electronic|metal|folk|reggae|blues"),
            F("duration", NUMBER, "length|runtime", adj="long|short", unit="minutes"),
            F("recorded_in", NUMBER, "recorded|recording", year_like=True),
            F("plays", NUMBER, "play count|streams|listens|spins|played|streamed", adj="popular"),
            F("released", DATE, "release date|release|dropped|published"),
            F("explicit", BOOL, "parental advisory|is explicit"),
            F("label", TEXT, "record label|imprint"),
            F("bpm", NUMBER, "tempo|beats per minute", adj="fast|slow"),
            F("mood", ENUM, "vibe|feel", "happy|sad|energetic|chill|dark|romantic|angry"),
            F("format", ENUM, "medium", "vinyl|cd|digital|cassette|streaming"),
            F("favorite", BOOL, "liked|starred|favourite|loved"),
            F("key", ENUM, "musical key", "c major|a minor|g major|e minor|d major|f major"),
            F("producer", TEXT, "produced by|engineer", person=True),
        ],
    ),
    "hr": (
        ["employees", "staff", "people", "hires"],
        [
            F("department", ENUM, "dept|org|function", "engineering|marketing|sales|finance|support|legal|operations|people ops"),
            F("manager", TEXT, "reports to|boss|supervisor|lead", person=True),
            F("salary", NUMBER, "pay|compensation|comp|base|paid", adj="rich|poor|expensive|cheap", unit="dollars"),
            F("hired", DATE, "hire date|start date|joined|joining date"),
            F("location", ENUM, "office|site|based in", "london|berlin|new york|remote|austin|toronto|singapore|dublin"),
            F("level", ENUM, "grade|band|seniority", "junior|mid|senior|staff|principal|director|vp"),
            F("remote", BOOL, "works remotely|wfh|remote worker"),
            F("team", TEXT, "squad|pod|group name"),
            F("age", NUMBER, "years old", adj="old|young", unit="years"),
            F("yob", NUMBER, "birthyear|birth yr", year_like=True),
            F("performance", NUMBER, "perf|review rating|performance score"),
            F("contract", ENUM, "employment type|worker type", "full time|part time|contractor|intern|temp"),
            F("active", BOOL, "employed|current|still here"),
            F("birthday", DATE, "date of birth|dob|born"),
            F("vacation_days", NUMBER, "pto|leave balance|holidays left"),
        ],
    ),
    "logistics": (
        ["shipments", "packages", "parcels", "deliveries"],
        [
            F("carrier", ENUM, "courier|shipper", "ups|fedex|dhl|usps|maersk|tnt|royal mail"),
            F("weight", NUMBER, "mass|weighs", adj="heavy|light", unit="kg"),
            F("origin", ENUM, "from warehouse|depot|hub", "hamburg|rotterdam|shanghai|chicago|memphis|leipzig|dubai"),
            F("destination", TEXT, "dest|going to|address"),
            F("shipped", DATE, "ship date|dispatched|sent|dispatch date"),
            F("delivered_at", DATE, "delivery date|arrived|received"),
            F("tracking_state", ENUM, "tracking|progress", "in transit|delayed|out for delivery|lost|customs|held|returned"),
            F("cost", NUMBER, "freight|freight cost|price"),
            F("fragile", BOOL, "breakable|handle with care"),
            F("pallets", NUMBER, "pallet count|skids"),
            F("route", TEXT, "lane|leg"),
            F("driver", TEXT, "trucker|courier name|delivered by", person=True),
            F("eta", DATE, "expected|arrival|expected delivery|arriving"),
            F("distance", NUMBER, "mileage|how far", adj="far|near|long|short", unit="km"),
            F("height_cm", NUMBER, "parcel height|how tall", adj="tall|short", unit="cm"),
            F("insured", BOOL, "insurance|covered"),
        ],
    ),
    "analytics": (
        ["sessions", "pageviews", "visits", "events"],
        [
            F("page", TEXT, "path|url|screen"),
            F("browser", ENUM, "user agent", "chrome|firefox|safari|edge|opera|brave"),
            F("device", ENUM, "device type|form factor", "desktop|mobile|tablet|tv|watch"),
            F("visitors", NUMBER, "users|uniques|unique visitors|people count", adj="busy|popular", unit="people"),
            F("views", NUMBER, "hits|impressions|page views"),
            F("bounce_rate", NUMBER, "bounce|bounces|bounce percentage"),
            F("session_length", NUMBER, "time on site|dwell time|engagement time", adj="long|short", unit="seconds"),
            F("referrer", TEXT, "referring site|came from|traffic source"),
            F("campaign", TEXT, "utm campaign|utm|promotion"),
            F("timestamp", DATE, "time|when|seen at|visited"),
            F("converted", BOOL, "conversion|did convert|purchased"),
            F("os", ENUM, "operating system|platform", "windows|macos|linux|ios|android|chromeos"),
            F("earnings", NUMBER, "ad revenue|monetization|income"),
            F("event_type", ENUM, "event name|action", "click|signup|purchase|pageview|download|share|login"),
            F("new_user", BOOL, "first visit|first time|new visitor"),
        ],
    ),
    "finance": (
        ["transactions", "invoices", "expenses", "payments"],
        [
            F("account", TEXT, "account name|ledger|wallet"),
            F("amount", NUMBER, "sum paid|net|gross|charge", adj="big|small|large|expensive", unit="dollars"),
            F("fy", NUMBER, "fiscal|financial period", year_like=True),
            F("currency", ENUM, "ccy|denomination", "usd|eur|gbp|jpy|chf|aud|inr"),
            F("type", ENUM, "transaction type|kind|entry type", "debit|credit|transfer|fee|dividend|interest|chargeback"),
            F("merchant", TEXT, "vendor|payee|supplier|counterparty"),
            F("booked", DATE, "posted|transaction date|settled|booked on"),
            F("balance", NUMBER, "running balance|remaining"),
            F("expense_category", ENUM, "cost center|budget line|spend category", "groceries|rent|utilities|salaries|marketing spend|software licenses|office"),
            F("approved", BOOL, "signed off|authorised|authorized"),
            F("invoice_number", TEXT, "invoice|reference|ref|memo"),
            F("tax", NUMBER, "vat|sales tax|gst"),
            F("due_on", DATE, "payment due|due by|pay by"),
            F("recurring", BOOL, "subscription|repeating|monthly charge"),
            F("risk", ENUM, "risk level|fraud risk", "safe|suspicious|flagged|blocked"),
            F("paid", BOOL, "settled up|cleared"),
        ],
    ),
}

# ------------------------------------------------------------ held-out domains
# Share no field word, alias word or enum value with the training pools.
EVAL_POOLS: dict[str, tuple[list[str], list[Field]]] = {
    "recipes": (
        ["recipes", "dishes", "meals"],
        [
            F("cuisine", ENUM, "kitchen|cooking tradition", "italian|mexican|thai|indian|greek|korean|ethiopian|peruvian"),
            F("difficulty", ENUM, "skill needed|complexity", "easy|moderate|hard|expert"),
            F("prep", NUMBER, "preparation|kitchen wait|cooking mins", adj="long|short|quick", unit="minutes"),
            F("calories", NUMBER, "kcal|energy content", adj="heavy|light", unit="kcal"),
            F("servings", NUMBER, "portions|serves|yield"),
            F("chef", TEXT, "cook|contributor", person=True),
            F("course", ENUM, "meal slot", "starter|main|dessert|snack|breakfast|brunch|appetizer"),
            F("vegetarian", BOOL, "veggie|meatless|herbivore friendly"),
            F("spicy", BOOL, "fiery|chili heavy"),
            F("ingredients", NUMBER, "ingredient tally|components"),
            F("added_on", DATE, "added|saved on|clipped"),
            F("stars", NUMBER, "cook thumbs|applause"),
            F("oven", BOOL, "baked|needs oven"),
        ],
    ),
    "realestate": (
        ["listings", "properties", "homes", "houses"],
        [
            F("bedrooms", NUMBER, "beds|br"),
            F("bathrooms", NUMBER, "baths|ba"),
            F("asking", NUMBER, "asking figure|sticker|listed for|priced", adj="expensive|cheap|pricey", unit="dollars"),
            F("sqft", NUMBER, "square feet|square footage|floor area|footage", adj="big|small|large", unit="sqft"),
            F("neighborhood", ENUM, "neighbourhood|district|area", "downtown|suburbs|waterfront|uptown|midtown|hills|old town"),
            F("listed_on", DATE, "listed|went live|hit the market|listing day"),
            F("dwelling", ENUM, "housing shape|structure", "condo|house|townhouse|apartment|loft|duplex|bungalow"),
            F("garage", BOOL, "parking|carport"),
            F("furnished", BOOL, "comes furnished|with furniture"),
            F("realtor", TEXT, "broker|listing person", person=True),
            F("year_built", NUMBER, "built|constructed|vintage", adj="old|new", year_like=True),
            F("pool", BOOL, "swimming pool|has pool"),
            F("sold", BOOL, "no longer available|already sold"),
            F("hoa", NUMBER, "association dues|community dues"),
            F("viewing_day", DATE, "viewing|showing|walkthrough"),
        ],
    ),
    "education": (
        ["students", "courses", "enrollments", "pupils"],
        [
            F("student", TEXT, "learner|pupil|enrollee", person=True),
            F("gpa", NUMBER, "mark|marks|scholastic result|graded", adj="good|bad|strong|weak"),
            F("subject", ENUM, "discipline|topic", "math|physics|chemistry|biology|history|literature|art|geography"),
            F("semester", ENUM, "term", "fall|spring|summer|winter"),
            F("enrolled", DATE, "enrollment|enrolled on|registered|matriculated"),
            F("instructor", TEXT, "teacher|professor|tutor|lecturer", person=True),
            F("credits", NUMBER, "ects|study load"),
            F("attendance", NUMBER, "presence|classes attended|turnout"),
            F("passed", BOOL, "pass|passing|aced the exam"),
            F("scholarship", BOOL, "funded|on scholarship|bursary"),
            F("campus", ENUM, "school building|which campus", "north campus|south campus|east campus|west campus|online"),
            F("homework", NUMBER, "assignments|exercises|problem sets"),
            F("graduation", DATE, "graduates|graduating|commencement"),
            F("tuition", NUMBER, "fees owed|tuition bill"),
        ],
    ),
    "travel": (
        ["trips", "bookings", "flights", "itineraries"],
        [
            F("city", ENUM, "town|place", "paris|rome|tokyo|lisbon|sydney|cairo|lima|oslo"),
            F("airline", ENUM, "flown with|operator", "lufthansa|delta|emirates|ryanair|qantas|klm|ana"),
            F("departure", DATE, "departs|leaving|depart|outbound"),
            F("inbound_flight", DATE, "returning|inbound|homebound"),
            F("nights", NUMBER, "overnights|nights away", adj="long|short", unit="nights"),
            F("fare", NUMBER, "airfare|ticket outlay", adj="expensive|cheap|pricey", unit="euros"),
            F("traveler", TEXT, "passenger|guest|traveller|flyer", person=True),
            F("cabin", ENUM, "seat class|travel class", "economy|business|first class|premium economy"),
            F("refundable", BOOL, "flexible|cancellable|free cancellation"),
            F("hotel", TEXT, "accommodation|lodging|inn"),
            F("stops", NUMBER, "layovers|connections|stopovers", adj="long|short|few"),
            F("reserved_on", DATE, "reserved|reservation day|bought on"),
            F("reviews", NUMBER, "guest feedback|feedback tally"),
            F("checked_bag", BOOL, "luggage included|bag included|baggage"),
            F("loyalty_tier", ENUM, "frequent flyer membership|loyalty", "bronze|silver|gold|platinum|diamond"),
        ],
    ),
}

TRAIN_DOMAINS = list(POOLS)
EVAL_DOMAINS = list(EVAL_POOLS)


# Pure function words carry no field identity, so sharing them is not a leak.
STOPWORDS = {"on", "to", "for", "with", "up", "off", "back", "going", "based", "first", "old",
             "by", "in", "at", "of", "the", "a", "an", "is", "has", "per", "and", "no"}


def _words(text: str) -> set[str]:
    return {w for w in text.lower().replace("_", " ").split() if w and w not in STOPWORDS}


def vocabulary(pools: dict[str, tuple[list[str], list[Field]]]) -> set[str]:
    out: set[str] = set()
    for _entities, fields in pools.values():
        for f in fields:
            out |= _words(f.name)
            for a in f.aliases:
                out |= _words(a)
            for v in f.values:
                out |= _words(v)
    return out  # polarity adjectives are shared lexicon words and are excluded on purpose


def assert_disjoint() -> None:
    """Held-out schemas must share no field word, alias word or enum value with training."""
    shared = vocabulary(POOLS) & vocabulary(EVAL_POOLS)
    if shared:
        raise AssertionError(f"held-out vocabulary overlaps training: {sorted(shared)}")


def date_target(schema: Schema) -> Field | None:
    """The date field a bare time phrase refers to: the only one, or the primary one."""
    dates = schema.of_kind(DATE)
    if len(dates) == 1:
        return dates[0]
    for f in dates:
        if f.primary:
            return f
    return None


def sample(split: str, rng: random.Random, domain: str | None = None) -> Schema:
    """A random schema: a domain and a random subset of its pool (4-9 fields, shuffled)."""
    pools = POOLS if split == "train" else EVAL_POOLS
    if domain is None:
        domain = rng.choice(list(pools))
    entities, pool = pools[domain]
    count = rng.randint(4, min(9, len(pool)))
    chosen = rng.sample(pool, count)
    fields: list[Field] = []
    for f in chosen:
        # Randomly hide aliases / drop enum values so the model cannot rely on a full spec.
        aliases = tuple(a for a in f.aliases if rng.random() < 0.8)
        values = f.values
        if values and rng.random() < 0.3 and len(values) > 3:
            values = tuple(rng.sample(values, rng.randint(3, len(values))))
        adjectives = tuple(a for a in f.adjectives if rng.random() < 0.8)
        fields.append(Field(f.name, f.kind, aliases, values, f.person, adjectives, f.year_like, f.unit))
    dates = [i for i, f in enumerate(fields) if f.kind == DATE]
    if len(dates) > 1 and rng.random() < 0.5:
        i = rng.choice(dates)
        fields[i] = Field(**{**fields[i].__dict__, "primary": True})
    return Schema(domain, rng.choice(entities), fields)


if __name__ == "__main__":
    assert_disjoint()
    print("train domains:", ", ".join(TRAIN_DOMAINS))
    print("eval domains: ", ", ".join(EVAL_DOMAINS))
    s = sample("train", random.Random(1))
    print(s.domain, s.entity)
    for f in s.fields:
        print(" ", f.name, f.kind, f.aliases, f.values)
