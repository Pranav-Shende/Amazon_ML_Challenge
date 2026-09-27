"""Central configuration for the entity-resolution pipeline.

Everything tunable lives here so that experiments are reproducible by
editing a single file (or overriding via CLI flags in ``run_pipeline.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Vocabulary used by the normalizer
# --------------------------------------------------------------------------

# Legal / trade suffixes stripped when deriving the *core* business name.
# They are kept in the "full" form of the name so that a genuine
# "Inc" vs "LLC" difference can still be scored as a weak signal.
NAME_SUFFIXES = frozenset(
    {
        "inc", "incorporated", "corp", "corporation", "co", "company",
        "ltd", "limited", "llc", "llp", "lp", "plc", "gmbh", "ag", "sa",
        "sas", "sarl", "bv", "nv", "ab", "oy", "as", "pty", "pte",
        "public", "holdings", "holding", "group", "trading",
        "enterprises", "enterprise", "srl", "spa", "kg", "kgaa", "oao",
        "pvt", "kft", "zrt", "doo", "shpk", "mb", "srlu", "srl",
        "services", "service", "stores", "store", "solutions",
    }
)

# Abbreviation -> expansion, used for BUSINESS NAMES only.
# Kept separate from the address map so that e.g. "Co" stays "company"
# in a name but is never expanded to a US state there.
NAME_ABBREVS = {
    "corp": "corporation", "corps": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private", "priv": "private",
    "co": "company", "cos": "company", "comp": "company",
    "dept": "department", "div": "division",
    "mfg": "manufacturing", "mfr": "manufacturer",
    "svcs": "services", "svc": "service",
    "assoc": "associates", "assn": "association",
    "intl": "international", "natl": "national",
    "mgmt": "management", "fin": "finance",
    "bk": "bank", "bldg": "building",
    "sys": "systems", "tech": "technologies", "eng": "engineering",
    "pharma": "pharmaceuticals", "pharm": "pharmacy",
    "rest": "restaurant", "resto": "restaurant",
    "cl": "clinic", "hosp": "hospital", "univ": "university",
    "sch": "school", "acad": "academy", "inst": "institute",
    "ctr": "center", "cent": "center",
    "ind": "industries", "indl": "industrial",
    "whsl": "wholesale", "ret": "retail",
    "dist": "distributors", "distr": "distributors",
    "mkt": "market", "supl": "supplies", "supp": "supplies",
    "prod": "products", "prov": "providers",
}

# Abbreviation -> expansion, used for ADDRESS tokens only.
ADDRESS_ABBREVS = {
    # roads / thoroughfares
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "a": "avenue", "blvd": "boulevard", "ln": "lane", "dr": "drive",
    "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "sq": "square", "ter": "terrace", "trl": "trail",
    "rt": "route", "expy": "expressway", "fwy": "freeway",
    "xing": "crossing", "jnc": "junction", "grn": "green", "prt": "park",
    "plz": "plaza", "sq": "square", "way": "way",

    # units / buildings
    "apt": "apartment", "ste": "suite", "rm": "room", "fl": "floor",
    "bldg": "building", "unit": "unit", "no": "number", "num": "number",
    "flr": "floor",

    # relation words
    "nr": "near", "opp": "opposite", "n": "near", "b": "behind",
    "adj": "adjacent", "op": "opposite",

    # corporate (address form)
    "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "co": "company", "rd": "road", "st": "street",

    # landmarks / generic
    "sbi": "statebankofindia", "hdfc": "hdfcbank", "icici": "icicibank",
    "atm": "atm", "main": "main", "cross": "cross",
    "police": "policestation", "ps": "policestation",
}

# Region / state abbreviations -- ONLY applied inside addresses, never names.
# Merged US + India + France tables; collisions are resolved by region tag
# but since we only expand when the token sits in an address, a rare
# ambiguity ("or" = Oregon vs a word) is harmless for blocking.
REGION_ABBREVS = {
    # United States
    "ca": "california", "ny": "newyork", "tx": "texas", "fl": "florida",
    "il": "illinois", "pa": "pennsylvania", "oh": "ohio", "ga": "georgia",
    "nc": "northcarolina", "mi": "michigan", "nj": "newjersey",
    "va": "virginia", "wa": "washington", "az": "arizona",
    "ma": "massachusetts", "tn": "tennessee", "mo": "missouri",
    "md": "maryland", "wi": "wisconsin", "mn": "minnesota",
    "al": "alabama", "la": "louisiana", "ky": "kentucky",
    "ok": "oklahoma", "ct": "connecticut", "ut": "utah", "ia": "iowa",
    "nv": "nevada", "ar": "arkansas", "ms": "mississippi",
    "ks": "kansas", "nm": "newmexico", "ne": "nebraska", "id": "idaho",
    "wv": "westvirginia", "hi": "hawaii", "nh": "newhampshire",
    "me": "maine", "mt": "montana", "ri": "rhodeisland",
    "de": "delaware", "sd": "southdakota", "nd": "northdakota",
    "ak": "alaska", "vt": "vermont", "wy": "wyoming",
    "dc": "washingtondc",

    # India
    "ap": "andhrapradesh", "ts": "telangana", "ka": "karnataka",
    "mp": "madhyapradesh", "mh": "maharashtra", "up": "uttarpradesh",
    "wb": "westbengal", "br": "bihar", "rj": "rajasthan",
    "gj": "gujarat", "od": "odisha", "pb": "punjab", "hr": "haryana",
    "jk": "jammukashmir", "cg": "chhattisgarh", "jh": "jharkhand",
    "uk": "uttarakhand", "hp": "himachalpradesh", "as": "assam",
    "kl": "kerala", "go": "goa",

    # France (test-only country)
    "iledefrance": "iledefrance", "idf": "iledefrance",
    "rhone": "rhone", "pac": "provencealpescotedazur",
}

# "&" and similar symbols become words before tokenisation.
SYMBOL_WORDS = {"&": " and ", "+": " plus ", "@": " at ", "/": " "}

# Tokens that carry no discriminating signal.
STOP_TOKENS = frozenset(
    {
        "the", "a", "an", "of", "and", "or", "for", "in", "on", "at",
        "to", "by", "with", "from", "de", "la", "el", "le", "les", "du",
    }
)


# --------------------------------------------------------------------------
# Pipeline settings
# --------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All knobs for blocking, featurisation, and thresholding."""

    # --- blocking / candidate generation ---------------------------------
    # Exact-key blocking over a sorted packed array (see blocking.py).
    # Records sharing a key with more than `df_cap` others are skipped for
    # that key: ubiquitous tokens ("street", "road") carry no signal, and the
    # cap is what keeps fan-out bounded on a 10M-record corpus.
    df_cap: int = 60
    # Hard budget per Source 1 row.  Raising it buys recall at scoring cost;
    # blocking determines the recall ceiling, so start here if F_0.5 looks
    # recall-capped.
    max_candidates_per_row: int = 80
    query_chunk: int = 5_000       # Source 1 rows held in flight at once

    # --- features ---------------------------------------------------------
    # Number of pairwise features (fixed schema; see features.FEATURE_NAMES).
    # Kept for reporting only -- the model consumes whatever it was trained on.

    # --- model ------------------------------------------------------------
    # Optimised on the held-out validation split for macro F_0.5.
    threshold: float = 0.52
    # Threshold used by the label-free heuristic scorer when there is no
    # ground truth to calibrate against.  Chosen on the precision-heavy
    # side, since F_0.5 weights precision 2x over recall.
    heuristic_threshold: float = 0.65
    calibrate_threshold: bool = True
    threshold_grid: tuple[float, ...] = tuple(
        round(0.20 + 0.02 * i, 2) for i in range(31)
    )
    negative_ratio: float = 4.0    # negatives sampled per positive when training
    random_state: int = 42

    # --- runtime ----------------------------------------------------------
    # Scoring is Python-bound (RapidFuzz + feature assembly), so the GIL
    # makes threads useless here; process workers each memory-map the same
    # target blobs, costing one copy of the data rather than N.
    workers: int = 0            # 0 => use every logical core
    # Training/cap knobs.  A 37-feature linear model saturates well before
    # it has seen every Source 1 row, so both passes accept a row cap: the
    # cost is linear in rows, and these caps are what keep a full run inside
    # a couple of hours on 7.6 GB of RAM.  0 = use everything.
    train_rows: int = 300_000
    calib_rows: int = 50_000
    verbose: bool = True

    extra: dict = field(default_factory=dict)

    def tag(self) -> str:
        """Short human-readable fingerprint for logs."""
        return (
            f"keys df_cap={self.df_cap} maxcand={self.max_candidates_per_row} "
            f"thr={self.threshold:.2f}"
        )
