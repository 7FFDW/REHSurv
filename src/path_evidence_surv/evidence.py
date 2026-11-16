"""Auditable assertion extraction. Scores are heuristics, not calibrated probabilities.

States describe the supplied summary, not biological truth or the original report.
No network or language-model service is used.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

STATES = {"unmentioned": 0, "present": 1, "absent": 2, "uncertain": 3}
EVIDENCE_SCHEMA = "report_binary12_v2"
CONCEPTS = (
    ("necrosis", r"\bnecro(?:sis|tic|ses)\b|\bcomedonecrosis\b"),
    ("lymphovascular_invasion", r"\b(?:lymphovascular|vascular|lymphatic)\s+(?:space\s+)?invasion\b"),
    ("lymph_node_metastasis", r"\b(?:lymph\s+nodes?|nodal\s+(?:metasta\w*|involvement|status))\b"),
    ("margin_involvement", r"\b(?:surgical\s+|resection\s+)?margins?\b"),
    ("in_situ_component", r"\bin\s+situ\b|\b(?:DCIS|LCIS|CIS)\b"),
    ("glandular_pattern", r"\b(?:glandular|gland[- ]forming|glands?|acinar|tubular|tubule\w*)\b"),
    ("papillary_pattern", r"\b(?:papillary|micropapillary)\b"),
    ("solid_pattern", r"\bsolid\b"),
    ("stromal_fibrosis", r"\b(?:fibrosis|fibrotic|fibrous|desmoplas\w*)\b|\bstromal\s+(?:fibro\w*|reaction)\b"),
    ("calcification", r"\b(?:calcificat\w*|calcified|microcalcificat\w*|psammoma)\b"),
    ("inflammatory_infiltration", r"\b(?:inflamm\w*|lymphocytic|lymphoplasmacytic|lymphoid\s+infiltrat\w*)\b"),
    ("mucinous_features", r"\b(?:mucin\w*|mucous|mucocellular)\b"),
)
CONCEPT_NAMES = [name for name, _ in CONCEPTS]
_MISSING = re.compile(
    r"\bunmentioned\b"
    r"|not\s+(?:mentioned|reported|specified|provided|documented"
    r"|assessed|evaluated|stated|available|described)"
    r"|\b(?:unspecified|unreported|unassessed)\b"
    r"|no\s+(?:mention|information|description)"
    r"|information\s+(?:is\s+)?(?:unavailable|missing)",
    re.I,
)
_UNCERTAIN = re.compile(r"cannot\s+(?:exclude|assess|determine)|cannot\s+be\s+(?:excluded|assessed)|indeterminate|uncertain|unclear|equivocal|suspicious|possible|possibly|may\s+(?:be|represent)|unable\s+to", re.I)
_NEG_PREFIX = re.compile(r"(?:\bno\b|\bwithout\b|\babsent\b|negative\s+for|absence\s+of|free\s+of)(?:\W+\w+){0,5}\W*$", re.I)
_NEG_SUFFIX = re.compile(r"^\W*(?:(?:is|are|was|were|has|have|of|the|tumou?r|cells?|carcinoma)\W+){0,5}(?:not\s+(?:seen|identified|present|involved)|absent|negative|uninvolved|clear|free\s+of)\b", re.I)
_POS_SUFFIX = re.compile(r"^\W*(?:(?:is|are|was|were|by|the|tumou?r|cells?|carcinoma)\W+){0,5}(?:positive|involved|metastatic|present|identified|involvement)\b", re.I)
_POS_PREFIX = re.compile(r"(?:positive|involved|metastatic)(?:\W+\w+){0,2}\W*$", re.I)
_OUTCOME = re.compile(r"\bsurviv\w*|\bhazard\b|\bprognos\w*|\bdeath\b|\bdied\b|\bfollow[- ]up\b|\brecurren\w*", re.I)


@dataclass
class Assertion:
    concept: str
    state: str
    score: float
    spans: list[dict]
    origin: str = "report_binary12_v2_uncalibrated"


def _assertion_segments(clause):
    """Carry leading negation through bare coordinated items, never a new assertion."""
    negative = False
    separator = re.compile(r"\b(?:but|however|whereas|and|or)\b|,", re.I)
    cursor = 0
    for delimiter in list(separator.finditer(clause)) + [None]:
        end = delimiter.start() if delimiter else len(clause)
        raw = clause[cursor:end]
        phrase = raw.strip()
        if phrase:
            leading_negative = bool(re.match(r"(?:no\b|without\b|negative\s+for\b|absence\s+of\b|free\s+of\b)", phrase, re.I))
            # A finite verb or explicit assertion starts its own scope.
            explicit = bool(re.search(r"\b(?:is|are|was|were|has|have|shows?|demonstrates?|present|positive|identified|seen|not|possible|uncertain|suspicious)\b", phrase, re.I))
            inherited = negative and not leading_negative and not explicit
            yield phrase, cursor + len(raw) - len(raw.lstrip()), inherited
            negative = leading_negative or inherited
        if delimiter:
            if delimiter.group().lower() in {'but', 'however', 'whereas'}:
                negative = False
            cursor = delimiter.end()


def extract_evidence(text: str) -> list[dict]:
    # Keep offsets into the original text. Conjunctions delimit negation scope.
    clauses = list(re.finditer(r"[^.;\n]+", text))
    results = []
    for concept, pattern in CONCEPTS:
        found = []
        for clause in clauses:
            for phrase, offset, inherited_negative in _assertion_segments(clause.group()):
                if not phrase or phrase.lower() in {"but", "however", "whereas"}:
                    continue
                # Simpler exact offsets for each concept occurrence within the clause.
                for match in re.finditer(pattern, phrase, re.I):
                    left, right = phrase[:match.start()], phrase[match.end():]
                    nearby = " ".join(left.split()[-7:]) + " " + match.group() + " " + " ".join(right.split()[:7])
                    if _MISSING.search(nearby):
                        state, score = "unmentioned", 0.0
                    elif _UNCERTAIN.search(nearby):
                        state, score = "uncertain", 0.45
                    elif re.search(r"\b(?:not|never)\s+(?:submitted|sampled|examined|received|applicable)\b", nearby, re.I):
                        state, score = "unmentioned", 0.0
                    elif inherited_negative or _NEG_PREFIX.search(left) or _NEG_SUFFIX.search(right):
                        state, score = "absent", 0.85
                    elif concept in {"margin_involvement", "lymph_node_metastasis"}:
                        # A bare mention of margins/nodes is not evidence of involvement.
                        if re.search(r"\b(?:negative|uninvolved|free\s+(?:of|from)|no\s+(?:tumou?r|carcinoma|metastasis))\b", right, re.I):
                            state, score = "absent", 0.85
                        elif concept == "lymph_node_metastasis" and re.search(r"\b0\s*(?:/|of)\s*\d+", phrase, re.I):
                            state, score = "absent", 0.85
                        elif _POS_SUFFIX.search(right) or _POS_PREFIX.search(left) or (
                            concept == "lymph_node_metastasis" and (
                                re.search(r"\b(?:metastatic|metastasis|metastases)\b", phrase, re.I)
                                or re.search(r"\b[1-9]\d*\s*(?:/|of)\s*\d+", phrase, re.I))):
                            state, score = "present", 0.80
                        else:
                            state, score = "uncertain", 0.40
                    else:
                        state, score = "present", 0.80
                    start = clause.start() + offset
                    found.append({"state": state, "score": score, "text": phrase,
                                  "start": max(start, clause.start()),
                                  "end": max(start, clause.start()) + len(phrase)})
        observed = {item["state"] for item in found if item["state"] != "unmentioned"}
        if {"present", "absent"}.issubset(observed) or "uncertain" in observed:
            state, score = "uncertain", 0.4
        elif observed:
            state = next(iter(observed)); score = max(x["score"] for x in found if x["state"] == state)
        else:
            state, score = "unmentioned", 0.0
        results.append(asdict(Assertion(concept, state, score, found)))
    return results


def vectorize(assertions: list[dict]) -> tuple[list[int], list[float]]:
    by_name = {item["concept"]: item for item in assertions}
    return ([STATES[by_name[name]["state"]] for name in CONCEPT_NAMES],
            [float(by_name[name]["score"]) for name in CONCEPT_NAMES])


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_flags(text: str) -> dict:
    return {"outcome_terms_for_review": bool(_OUTCOME.search(text)),
            "word_count": len(re.findall(r"\b[\w-]+\b", text))}


def rewrite_style(text: str) -> str:
    """Controlled lexical/format perturbation; not a clinical equivalence guarantee."""
    text = re.sub(r"The pathology report (?:indicates|describes|shows) that\s*", "", text, flags=re.I)
    text = re.sub(r"\bnot identified\b", "not seen", text, flags=re.I)
    text = re.sub(r"\bwithout evidence of\b", "no evidence of", text, flags=re.I)
    sentences = [s.strip() for s in re.split(r"[.\n]+", text) if s.strip()]
    return ". ".join(reversed(sentences)) + ("." if sentences else "")


def delete_concept_sentences(text: str, concept: str) -> str:
    pattern = dict(CONCEPTS)[concept]
    return ". ".join(s.strip() for s in re.split(r"[.\n]+", text)
                     if s.strip() and not re.search(pattern, s, re.I))
