import re
from app.config import MEMORY_PATTERNS
from app.schemas import (
    MemoryRequest,
    MemoryResponse,
    MemoryCandidate,
    MemoryType,
)


# ─── Entity patterns ──────────────────────────────────────────────────────────
# These go beyond simple keyword matching into structured extraction.
# Each pattern group captures a specific memory type with context.

LIFE_EVENT_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern, event title template)
    (r"\b(my|our|the)?\s*(dog|cat|pet|bird|fish)\s*(passed away|died|was put down|is gone)\b",
     "Pet loss"),
    (r"\b(passed away|died|lost (my|our|their|his|her))\s+\w+\b",
     "Bereavement"),
    (r"\b(broke up|broke up with|ended (the|our|my) relationship|got dumped|was dumped|left (me|him|her|them))\b",
     "Relationship breakdown"),
    (r"\b(lost (my|a) job|was fired|got fired|laid off|made redundant|lost (my|a) position)\b",
     "Job loss"),
    (r"\b(moved (out|in|away|to|from)|relocated|new (house|apartment|flat|city|country))\b",
     "Relocation"),
    (r"\b(diagnosed with|told (i have|i had|i was)|found out (i have|i had))\b",
     "Medical diagnosis"),
    (r"\b(accident|injury|surgery|operation|hospitalised|hospitalized|in hospital)\b",
     "Medical event"),
    (r"\b(divorced|divorce|separation|separated from)\b",
     "Divorce or separation"),
    (r"\b(graduated|finished (school|university|college)|dropped out)\b",
     "Academic transition"),
    (r"\b(pregnant|had a baby|gave birth|miscarriage|lost the baby)\b",
     "Pregnancy or birth event"),
    (r"\b(was (abused|assaulted|attacked|harassed|bullied))\b",
     "Trauma or abuse"),
    (r"\b(evicted|lost (my|our) home|became homeless)\b",
     "Housing crisis"),
]

RELATIONSHIP_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, title template, memory type hint)
    (r"\bmy (mom|mother|mum)\b",               "Mother",           "family"),
    (r"\bmy (dad|father|pop|papa)\b",           "Father",           "family"),
    (r"\bmy (sister|brother|sibling)\b",        "Sibling",          "family"),
    (r"\bmy (partner|boyfriend|girlfriend|husband|wife|spouse|fiancé|fiancée)\b",
     "Romantic partner",  "relationship"),
    (r"\bmy (best friend|closest friend)\b",    "Best friend",      "friendship"),
    (r"\bmy (friend|mate)\b",                   "Friend",           "friendship"),
    (r"\bmy (therapist|counselor|psychiatrist|psychologist)\b",
     "Therapist or mental health professional", "support"),
    (r"\bmy (dog|cat|pet)\b",                   "Pet",              "support"),
    (r"\bmy (boss|manager|supervisor)\b",       "Boss or manager",  "work"),
    (r"\bmy (colleague|coworker|workmate)\b",   "Colleague",        "work"),
    (r"\bmy (teacher|professor|tutor|mentor)\b","Mentor or teacher","support"),
    (r"\bmy (son|daughter|child|kid|baby)\b",   "Child",            "family"),
    (r"\bmy (grandma|grandpa|grandmother|grandfather|gran|granny)\b",
     "Grandparent",       "family"),
]

PROTECTIVE_FACTOR_PATTERNS: list[tuple[str, str]] = [
    (r"\b(my (dog|cat|pet))\s+.{0,30}(help|keeps|loves|comfort|ground)\b",
     "Pet as comfort"),
    (r"\b(music|playing (guitar|piano|drums|music))\s+.{0,30}(help|keeps|calm|escape)\b",
     "Music as coping"),
    (r"\b(working out|gym|running|exercise|yoga)\s+.{0,30}(help|keeps|better|cope)\b",
     "Physical exercise"),
    (r"\b(journaling|writing|drawing|painting|art)\s+.{0,30}(help|keeps|better|cope|express)\b",
     "Creative outlet"),
    (r"\b(looking forward to|excited (about|for)|can't wait (for|to))\b",
     "Future positive anchor"),
    (r"\b(my (faith|religion|church|mosque|temple|god|prayer))\b",
     "Spiritual or religious support"),
    (r"\b(my support (group|network|system|team))\b",
     "Support network"),
    (r"\btherapy (is helping|has been helping|helps|helped)\b",
     "Therapy as positive support"),
]

RECURRING_THEME_PATTERNS: list[tuple[str, str]] = [
    (r"\b(always feel|keep feeling|constantly feel|never stop feeling)\b",
     "Recurring emotional state"),
    (r"\b(every (time|night|morning|week|day))\s+.{0,40}(feel|cry|panic|spiral|break down)\b",
     "Recurring emotional episode"),
    (r"\b(this always (happens|comes back)|it never (goes away|gets better|stops))\b",
     "Persistent pattern"),
    (r"\b(since (i was|childhood|school|i was a kid))\s+.{0,50}(feel|struggle|have been)\b",
     "Long-standing pattern"),
    (r"\b(no (one|body) (ever|always|never))\s+.{0,40}(understands|cares|listens|believes)\b",
     "Persistent feeling of being unheard"),
    (r"\b(i (always|never)) (feel (good enough|worthy|loveable|safe|wanted))\b",
     "Core self-belief pattern"),
]


# ─── Extraction helpers ───────────────────────────────────────────────────────

def _extract_surrounding_context(
    text: str,
    start: int,
    end: int,
    window: int = 80,
) -> str:
    """
    Extract text surrounding a match for richer description.
    Clips to sentence boundaries where possible.
    """
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)
    snippet = text[context_start:context_end].strip()

    # Try to trim to sentence boundaries
    sentence_end = re.search(r"[.!?]", snippet[end - context_start:])
    if sentence_end:
        snippet = snippet[:end - context_start + sentence_end.start() + 1]

    return snippet.strip()


def _extract_life_events(text: str) -> list[MemoryCandidate]:
    candidates = []

    for pattern, title in LIFE_EVENT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            description = _extract_surrounding_context(
                text, match.start(), match.end()
            )
            candidates.append(
                MemoryCandidate(
                    type=MemoryType.LIFE_EVENT,
                    title=title,
                    description=description or match.group(0),
                    confidence=0.80,
                )
            )

    return candidates


def _extract_relationships(text: str) -> list[MemoryCandidate]:
    candidates = []
    seen_titles: set[str] = set()

    for pattern, title, _ in RELATIONSHIP_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches and title not in seen_titles:
            seen_titles.add(title)
            # Take the first occurrence for description
            match = matches[0]
            description = _extract_surrounding_context(
                text, match.start(), match.end()
            )
            candidates.append(
                MemoryCandidate(
                    type=MemoryType.RELATIONSHIP,
                    title=title,
                    description=description or match.group(0),
                    confidence=0.75,
                )
            )

    return candidates


def _extract_protective_factors(text: str) -> list[MemoryCandidate]:
    candidates = []

    for pattern, title in PROTECTIVE_FACTOR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            description = _extract_surrounding_context(
                text, match.start(), match.end()
            )
            candidates.append(
                MemoryCandidate(
                    type=MemoryType.PROTECTIVE_FACTOR,
                    title=title,
                    description=description or match.group(0),
                    confidence=0.70,
                )
            )

    return candidates


def _extract_recurring_themes(text: str) -> list[MemoryCandidate]:
    candidates = []

    for pattern, title in RECURRING_THEME_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            description = _extract_surrounding_context(
                text, match.start(), match.end()
            )
            candidates.append(
                MemoryCandidate(
                    type=MemoryType.RECURRING_THEME,
                    title=title,
                    description=description or match.group(0),
                    confidence=0.65,
                )
            )

    return candidates


# ─── Deduplication ────────────────────────────────────────────────────────────

def _deduplicate(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
    """
    Remove duplicate candidates with the same type + title.
    Keeps the one with the highest confidence.
    """
    seen: dict[str, MemoryCandidate] = {}

    for candidate in candidates:
        key = f"{candidate.type.value}::{candidate.title.lower()}"
        if key not in seen or candidate.confidence > seen[key].confidence:
            seen[key] = candidate

    return list(seen.values())


# ─── Main service function ────────────────────────────────────────────────────

def extract_memory_candidates(request: MemoryRequest) -> MemoryResponse:
    """
    Extract structured memory candidates from patient text.

    Runs four extraction passes:
        1. Life events
        2. Relationships
        3. Protective factors
        4. Recurring themes

    Returns deduplicated, sorted candidates for therapist review.
    """
    text = request.text

    all_candidates: list[MemoryCandidate] = []

    all_candidates.extend(_extract_life_events(text))
    all_candidates.extend(_extract_relationships(text))
    all_candidates.extend(_extract_protective_factors(text))
    all_candidates.extend(_extract_recurring_themes(text))

    # Deduplicate
    unique = _deduplicate(all_candidates)

    # Sort: life events first (highest clinical salience),
    # then by confidence descending within each type
    type_order = {
        MemoryType.LIFE_EVENT:        0,
        MemoryType.RECURRING_THEME:   1,
        MemoryType.RELATIONSHIP:      2,
        MemoryType.PROTECTIVE_FACTOR: 3,
    }
    unique.sort(
        key=lambda c: (type_order.get(c.type, 99), -c.confidence)
    )

    return MemoryResponse(
        candidates=unique,
        count=len(unique),
    )