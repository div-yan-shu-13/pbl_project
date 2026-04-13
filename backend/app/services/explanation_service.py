import re
from app.config import SIGNAL_LEXICONS
from app.schemas import EvidenceSpan


# ─── Scoring weights ──────────────────────────────────────────────────────────
# Not all signals carry the same clinical weight.
# Crisis-level signals score higher so they surface first in the UI.

SIGNAL_WEIGHTS: dict[str, float] = {
    "self-harm language":    1.0,
    "existential finality":  0.95,
    "hopelessness":          0.80,
    "grief":                 0.70,
    "isolation":             0.65,
    "withdrawal":            0.60,
    "anxiety":               0.55,
    "sleep issues":          0.45,
}

# How many spans to return maximum — keeps UI readable
MAX_SPANS = 8

# Minimum phrase length to avoid matching single noisy characters
MIN_PHRASE_LENGTH = 3


# ─── Core matching ────────────────────────────────────────────────────────────

def _find_phrase_in_text(
    text: str,
    phrase: str,
) -> list[tuple[int, int]]:
    """
    Find all occurrences of a phrase in text.
    Returns list of (start_idx, end_idx) tuples.
    Case-insensitive, whole-word aware where possible.
    """
    matches = []
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)

    for match in pattern.finditer(text):
        matches.append((match.start(), match.end()))

    return matches


def _overlaps_existing(
    start: int,
    end: int,
    existing: list[EvidenceSpan],
) -> bool:
    """
    Check if a new span overlaps with any already accepted span.
    Prevents the same text from being highlighted twice with different labels.
    """
    for span in existing:
        if span.start_idx is None or span.end_idx is None:
            continue
        # Overlap if one starts before the other ends
        if not (end <= span.start_idx or start >= span.end_idx):
            return True
    return False


# ─── Main extraction ──────────────────────────────────────────────────────────

def extract_evidence_spans(text: str) -> list[EvidenceSpan]:
    """
    Scan text for signal phrases from all lexicons.
    Returns a ranked, deduplicated list of EvidenceSpan objects.

    The text passed here should be display text (lightly cleaned)
    so that character indices align with what the frontend renders.

    Steps:
        1. Scan every lexicon for phrase matches
        2. Assign base score from signal weight
        3. Boost score for longer, more specific phrases
        4. Deduplicate overlapping spans
        5. Sort by score descending
        6. Return top MAX_SPANS results
    """
    if not text or not text.strip():
        return []

    candidates: list[tuple[float, str, int, int, str]] = []
    # Each candidate: (score, phrase_text, start, end, label)

    for label, phrases in SIGNAL_LEXICONS.items():
        base_weight = SIGNAL_WEIGHTS.get(label, 0.5)

        for phrase in phrases:
            if len(phrase) < MIN_PHRASE_LENGTH:
                continue

            occurrences = _find_phrase_in_text(text, phrase)

            for start, end in occurrences:
                matched_text = text[start:end]

                # Boost longer phrases — they're more specific
                length_boost = min(len(phrase) / 30, 0.15)
                score = round(base_weight + length_boost, 4)
                score = min(score, 1.0)

                candidates.append((score, matched_text, start, end, label))

    if not candidates:
        return []

    # Sort by score descending so highest-weight signals come first
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate — keep highest-scoring span when overlaps exist
    accepted: list[EvidenceSpan] = []

    for score, phrase_text, start, end, label in candidates:
        if len(accepted) >= MAX_SPANS:
            break

        if _overlaps_existing(start, end, accepted):
            continue

        accepted.append(
            EvidenceSpan(
                text=phrase_text,
                label=label,
                score=score,
                start_idx=start,
                end_idx=end,
            )
        )

    return accepted


# ─── Grouped summary for UI ───────────────────────────────────────────────────

def group_spans_by_label(spans: list[EvidenceSpan]) -> dict[str, list[str]]:
    """
    Group evidence spans by their signal label.
    Used by the frontend to render signal chips with supporting phrases.

    Example output:
    {
        "hopelessness": ["no point", "feel empty"],
        "grief": ["passed away"],
    }
    """
    grouped: dict[str, list[str]] = {}

    for span in spans:
        if span.label not in grouped:
            grouped[span.label] = []
        if span.text not in grouped[span.label]:
            grouped[span.label].append(span.text)

    return grouped


# ─── Fallback for short texts ─────────────────────────────────────────────────

def get_fallback_spans(signal_labels: list[str]) -> list[EvidenceSpan]:
    """
    When no phrase-level matches are found but the model still
    predicts a non-low class, return generic signal indicators
    so the UI is not completely empty.

    This can happen with short or abstractly-phrased text where
    the model picks up distributional signals the lexicon misses.
    """
    fallback = []

    for label in signal_labels[:3]:  # Max 3 fallbacks
        if label in SIGNAL_WEIGHTS:
            fallback.append(
                EvidenceSpan(
                    text=f"[{label} detected — no specific phrase identified]",
                    label=label,
                    score=round(SIGNAL_WEIGHTS.get(label, 0.4) * 0.6, 4),
                    start_idx=None,
                    end_idx=None,
                )
            )

    return fallback