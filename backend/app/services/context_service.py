import re
from app.config import THIRD_PERSON_INDICATORS, SUPPORT_SEEKING_INDICATORS
from app.schemas import ContextLabel


# ─── First-person self-reference patterns ─────────────────────────────────────
# These are the clearest signals that text is self-directed.
# Derived from your SHAP findings — "I", "me", "myself" were
# consistently the strongest self-directed markers. [pbl_project]

FIRST_PERSON_CRISIS_PATTERNS: list[str] = [
    r"\bi (want to|wanted to|need to|am going to|will) (die|end it|disappear|hurt myself)\b",
    r"\bi (can't|cannot) (go on|take this|do this anymore|keep going)\b",
    r"\bi (don't|do not) (want to|see a reason to) (live|be here|exist|continue)\b",
    r"\bi (feel|felt) (empty|numb|hollow|nothing|hopeless|worthless|useless)\b",
    r"\bi (am|was) (so alone|completely alone|totally alone|all alone)\b",
    r"\bi (wish|wished) i (was|were|wasn't|weren't) (here|alive|born)\b",
    r"\b(kill|hurt|harm) my(self)?\b",
    r"\bending (my|this) life\b",
    r"\bno (point|reason|purpose) (in|to) (living|life|going on|existing)\b",
    r"\bi (have been|have|had) (thoughts of|thinking about) (suicide|ending it)\b",
]

FIRST_PERSON_DISTRESS_PATTERNS: list[str] = [
    r"\bi (feel|felt|am feeling|was feeling)\b",
    r"\bmy (life|mind|head|heart|soul)\b",
    r"\bi (can't|cannot|don't|didn't|haven't)\b",
    r"\bi (am|was|have been|keep)\b",
    r"\bmy (therapist|counselor|doctor|meds|medication)\b",
]

# ─── Third-person patterns ────────────────────────────────────────────────────
# Your SHAP analysis found that the model confused third-person suicide
# discussion (e.g. "my friend attempted") with self-directed crisis.
# This is the key correction layer. [pbl_project]

THIRD_PERSON_CRISIS_PATTERNS: list[str] = [
    r"\b(my|a) (friend|sister|brother|mom|dad|partner|colleague|roommate|classmate)\b.{0,60}(suicide|died|passed|attempt|hurt|harm)\b",
    r"\b(she|he|they) (tried|attempted|is thinking about|wants to)\b.{0,40}(suicide|die|end it|hurt)\b",
    r"\b(asking for|this is for) a friend\b",
    r"\bsomeone (i know|close to me|in my life)\b.{0,40}(struggling|in crisis|suicidal|depressed)\b",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _count_first_person(text: str) -> int:
    """Count first-person singular pronoun occurrences."""
    pattern = r"\b(i|me|my|mine|myself)\b"
    return len(re.findall(pattern, text, re.IGNORECASE))


def _count_third_person(text: str) -> int:
    """Count third-person pronoun and indicator occurrences."""
    count = 0

    # Pronouns
    pattern = r"\b(he|she|they|him|her|them|his|hers|their|my friend|my sister|my brother)\b"
    count += len(re.findall(pattern, text, re.IGNORECASE))

    # Explicit indicators from config
    for indicator in THIRD_PERSON_INDICATORS:
        if indicator.lower() in text.lower():
            count += 2  # Weight explicit indicators more heavily

    return count


def _has_support_seeking(text: str) -> bool:
    """Check if text is framed as a support request rather than self-disclosure."""
    for indicator in SUPPORT_SEEKING_INDICATORS:
        if indicator.lower() in text.lower():
            return True

    # Additional pattern-based check
    support_patterns = [
        r"\b(does anyone|has anyone|can anyone)\b",
        r"\b(need (advice|help|support|someone to talk))\b",
        r"\b(what (should|can|do) i do)\b",
        r"\b(looking for (help|advice|support|resources))\b",
        r"\b(any (advice|tips|suggestions|thoughts))\b",
    ]
    for pattern in support_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def _has_third_person_crisis(text: str) -> bool:
    """
    Check if crisis-level language is specifically about someone else.
    This is the core fix for the Depression/SuicideWatch confusion
    found in your SHAP analysis. [pbl_project]
    """
    for pattern in THIRD_PERSON_CRISIS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _has_first_person_crisis(text: str) -> bool:
    """Check for explicit self-directed crisis language."""
    for pattern in FIRST_PERSON_CRISIS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ─── Main context detection ───────────────────────────────────────────────────

def detect_context(text: str) -> ContextLabel:
    """
    Classify the perspective and intent of patient text.

    Priority order (most specific to least):
        1. Support-seeking — framed as asking for help/advice
        2. Third-person crisis — crisis language about someone else
        3. Self-directed — explicit first-person crisis or distress
        4. Ambiguous — cannot confidently determine

    This directly addresses the model's known failure mode:
    flagging third-person suicide discussion as SuicideWatch.
    [pbl_project SHAP analysis]
    """

    # ── 1. Support-seeking check first ───────────────────────────────────────
    # e.g. "does anyone else feel this way?" or "need advice for my friend"
    if _has_support_seeking(text):
        return ContextLabel.SUPPORT_SEEKING

    # ── 2. Third-person crisis ────────────────────────────────────────────────
    # e.g. "my friend attempted suicide" — this should NOT escalate as
    # self-directed crisis even if model score is high
    if _has_third_person_crisis(text):
        return ContextLabel.THIRD_PERSON

    # ── 3. Self-directed ──────────────────────────────────────────────────────
    # Explicit first-person crisis patterns take priority
    if _has_first_person_crisis(text):
        return ContextLabel.SELF_DIRECTED

    # ── 4. Pronoun ratio disambiguation ───────────────────────────────────────
    # When no explicit patterns match, use pronoun balance
    # as a softer signal
    first_count = _count_first_person(text)
    third_count = _count_third_person(text)

    if first_count == 0 and third_count == 0:
        return ContextLabel.AMBIGUOUS

    # Clear first-person dominance
    if first_count > 0 and third_count == 0:
        # Check for at least some distress signal
        for pattern in FIRST_PERSON_DISTRESS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return ContextLabel.SELF_DIRECTED
        return ContextLabel.AMBIGUOUS

    # Clear third-person dominance
    if third_count > first_count * 1.5:
        return ContextLabel.THIRD_PERSON

    # Mixed — likely self-directed with references to others
    if first_count >= third_count:
        return ContextLabel.SELF_DIRECTED

    return ContextLabel.AMBIGUOUS


# ─── Utility for UI ───────────────────────────────────────────────────────────

def get_context_explanation(context: ContextLabel) -> str:
    """
    Human-readable explanation of the context label.
    Used in the therapist UI tooltip.
    """
    explanations = {
        ContextLabel.SELF_DIRECTED: (
            "Language appears to be about the patient's own experience. "
            "First-person distress or crisis framing detected."
        ),
        ContextLabel.THIRD_PERSON: (
            "Language appears to be about someone else. "
            "Crisis vocabulary may relate to another person, not the patient."
        ),
        ContextLabel.SUPPORT_SEEKING: (
            "Text is framed as a request for help, advice, or information. "
            "May be self-directed or about another person."
        ),
        ContextLabel.AMBIGUOUS: (
            "Context could not be determined confidently. "
            "Therapist should assess perspective directly."
        ),
    }
    return explanations[context]