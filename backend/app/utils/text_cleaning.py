import re
import unicodedata
from app.config import settings


# ─── Contraction expansion ────────────────────────────────────────────────────
# Keep these — mental health text is full of contractions and they carry meaning

CONTRACTIONS: dict[str, str] = {
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "i can't": "i cannot",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "he's": "he is",
    "she's": "she is",
    "what's": "what is",
    "who's": "who is",
    "not gonna": "not going to",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
}


# ─── Emoticon → text ──────────────────────────────────────────────────────────
# Common emoticons that carry sentiment signal in patient messages

EMOTICONS: dict[str, str] = {
    ":)": "happy",
    ":-)": "happy",
    ":(": "sad",
    ":-(": "sad",
    ":'(": "crying",
    ":')": "happy crying",
    ":/": "uncertain",
    ":|": "neutral",
    ">:(": "angry",
    ":D": "very happy",
    ";)": "winking",
    "<3": "love",
    "</3": "heartbreak",
}


# ─── Core cleaning functions ──────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII-safe form."""
    return unicodedata.normalize("NFKC", text)


def replace_emoticons(text: str) -> str:
    """Replace emoticons with descriptive words before other cleaning."""
    for emoticon, label in EMOTICONS.items():
        text = text.replace(emoticon, f" {label} ")
    return text


def expand_contractions(text: str) -> str:
    """Expand contractions — important for preserving negation meaning."""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text)
    return text


def remove_urls(text: str) -> str:
    """Strip URLs."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_reddit_markdown(text: str) -> str:
    """Remove Reddit-specific markdown artifacts."""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)   # bold/italic
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)       # underline
    text = re.sub(r"~~(.*?)~~", r"\1", text)                # strikethrough
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text)       # code
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # list items
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links
    text = re.sub(r"&gt;.*$", "", text, flags=re.MULTILINE)  # blockquotes
    return text


def remove_usernames_and_subreddits(text: str) -> str:
    """Strip Reddit usernames and subreddit mentions."""
    text = re.sub(r"u/\S+", "", text)
    text = re.sub(r"r/\S+", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and newlines."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def remove_excessive_punctuation(text: str) -> str:
    """
    Reduce repeated punctuation but don't strip all punctuation.
    '!!!!!' → '!' but '...' stays — ellipsis carries meaning.
    """
    text = re.sub(r"([!?])\1{2,}", r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    return text


# ─── Important: what NOT to clean ────────────────────────────────────────────
# Do NOT:
# - remove all punctuation (? and . carry sentence structure)
# - remove stop words (i, me, myself carry first-person signal)
# - aggressively stem/lemmatize (kills nuance in clinical language)
# - lowercase everything before emoticon replacement (breaks some patterns)


# ─── Main cleaning pipeline ───────────────────────────────────────────────────

def clean_text(text: str, for_model: bool = True) -> str:
    """
    Full cleaning pipeline.

    Args:
        text: Raw input text from therapist paste or chat.
        for_model: If True, applies full pipeline including contraction
                   expansion. If False, lighter clean for display purposes.

    Returns:
        Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = replace_emoticons(text)
    text = remove_urls(text)
    text = remove_reddit_markdown(text)
    text = remove_usernames_and_subreddits(text)
    text = remove_excessive_punctuation(text)

    if for_model:
        text = expand_contractions(text)

    text = normalize_whitespace(text)

    # Hard length guard — truncate to avoid model overflow
    # Keep it generous since long clinical text is valid
    max_chars = settings.max_token_length * 6
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


def is_too_short(text: str) -> bool:
    """Check if text is too short to produce a meaningful analysis."""
    return len(text.strip().split()) < settings.min_text_length


def get_display_text(text: str) -> str:
    """
    Lighter clean for showing text back in the UI with highlights.
    Preserves original casing and most formatting.
    """
    return clean_text(text, for_model=False)