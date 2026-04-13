from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import model_validator


# ─── Resolve base paths first, before Settings ────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class Settings(BaseSettings):

    # App
    app_name: str = "Therapist Assistant API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Model paths — defaults computed from BASE_DIR
    model_dir: str = str(MODELS_DIR / "classifier")
    label_encoder_path: str = str(MODELS_DIR / "label_encoder.pkl")

    # Inference
    max_token_length: int = 256
    min_text_length: int = 5
    device: str = "cpu"
    batch_size: int = 1
    confidence_threshold_strong: float = 0.80
    confidence_threshold_medium: float = 0.55

    # CORS
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
    ]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": (),
    }


# ─── Singleton ────────────────────────────────────────────────────────────────

settings = Settings()

# ─── Label mappings ───────────────────────────────────────────────────────────

RISK_TIER_MAP: dict[str, str] = {
    "Control":      "Low",
    "Anxiety":      "Moderate",
    "Depression":   "High",
    "BPD":          "High",
    "SuicideWatch": "Crisis",
}

CLASS_SIGNAL_MAP: dict[str, list[str]] = {
    "Control":      ["general wellbeing"],
    "Anxiety":      ["anxiety", "worry", "hypervigilance"],
    "Depression":   ["hopelessness", "low mood", "withdrawal", "fatigue"],
    "BPD":          ["emotional dysregulation", "identity distress", "abandonment fears"],
    "SuicideWatch": ["self-harm language", "crisis ideation", "existential finality"],
}

SIGNAL_LEXICONS: dict[str, list[str]] = {
    "hopelessness": [
        "no point", "pointless", "what's the point", "nothing matters",
        "can't go on", "no reason", "empty", "numb", "hollow",
        "hopeless", "hopelessness", "never get better",
    ],
    "self-harm language": [
        "kill myself", "end it", "end my life", "hurt myself",
        "cut myself", "can't take it", "want to die", "better off dead",
        "suicide", "suicidal", "overdose", "won't be here",
    ],
    "grief": [
        "passed away", "died", "loss", "lost my", "grieving",
        "miss them", "miss him", "miss her", "funeral", "gone forever",
    ],
    "isolation": [
        "alone", "lonely", "no one", "no friends", "isolated",
        "no one cares", "nobody", "by myself", "all by myself",
    ],
    "withdrawal": [
        "don't want to", "can't face", "staying in", "not going out",
        "avoiding", "hiding", "don't see the point of going",
    ],
    "anxiety": [
        "worried", "anxious", "panic", "can't stop thinking",
        "overthinking", "scared", "nervous", "fear", "dread",
    ],
    "sleep issues": [
        "can't sleep", "not sleeping", "insomnia", "wide awake",
        "up all night", "exhausted", "no energy",
    ],
    "existential finality": [
        "disappear", "not exist", "cease to exist", "never wake up",
        "wish i wasn't here", "wish i was dead", "don't want to exist",
    ],
}

THIRD_PERSON_INDICATORS: list[str] = [
    "my friend", "my sister", "my brother", "my mom", "my dad",
    "my partner", "my colleague", "he said", "she said", "they said",
    "asking for a friend", "someone i know",
]

SUPPORT_SEEKING_INDICATORS: list[str] = [
    "can someone help", "does anyone", "any advice", "what should i do",
    "how do i", "looking for support", "need help", "anyone else feel",
]

MEMORY_PATTERNS: dict[str, list[str]] = {
    "life_event": [
        "passed away", "died", "broke up", "lost my job", "moved",
        "got divorced", "was diagnosed", "had an accident", "lost my",
        "fired", "expelled", "graduated", "got married", "had a baby",
    ],
    "relationship": [
        "my mom", "my dad", "my sister", "my brother", "my partner",
        "my boyfriend", "my girlfriend", "my husband", "my wife",
        "my friend", "my therapist", "my boss", "my dog", "my cat",
    ],
    "protective_factor": [
        "helps me", "keeps me going", "i love", "looking forward to",
        "makes me happy", "my dog", "my cat", "my music", "my art",
        "my family", "working out", "journaling",
    ],
    "recurring_theme": [
        "always feel", "keep feeling", "every time", "whenever",
        "it never goes away", "i always", "this always happens",
    ],
}