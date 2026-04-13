from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class RiskTier(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRISIS = "Crisis"


class ContextLabel(str, Enum):
    SELF_DIRECTED = "self-directed"
    THIRD_PERSON = "third-person"
    SUPPORT_SEEKING = "support-seeking"
    AMBIGUOUS = "ambiguous"


class ConfidenceLevel(str, Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    CAUTIOUS = "cautious"


class MemoryType(str, Enum):
    LIFE_EVENT = "life_event"
    RELATIONSHIP = "relationship"
    RECURRING_THEME = "recurring_theme"
    PROTECTIVE_FACTOR = "protective_factor"


class MemoryCandidateStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class InputMode(str, Enum):
    PASTE = "paste"
    CHAT = "chat"


# ─── Sub-models ───────────────────────────────────────────────────────────────

class EvidenceSpan(BaseModel):
    text: str = Field(..., description="The exact phrase from the input text")
    label: str = Field(..., description="Signal category e.g. hopelessness, grief")
    score: float = Field(..., ge=0.0, le=1.0, description="Importance score 0-1")
    start_idx: Optional[int] = Field(None, description="Character start index in source text")
    end_idx: Optional[int] = Field(None, description="Character end index in source text")


class MemoryCandidate(BaseModel):
    type: MemoryType
    title: str = Field(..., description="Short label e.g. 'Dog passed away'")
    description: str = Field(..., description="Full extracted context")
    confidence: float = Field(..., ge=0.0, le=1.0)


# ─── Analyze ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Patient text to analyze")
    patient_id: Optional[str] = Field(None, description="Patient ID for memory context")
    mode: InputMode = Field(InputMode.PASTE, description="Input mode: paste or chat")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I don't really see the point of anything lately. I just feel empty at home.",
                "patient_id": "patient_001",
                "mode": "paste"
            }
        }
    }


class AnalyzeResponse(BaseModel):
    risk_tier: RiskTier
    context_label: ContextLabel
    signal_labels: list[str] = Field(..., description="Detected signal categories")
    confidence: ConfidenceLevel
    summary: str = Field(..., description="Short neutral clinical summary")
    evidence_spans: list[EvidenceSpan]
    raw_class: str = Field(..., description="Direct model output class")
    raw_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")


# ─── Memory extraction ────────────────────────────────────────────────────────

class MemoryRequest(BaseModel):
    text: str = Field(..., min_length=5)
    patient_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "My dog passed away last week. He was 12 years old. I live alone so it hit hard.",
                "patient_id": "patient_001"
            }
        }
    }


class MemoryResponse(BaseModel):
    candidates: list[MemoryCandidate]
    count: int


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

    model_config = {"protected_namespaces": ()}