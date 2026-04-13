import pickle
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import (
    settings,
    RISK_TIER_MAP,
    CLASS_SIGNAL_MAP,
)
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ContextLabel,
    RiskTier,
    ConfidenceLevel,
    EvidenceSpan,
)
from app.utils.text_cleaning import clean_text, is_too_short, get_display_text
from app.services.explanation_service import extract_evidence_spans
from app.services.context_service import detect_context


# ─── Model state ──────────────────────────────────────────────────────────────
# Loaded once at startup, reused across all requests

_model: AutoModelForSequenceClassification | None = None
_tokenizer: AutoTokenizer | None = None
_label_encoder = None  # sklearn LabelEncoder
_id2label: dict[int, str] = {}


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_model() -> None:
    global _model, _tokenizer, _id2label

    model_dir = settings.model_dir  # already a string

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Copy your saved model files into models/classifier/"
        )

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _model.to(settings.device)
    _model.eval()

    # Read labels directly from config.json — no pkl needed
    _id2label = {
        int(k): v
        for k, v in _model.config.id2label.items()
    }

    print(f"[model_service] Model loaded from {model_dir}")
    print(f"[model_service] Classes: {list(_id2label.values())}")
    print(f"[model_service] Device: {settings.device}")


def is_model_loaded() -> bool:
    return _model is not None and _tokenizer is not None


# ─── Inference ────────────────────────────────────────────────────────────────

def _run_inference(text: str) -> tuple[str, float, np.ndarray]:
    """
    Run tokenization and forward pass.

    Returns:
        predicted_class: str
        confidence_score: float (max softmax probability)
        all_probs: np.ndarray of shape (num_classes,)
    """
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=settings.max_token_length,
    )

    inputs = {k: v.to(settings.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    all_probs = probs.cpu().numpy()
    predicted_id = int(np.argmax(all_probs))
    confidence_score = float(all_probs[predicted_id])
    predicted_class = _id2label[predicted_id]

    return predicted_class, confidence_score, all_probs


# ─── Confidence bucketing ─────────────────────────────────────────────────────

def _get_confidence_level(score: float) -> ConfidenceLevel:
    if score >= settings.confidence_threshold_strong:
        return ConfidenceLevel.STRONG
    elif score >= settings.confidence_threshold_medium:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.CAUTIOUS


# ─── Summary generation ───────────────────────────────────────────────────────

def _build_summary(
    predicted_class: str,
    risk_tier: str,
    signals: list[str],
    context: ContextLabel,
    confidence: ConfidenceLevel,
) -> str:
    """
    Build a short neutral clinical summary string.
    No diagnosis language — signals and patterns only.
    """
    signal_str = ", ".join(signals) if signals else "general distress markers"
    context_str = context.value.replace("-", " ")

    confidence_note = {
        ConfidenceLevel.STRONG:   "The model returned a high-confidence assessment.",
        ConfidenceLevel.MEDIUM:   "The model returned a moderate-confidence assessment.",
        ConfidenceLevel.CAUTIOUS: "The model is uncertain — therapist judgment should take priority.",
    }[confidence]

    return (
        f"Text analysis suggests {risk_tier.lower()} risk. "
        f"Language is {context_str} in nature. "
        f"Detected signal patterns include: {signal_str}. "
        f"{confidence_note}"
    )


# ─── Main service function ────────────────────────────────────────────────────

def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Full analysis pipeline for a single text input.

    Steps:
        1. Validate and clean text
        2. Run model inference
        3. Map to risk tier and signals
        4. Detect context label
        5. Extract evidence spans
        6. Build and return structured response
    """
    if not is_model_loaded():
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    # 1. Clean
    cleaned = clean_text(request.text, for_model=True)

    if is_too_short(cleaned):
        # Return a safe low-confidence response rather than an error
        return AnalyzeResponse(
            risk_tier=RiskTier.LOW,
            context_label=ContextLabel.AMBIGUOUS,
            signal_labels=["insufficient text"],
            confidence=ConfidenceLevel.CAUTIOUS,
            summary="Text is too short for a meaningful analysis. Please provide more context.",
            evidence_spans=[],
            raw_class="unknown",
            raw_score=0.0,
        )

    # 2. Inference
    predicted_class, raw_score, all_probs = _run_inference(cleaned)

    # 3. Map outputs
    risk_tier_str = RISK_TIER_MAP.get(predicted_class, "Low")
    risk_tier = RiskTier(risk_tier_str)
    signal_labels = CLASS_SIGNAL_MAP.get(predicted_class, ["general distress"])
    confidence = _get_confidence_level(raw_score)

    # 4. Context detection runs on ORIGINAL text (not model-cleaned)
    # because third-person pronouns and support-seeking phrases
    # are easier to catch before contraction expansion alters phrasing
    context_label = detect_context(request.text)

    # 5. Evidence spans from display text so character indices
    # line up with what the frontend renders
    display_text = get_display_text(request.text)
    evidence_spans = extract_evidence_spans(display_text)

    # 6. Summary
    summary = _build_summary(
        predicted_class,
        risk_tier_str,
        signal_labels,
        context_label,
        confidence,
    )

    return AnalyzeResponse(
        risk_tier=risk_tier,
        context_label=context_label,
        signal_labels=signal_labels,
        confidence=confidence,
        summary=summary,
        evidence_spans=evidence_spans,
        raw_class=predicted_class,
        raw_score=round(raw_score, 4),
    )