from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    MemoryRequest,
    MemoryResponse,
    HealthResponse,
)
from app.services.model_service import (
    load_model,
    analyze_text,
    is_model_loaded,
)
from app.services.memory_service import extract_memory_candidates


# ─── Lifespan ─────────────────────────────────────────────────────────────────
# Load model once at startup, release at shutdown.
# Using lifespan instead of deprecated @app.on_event("startup")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    print("[startup] Loading model...")
    try:
        load_model()
        print("[startup] Model ready.")
    except FileNotFoundError as e:
        # Don't crash the server if model files aren't present yet.
        # The /health endpoint will report model_loaded: false so you
        # can still test routes while setting up model artifacts.
        print(f"[startup] WARNING: {e}")
        print("[startup] Server starting without model. /analyze will return 503.")

    yield

    # ── Shutdown ──
    print("[shutdown] Cleaning up.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Therapist-side clinical decision support API. "
        "Provides explainable risk analysis and memory extraction "
        "for text-based patient interactions."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow your Next.js frontend to call this API during local development.

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Exception handlers ───────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Return clean validation errors instead of FastAPI's default 422 blob."""
    errors = exc.errors()
    messages = [
        f"{' → '.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in errors
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "detail": messages,
        },
    )


@app.exception_handler(RuntimeError)
async def runtime_exception_handler(
    request: Request,
    exc: RuntimeError,
):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Service unavailable",
            "detail": str(exc),
        },
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health() -> HealthResponse:
    """
    Returns service status and whether the model is loaded.
    Use this to verify the API is ready before sending analyze requests.
    """
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        version=settings.app_version,
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze patient text",
    tags=["Inference"],
    responses={
        200: {"description": "Analysis complete"},
        422: {"description": "Validation error — check request body"},
        503: {"description": "Model not loaded"},
    },
)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Core endpoint. Accepts patient text and returns:

    - **risk_tier** — Low / Moderate / High / Crisis
    - **context_label** — self-directed / third-person / support-seeking / ambiguous
    - **signal_labels** — detected clinical signal categories
    - **confidence** — strong / medium / cautious
    - **summary** — short neutral clinical summary
    - **evidence_spans** — highlighted phrases with positions and labels
    - **raw_class** — direct model output class
    - **raw_score** — model confidence score (0–1)

    The `patient_id` field is optional. When provided, it is passed
    through for the frontend to associate results with a patient record.
    The API itself is stateless — persistence is handled by the Next.js
    app layer.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded. "
                "Check that model artifacts exist in /models/classifier "
                "and restart the server."
            ),
        )

    try:
        return analyze_text(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@app.post(
    "/extract-memory",
    response_model=MemoryResponse,
    summary="Extract memory candidates from text",
    tags=["Memory"],
    responses={
        200: {"description": "Extraction complete"},
        422: {"description": "Validation error — check request body"},
    },
)
def extract_memory(request: MemoryRequest) -> MemoryResponse:
    """
    Extracts structured memory candidates from patient text.

    Returns a list of candidates for therapist review — not saved
    automatically. The therapist accepts, edits, or rejects each
    candidate from the frontend before anything is persisted.

    Memory types:
    - **life_event** — bereavement, job loss, breakup, diagnosis, etc.
    - **relationship** — key people in the patient's life
    - **protective_factor** — things that help or ground the patient
    - **recurring_theme** — persistent emotional patterns
    """
    try:
        return extract_memory_candidates(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory extraction failed: {str(e)}",
        )