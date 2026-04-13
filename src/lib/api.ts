// src/lib/api.ts

const BASE_URL   = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const TIMEOUT_MS = 30_000

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function apiFetch<T>(
  path:    string,
  options: RequestInit = {},
  base = BASE_URL
): Promise<T> {
  const controller = new AbortController()
  const timer      = setTimeout(() => controller.abort(), TIMEOUT_MS)

  try {
    const res = await fetch(`${base}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal:  controller.signal,
      ...options,
    })

    if (!res.ok) {
      let message = `HTTP ${res.status}`
      try {
        const body = await res.json()
        message = body?.detail ?? body?.error ?? message
      } catch { /* keep status message */ }
      throw new ApiError(message, res.status)
    }

    return res.json() as Promise<T>
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("Request timed out. The model may still be loading.", 408)
    }
    throw new ApiError(
      err instanceof Error ? err.message : "Network error",
      0
    )
  } finally {
    clearTimeout(timer)
  }
}

export class ApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message)
    this.name = "ApiError"
  }
}

export function getErrorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message
  if (err instanceof Error)    return err.message
  return "An unexpected error occurred."
}

// ─── Types ────────────────────────────────────────────────────────────────────

export type RiskTier       = "Low" | "Moderate" | "High" | "Crisis"
export type ContextLabel   = "self-directed" | "third-person" | "support-seeking" | "ambiguous"
export type ConfidenceLevel = "strong" | "medium" | "cautious"
export type MemoryType     = "life_event" | "relationship" | "recurring_theme" | "protective_factor"
export type InputMode      = "paste" | "chat"

export interface EvidenceSpan {
  text:      string
  label:     string
  score:     number
  start_idx: number | null
  end_idx:   number | null
}

export interface MemoryCandidate {
  id?:         string   // present when returned from DB
  type:        MemoryType
  title:       string
  description: string
  confidence:  number
}

export interface AnalyzeResponse {
  // from FastAPI model
  risk_tier:      RiskTier
  context_label:  ContextLabel
  signal_labels:  string[]
  confidence:     ConfidenceLevel
  summary:        string
  evidence_spans: EvidenceSpan[]
  raw_class:      string
  raw_score:      number
  // added by Next.js route after DB write
  session_id:     string
  analysis_id:    string
  candidates:     MemoryCandidate[]
}

// ─── /health (still hits FastAPI directly — used by ApiStatus component) ──────

export interface HealthResponse {
  status:       string
  model_loaded: boolean
  version:      string
}

export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health")
}

// ─── runFullAnalysis — now calls Next.js route, not FastAPI directly ──────────

export interface RunAnalysisRequest {
  text:       string
  patient_id: string
  session_id?: string
  mode?:      InputMode
}

export async function runFullAnalysis(
  req: RunAnalysisRequest
): Promise<AnalyzeResponse> {
  // "/api/analyze" is a relative path — hits the Next.js route handler.
  // That route calls FastAPI internally and persists everything to DB.
  return apiFetch<AnalyzeResponse>(
    "/api/analyze",
    {
      method: "POST",
      body:   JSON.stringify(req),
    },
    "" // empty base so it resolves relative to the Next.js origin
  )
}