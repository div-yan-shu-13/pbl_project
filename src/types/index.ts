// ─── Enums ────────────────────────────────────────────────────────────────────

export type RiskTier = "Low" | "Moderate" | "High" | "Crisis"

export type ContextLabel =
  | "self-directed"
  | "third-person"
  | "support-seeking"
  | "ambiguous"

export type ConfidenceLevel = "strong" | "medium" | "cautious"

export type MemoryType =
  | "life_event"
  | "relationship"
  | "recurring_theme"
  | "protective_factor"

export type MemoryCandidateStatus = "pending" | "accepted" | "rejected"

export type InputMode = "paste" | "chat"


// ─── API request types ────────────────────────────────────────────────────────

export interface AnalyzeRequest {
  text: string
  patient_id?: string
  session_id?: string
  mode: InputMode
}

export interface MemoryRequest {
  text: string
  patient_id?: string
}


// ─── API response types ───────────────────────────────────────────────────────

export interface EvidenceSpan {
  text: string
  label: string
  score: number
  start_idx: number | null
  end_idx: number | null
}

export interface AnalysisResult {
  risk_tier: RiskTier
  context_label: ContextLabel
  signal_labels: string[]
  confidence: ConfidenceLevel
  summary: string
  evidence_spans: EvidenceSpan[]
  raw_class: string
  raw_score: number
}

export interface MemoryCandidate {
  id?: string
  patient_id?: string
  session_id?: string
  type: MemoryType
  title: string
  description: string
  confidence: number
  status?: string
}

export interface MemoryResponse {
  candidates: MemoryCandidate[]
  count: number
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  version: string
}


// ─── App-level types (not from API) ──────────────────────────────────────────

export interface Patient {
  id: string
  display_name: string
  notes?: string
  created_at: string
  last_session?: string
  last_risk_tier?: RiskTier
}

export interface Session {
  id: string
  patient_id: string
  source_type: InputMode
  raw_text: string
  created_at: string
  analysis?: AnalysisResult
  therapist_notes?: string
}

export interface SavedMemory {
  id: string
  patient_id: string
  type: MemoryType
  title: string
  description: string
  first_seen_session_id: string
  last_seen_session_id: string
  therapist_verified: boolean
  status: "active" | "resolved" | "archived"
  created_at: string
  updated_at: string
}


// ─── UI state types ───────────────────────────────────────────────────────────

export interface WorkspaceState {
  // Current patient
  activePatient: Patient | null

  // Current input
  inputText: string
  inputMode: InputMode

  // Analysis
  analysisResult: AnalysisResult | null
  isAnalyzing: boolean
  analysisError: string | null

  // Memory
  memoryCandidates: MemoryCandidate[]
  savedMemories: SavedMemory[]
  isExtractingMemory: boolean

  // Session history
  sessions: Session[]
}


// ─── Helpers ──────────────────────────────────────────────────────────────────

export const RISK_TIER_CONFIG: Record<
  RiskTier,
  { label: string; color: string; bg: string; border: string }
> = {
  Low: {
    label: "Low Risk",
    color: "text-emerald-700 dark:text-emerald-400",
    bg: "bg-emerald-50 dark:bg-emerald-950/40",
    border: "border-emerald-200 dark:border-emerald-800",
  },
  Moderate: {
    label: "Moderate Risk",
    color: "text-amber-700 dark:text-amber-400",
    bg: "bg-amber-50 dark:bg-amber-950/40",
    border: "border-amber-200 dark:border-amber-800",
  },
  High: {
    label: "High Risk",
    color: "text-orange-700 dark:text-orange-400",
    bg: "bg-orange-50 dark:bg-orange-950/40",
    border: "border-orange-200 dark:border-orange-800",
  },
  Crisis: {
    label: "Crisis",
    color: "text-red-700 dark:text-red-400",
    bg: "bg-red-50 dark:bg-red-950/40",
    border: "border-red-200 dark:border-red-800",
  },
}

export const CONTEXT_LABEL_CONFIG: Record<
  ContextLabel,
  { label: string; description: string }
> = {
  "self-directed": {
    label: "Self-directed",
    description: "Language appears to be about the patient's own experience.",
  },
  "third-person": {
    label: "Third-person",
    description: "Crisis vocabulary may relate to another person, not the patient.",
  },
  "support-seeking": {
    label: "Support-seeking",
    description: "Text is framed as a request for help or advice.",
  },
  ambiguous: {
    label: "Ambiguous",
    description: "Context could not be determined — assess perspective directly.",
  },
}

export const MEMORY_TYPE_CONFIG: Record<
  MemoryType,
  { label: string; icon: string; color: string }
> = {
  life_event:        { label: "Life Event",        icon: "calendar", color: "var(--color-blue)" },
  relationship:      { label: "Relationship",      icon: "users",    color: "var(--color-primary)" },
  recurring_theme:   { label: "Recurring Theme",   icon: "repeat",   color: "var(--color-gold)" },
  protective_factor: { label: "Protective Factor", icon: "shield",   color: "var(--color-success)" },
}

export const CONFIDENCE_CONFIG: Record<
  ConfidenceLevel,
  { label: string; description: string }
> = {
  strong:   { label: "Strong",   description: "High-confidence assessment." },
  medium:   { label: "Medium",   description: "Moderate-confidence assessment." },
  cautious: { label: "Cautious", description: "Low confidence — therapist judgment should take priority." },
}