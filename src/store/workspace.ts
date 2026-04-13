// src/store/workspace.ts
// Zustand is now purely in-memory UI state.
// All persistence is DB-backed via API routes.
// No more localStorage. No more persist middleware.

import { create } from "zustand"
import { getErrorMessage } from "@/lib/api"
import type {
  Patient,
  Session,
  AnalysisResult,
  MemoryCandidate,
  SavedMemory,
  InputMode,
  ContextLabel,
} from "@/types"

interface WorkspaceStore {
  // ── Active patient ────────────────────────────────────────────────────────
  activePatient: Patient | null
  setActivePatient: (patient: Patient | null) => void

  // ── Input ─────────────────────────────────────────────────────────────────
  inputText: string
  inputMode: InputMode
  setInputText: (text: string) => void
  setInputMode: (mode: InputMode) => void
  clearInput:   () => void

  // ── Analysis ──────────────────────────────────────────────────────────────
  analysisResult: AnalysisResult | null
  isAnalyzing:    boolean
  analysisError:  string | null
  runAnalysis:    () => Promise<void>
  clearAnalysis:  () => void

  // ── Memory candidates ─────────────────────────────────────────────────────
  memoryCandidates:      MemoryCandidate[]
  isAcceptingCandidate:  string | null   // candidateId currently being saved
  saveMemoryFromCandidate: (candidate: MemoryCandidate) => Promise<void>
  acceptMemoryCandidate: (index: number) => Promise<void>
  rejectMemoryCandidate: (index: number) => void
  clearMemoryCandidates: () => void

  // ── Saved memories ────────────────────────────────────────────────────────
  savedMemories:    SavedMemory[]
  isLoadingMemories: boolean
  fetchMemories:    (patientId: string) => Promise<void>
  removeMemory:     (id: string) => Promise<void>

  // ── Sessions / history ────────────────────────────────────────────────────
  sessions:                  Session[]
  isLoadingSessions:         boolean
  globalRecentSessions:      Session[]
  isLoadingGlobalSessions:   boolean
  fetchSessions:             (patientId: string) => Promise<void>
  fetchGlobalRecentSessions: () => Promise<void>
  clearSessions:             () => void
  loadSession:               (session: Session) => void

  // ── UI ────────────────────────────────────────────────────────────────────
  activeTab:    "analysis" | "memory" | "history" | "chat"
  setActiveTab: (tab: "analysis" | "memory" | "history" | "chat") => void

  // ── Reset (volatile state only) ───────────────────────────────────────────
  resetWorkspace: () => void
}

const initialVolatile = {
  inputText:            "",
  inputMode:            "paste" as InputMode,
  analysisResult:       null,
  isAnalyzing:          false,
  analysisError:        null,
  memoryCandidates:     [],
  isAcceptingCandidate: null,
  activeTab:            "analysis" as const,
}

export const useWorkspaceStore = create<WorkspaceStore>()((set, get) => ({
  // ── Initial state ─────────────────────────────────────────────────────────
  ...initialVolatile,
  activePatient:            null,
  savedMemories:            [],
  isLoadingMemories:        false,
  sessions:                 [],
  isLoadingSessions:        false,
  globalRecentSessions:     [],
  isLoadingGlobalSessions:  false,

  // ── Patient ───────────────────────────────────────────────────────────────
  setActivePatient: (patient) => set({ activePatient: patient }),

  // ── Input ─────────────────────────────────────────────────────────────────
  setInputText: (text) => set({ inputText: text }),
  setInputMode: (mode) => set({ inputMode: mode }),
  clearInput:   ()     => set({ inputText: "" }),

  // ── runAnalysis ───────────────────────────────────────────────────────────
  // Calls /api/analyze (Next.js route) which handles FastAPI + DB writes.
  // On success, prepends the new session to local history and sets candidates.
  runAnalysis: async () => {
    const { inputText, inputMode, activePatient } = get()
    if (!inputText.trim()) return
    if (!activePatient) {
      set({ analysisError: "No active patient selected." })
      return
    }

    set({
      isAnalyzing:      true,
      analysisError:    null,
      analysisResult:   null,
      memoryCandidates: [],
    })

    try {
      const res = await fetch("/api/analyze", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          text:       inputText.trim(),
          patient_id: activePatient.id,
          mode:       inputMode,
        }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body?.error ?? `Analysis failed (${res.status})`)
      }

      const data = await res.json()

      // Build AnalysisResult from response (matches existing UI type)
      const analysisResult: AnalysisResult = {
        risk_tier:      data.risk_tier,
        context_label:  data.context_label,
        signal_labels:  data.signal_labels,
        confidence:     data.confidence,
        summary:        data.summary,
        evidence_spans: data.evidence_spans,
        raw_class:      data.raw_class,
        raw_score:      data.raw_score,
      }

      // Build session record using the DB-assigned id
      const session: Session = {
        id:          data.session_id,
        patient_id:  activePatient.id,
        source_type: inputMode,
        raw_text:    inputText.trim(),
        created_at:  new Date().toISOString(),
        analysis:    analysisResult,
      }

      // Candidates now carry DB ids — needed for accept flow
      const candidates: MemoryCandidate[] = data.candidates ?? []

      set((state) => ({
        analysisResult,
        memoryCandidates: candidates,
        isAnalyzing:      false,
        activeTab:        "analysis",
        // Prepend to local session cache — newest first
        sessions: [session, ...state.sessions].slice(0, 50),
      }))

    } catch (error) {
      set({
        analysisError: getErrorMessage(error),
        isAnalyzing:   false,
      })
    }
  },

  clearAnalysis: () =>
    set({
      analysisResult:   null,
      analysisError:    null,
      memoryCandidates: [],
    }),

  // ── saveMemoryFromCandidate ───────────────────────────────────────────────
  // Reusable logic to persist a memory candidate to the database.
  saveMemoryFromCandidate: async (candidate) => {
    if (!candidate.id) {
      console.warn("[saveMemoryFromCandidate] Candidate has no DB id — skipping.")
      return
    }

    set({ isAcceptingCandidate: candidate.id })

    try {
      const res = await fetch("/api/memories", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ candidateId: candidate.id }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body?.error ?? "Failed to save memory")
      }

      const { memory } = await res.json()
      const savedMemory: SavedMemory = normaliseMemory(memory)

      set((state) => ({
        savedMemories:        [...state.savedMemories, savedMemory],
        isAcceptingCandidate: null,
      }))
    } catch (err) {
      console.error("[saveMemoryFromCandidate]", err)
      set({ isAcceptingCandidate: null })
      throw err // Rethrow to allow caller to handle UI error states
    }
  },

  // ── acceptMemoryCandidate ─────────────────────────────────────────────────
  // Used by the Analysis tab to accept a candidate from the workspace store.
  acceptMemoryCandidate: async (index) => {
    const { memoryCandidates, saveMemoryFromCandidate } = get()
    const candidate = memoryCandidates[index]
    if (!candidate) return

    try {
      await saveMemoryFromCandidate(candidate)
      // On success, remove from the local pending list
      set((state) => ({
        memoryCandidates: state.memoryCandidates.filter((_, i) => i !== index),
      }))
    } catch (err) {
      // Error is already logged in saveMemoryFromCandidate
    }
  },

  // Reject is local-only — no DB write needed.
  // The candidate row stays as "pending" in the DB (ignored unless queried).
  rejectMemoryCandidate: (index) =>
    set((state) => ({
      memoryCandidates: state.memoryCandidates.filter((_, i) => i !== index),
    })),

  clearMemoryCandidates: () => set({ memoryCandidates: [] }),

  // ── fetchMemories ─────────────────────────────────────────────────────────
  // Called when the memory tab is opened or the patient workspace loads.
  fetchMemories: async (patientId) => {
    set({ isLoadingMemories: true })
    try {
      const res = await fetch(`/api/memories?patientId=${patientId}`)
      if (!res.ok) throw new Error(`Failed to fetch memories (${res.status})`)
      const { memories } = await res.json()
      set({
        savedMemories:     memories.map(normaliseMemory),
        isLoadingMemories: false,
      })
    } catch (err) {
      console.error("[fetchMemories]", err)
      set({ isLoadingMemories: false })
    }
  },

  // ── removeMemory ──────────────────────────────────────────────────────────
  // Soft-deletes via DELETE /api/memories/:id (sets status = "archived").
  // Optimistically removes from local state first.
  removeMemory: async (id) => {
    // Optimistic update
    set((state) => ({
      savedMemories: state.savedMemories.filter((m) => m.id !== id),
    }))
    try {
      const res = await fetch(`/api/memories/${id}`, { method: "DELETE" })
      if (!res.ok) {
        // Roll back on failure
        console.error("[removeMemory] Failed — local state may be stale, refetch.")
      }
    } catch (err) {
      console.error("[removeMemory]", err)
    }
  },

  // ── fetchSessions ─────────────────────────────────────────────────────────
  // Called when the history tab is opened or the patient workspace loads.
  fetchSessions: async (patientId) => {
    set({ isLoadingSessions: true })
    try {
      const res = await fetch(`/api/sessions?patientId=${patientId}`)
      if (!res.ok) throw new Error(`Failed to fetch sessions (${res.status})`)
      const { sessions } = await res.json()
      set({
        sessions:          sessions.map(normaliseSession),
        isLoadingSessions: false,
      })
    } catch (err) {
      console.error("[fetchSessions]", err)
      set({ isLoadingSessions: false })
    }
  },

  // ── fetchGlobalRecentSessions ─────────────────────────────────────────────
  // Dashboard-specific: fetches recent sessions across all patients.
  fetchGlobalRecentSessions: async () => {
    set({ isLoadingGlobalSessions: true })
    try {
      const res = await fetch("/api/sessions")
      if (!res.ok) throw new Error("Failed to fetch global sessions")
      const { sessions } = await res.json()
      set({
        globalRecentSessions:    sessions.map(normaliseSession),
        isLoadingGlobalSessions: false,
      })
    } catch (err) {
      console.error("[fetchGlobalRecentSessions]", err)
      set({ isLoadingGlobalSessions: false })
    }
  },

  clearSessions: () => set({ sessions: [] }),

  // ── loadSession ───────────────────────────────────────────────────────────
  // Populate the workspace with a past session's transcript and analysis.
  loadSession: (session) => {
    set({
      inputText:      session.raw_text,
      analysisResult: session.analysis ?? null,
      activeTab:      "analysis",
    })
  },

  // ── UI ────────────────────────────────────────────────────────────────────
  setActiveTab: (tab) => set({ activeTab: tab }),

  // ── resetWorkspace ────────────────────────────────────────────────────────
  // Clears only volatile state. Sessions + memories are NOT cleared here —
  // they're fetched fresh from the DB when the workspace loads.
  resetWorkspace: () =>
    set({
      ...initialVolatile,
      savedMemories: [],
      sessions:      [],
    }),
}))

// ─── Normalisers — DB camelCase → UI snake_case ───────────────────────────────

function normaliseMemory(m: any): SavedMemory {
  return {
    id:                    m.id,
    patient_id:            m.patient_id,
    type:                  m.type,
    title:                 m.title,
    description:           m.description,
    therapist_verified:    m.therapist_verified,
    status:                m.status,
    first_seen_session_id: m.first_seen_session_id ?? null,
    last_seen_session_id:  m.last_seen_session_id  ?? null,
    created_at:            m.created_at,
    updated_at:            m.updated_at,
  }
}

function normaliseSession(s: any): Session {
  return {
    id:          s.id,
    patient_id:  s.patient_id,
    source_type: s.source_type,
    raw_text:    s.raw_text,
    created_at:  s.created_at,
    analysis:    s.analysis ? normaliseAnalysis(s.analysis) : undefined,
  }
}

function normaliseAnalysis(a: any): AnalysisResult {
  return {
    risk_tier:      a.risk_tier,
    context_label:  normContextLabel(a.context_label) as ContextLabel,
    signal_labels:  a.signal_labels,
    confidence:     a.confidence,
    summary:        a.summary,
    evidence_spans: (a.evidenceSpans ?? []).map((e: any) => ({
      text:      e.text,
      label:     e.label,
      score:     e.score,
      start_idx: e.start_idx,
      end_idx:   e.end_idx,
    })),
    raw_class: a.raw_class,
    raw_score: a.raw_score,
  }
}

// DB stores enum keys (self_directed), UI expects hyphenated (self-directed)
function normContextLabel(s: string): string {
  return s.replace(/_/g, "-")
}