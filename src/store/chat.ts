import { create } from "zustand"
import { runFullAnalysis, getErrorMessage } from "@/lib/api"
import type { AnalysisResult, MemoryCandidate, ContextLabel } from "@/types"

export type ChatRole = "patient" | "therapist"

export interface ChatMessage {
  id:        string
  role:      ChatRole
  text:      string
  timestamp: string
  flagged?:  boolean   // true when sent during a High/Crisis analysis
}

interface ChatStore {
  messages:        ChatMessage[]
  inputMode:       ChatRole
  autoAnalyze:     boolean
  analyzeCount:    number

  // Analysis state
  activeSessionId: string | null
  chatAnalysis:    AnalysisResult | null
  chatCandidates:  MemoryCandidate[]
  isAnalyzing:     boolean
  analysisError:   string | null

  // Actions
  addMessage:      (role: ChatRole, text: string) => void
  setInputMode:    (mode: ChatRole) => void
  toggleAutoAnalyze: () => void
  runChatAnalysis: (patientId?: string) => Promise<void>
  clearChat:       () => void
}

const buildText = (messages: ChatMessage[]) =>
  messages
    .filter((m) => m.role === "patient")
    .map((m) => m.text)
    .join("\n")

export const useChatStore = create<ChatStore>()((set, get) => ({
  messages:       [],
  inputMode:      "patient",
  autoAnalyze:    true,
  analyzeCount:   0,
  activeSessionId: null,
  chatAnalysis:   null,
  chatCandidates: [],
  isAnalyzing:    false,
  analysisError:  null,

  addMessage: (role, text) => {
    const message: ChatMessage = {
      id:        crypto.randomUUID(),
      role,
      text,
      timestamp: new Date().toISOString(),
    }
    set((s) => ({ messages: [...s.messages, message] }))
  },

  setInputMode:      (mode)  => set({ inputMode: mode }),
  toggleAutoAnalyze: ()      => set((s) => ({ autoAnalyze: !s.autoAnalyze })),

  runChatAnalysis: async (patientId) => {
    const text = buildText(get().messages)
    if (!text.trim()) return

    set({ isAnalyzing: true, analysisError: null })

    try {
      const data = await runFullAnalysis({
        text,
        patient_id: patientId || "",
        session_id: get().activeSessionId || undefined,
        mode: "chat"
      })

      // Flag the most recent patient message if High/Crisis
      const isFlagged = data.risk_tier === "High" || data.risk_tier === "Crisis"
      if (isFlagged) {
        set((s) => {
          const msgs = [...s.messages]
          // find last patient message
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === "patient") { msgs[i] = { ...msgs[i], flagged: true }; break }
          }
          return { messages: msgs }
        })
      }

      set((s) => ({
        chatAnalysis:   {
          risk_tier: data.risk_tier,
          context_label: data.context_label as ContextLabel,
          signal_labels: data.signal_labels,
          confidence: data.confidence,
          summary: data.summary,
          evidence_spans: data.evidence_spans,
          raw_class: data.raw_class,
          raw_score: data.raw_score,
        },
        chatCandidates: data.candidates,
        isAnalyzing:    false,
        analyzeCount:   s.analyzeCount + 1,
        activeSessionId: data.session_id,
      }))
    } catch (err) {
      set({ analysisError: getErrorMessage(err), isAnalyzing: false })
    }
  },

  clearChat: () =>
    set({
      messages:       [],
      activeSessionId: null,
      chatAnalysis:   null,
      chatCandidates: [],
      isAnalyzing:    false,
      analysisError:  null,
      analyzeCount:   0,
    }),
}))