"use client"

import { useEffect, useRef } from "react"
import { use } from "react"
import { useWorkspaceStore } from "@/store/workspace"
import { usePatientsStore } from "@/store/patients"
import { RISK_TIER_CONFIG, CONTEXT_LABEL_CONFIG, MEMORY_TYPE_CONFIG, type RiskTier } from "@/types"
import { cn, wordCount } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { ThemeToggle } from "@/components/theme-toggle"
import { ApiStatus } from "@/components/api-status"
import { ChatPanel } from "@/components/chat-panel"
import { ChatInsights } from "@/components/chat-insights"
import {
  Brain, Users, ChevronLeft, Send, Loader2,
  AlertTriangle, ShieldCheck, Info, Check, X,
  Calendar, Repeat, Shield, Clock, Trash2, MessageSquare,
} from "lucide-react"
import Link from "next/link"

const MEMORY_ICONS = {
  life_event: Calendar,
  relationship: Users,
  recurring_theme: Repeat,
  protective_factor: Shield,
}

// ─── Risk tier banner ─────────────────────────────────────────────────────────

function RiskBanner({ tier }: { tier: RiskTier }) {
  if (tier === "Low" || tier === "Moderate") return null
  const isCrisis = tier === "Crisis"
  const cfg = RISK_TIER_CONFIG[tier]
  return (
    <div className={cn(
      "flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm font-medium",
      cfg.bg, cfg.border, cfg.color
    )}>
      <AlertTriangle className="w-4 h-4 shrink-0" />
      <span>
        {isCrisis
          ? "Crisis-level risk detected. Immediate review recommended."
          : "High-risk indicators present. Consider direct follow-up."}
      </span>
    </div>
  )
}

// ─── Evidence text ────────────────────────────────────────────────────────────

function EvidenceText({
  text,
  spans,
}: {
  text: string
  spans: { text: string; label: string; start_idx: number | null; end_idx: number | null }[]
}) {
  const indexedSpans = spans.filter((s) => s.start_idx !== null && s.end_idx !== null)

  if (!indexedSpans.length) {
    return (
      <p className="text-sm text-[var(--color-text)] leading-relaxed whitespace-pre-wrap">
        {text}
      </p>
    )
  }

  const sorted = [...indexedSpans].sort((a, b) => a.start_idx! - b.start_idx!)
  const parts: React.ReactNode[] = []
  let cursor = 0

  sorted.forEach((span, i) => {
    const start = span.start_idx!
    const end = span.end_idx!
    if (start > cursor) {
      parts.push(<span key={`plain-${i}`}>{text.slice(cursor, start)}</span>)
    }
    parts.push(
      <TooltipProvider key={`span-${i}`}>
        <Tooltip>
          <TooltipTrigger asChild>
            <mark className="evidence-highlight" data-label={span.label}>
              {text.slice(start, end)}
            </mark>
          </TooltipTrigger>
          <TooltipContent side="top" className="text-xs max-w-[200px]">
            {span.label}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
    cursor = end
  })

  if (cursor < text.length) {
    parts.push(<span key="plain-end">{text.slice(cursor)}</span>)
  }

  return (
    <p className="text-sm text-[var(--color-text)] leading-relaxed whitespace-pre-wrap">
      {parts}
    </p>
  )
}

// ─── Analysis panel ───────────────────────────────────────────────────────────

function AnalysisPanel() {
  const { analysisResult, isAnalyzing, analysisError, inputText } = useWorkspaceStore()

  if (isAnalyzing) {
    return (
      <div className="space-y-4 p-4">
        <div className="skeleton h-10 w-full rounded-lg" />
        <div className="space-y-2">
          <div className="skeleton h-4 w-24 rounded" />
          <div className="skeleton h-4 w-full rounded" />
          <div className="skeleton h-4 w-5/6 rounded" />
        </div>
        <div className="flex gap-2">
          <div className="skeleton h-6 w-16 rounded-full" />
          <div className="skeleton h-6 w-20 rounded-full" />
          <div className="skeleton h-6 w-14 rounded-full" />
        </div>
        <div className="space-y-2">
          <div className="skeleton h-3 w-20 rounded" />
          {[1, 2, 3].map((i) => (
            <div key={i} className="skeleton h-8 w-full rounded-md" />
          ))}
        </div>
      </div>
    )
  }

  if (analysisError) {
    return (
      <div className="m-4 px-4 py-3 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800">
        <p className="text-sm text-red-600 dark:text-red-400 font-medium">Analysis failed</p>
        <p className="text-xs text-red-500 dark:text-red-400 mt-1">{analysisError}</p>
      </div>
    )
  }

  if (!analysisResult) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[240px] text-center px-6 py-12">
        <Brain className="w-8 h-8 text-[var(--color-text-faint)] mb-3" />
        <p className="text-sm font-medium text-[var(--color-text-muted)]">No analysis yet</p>
        <p className="text-xs text-[var(--color-text-faint)] mt-1 max-w-[200px]">
          Paste a session note or message and click Analyze.
        </p>
      </div>
    )
  }

  const cfg = RISK_TIER_CONFIG[analysisResult.risk_tier]
  const ctxConfig = CONTEXT_LABEL_CONFIG[analysisResult.context_label]

  return (
    <div className="p-4 space-y-4">
      <RiskBanner tier={analysisResult.risk_tier} />

      <div className={cn(
        "flex items-center justify-between px-4 py-3 rounded-lg border",
        cfg.bg, cfg.border
      )}>
        <div>
          <p className={cn("text-base font-bold", cfg.color)}>{cfg.label}</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-0.5 capitalize">
            {analysisResult.confidence} confidence · {analysisResult.raw_class}
          </p>
        </div>
        <span className="text-xs font-mono text-[var(--color-text-faint)]">
          {analysisResult.raw_score.toFixed(3)}
        </span>
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1.5">
          <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">
            Context
          </p>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Info className="w-3 h-3 text-[var(--color-text-faint)]" />
              </TooltipTrigger>
              <TooltipContent className="text-xs max-w-[200px]">
                {ctxConfig.description}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Badge variant="outline" className="text-xs capitalize">
          {ctxConfig.label}
        </Badge>
      </div>

      <Separator className="bg-[var(--color-divider)]" />

      {analysisResult.signal_labels.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">
            Signals Detected
          </p>
          <div className="flex flex-wrap gap-1.5">
            {analysisResult.signal_labels.map((sig) => (
              <span
                key={sig}
                className="text-xs px-2 py-0.5 rounded-full bg-[var(--color-surface-offset)] border border-[var(--color-border)] text-[var(--color-text-muted)]"
              >
                {sig}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-1.5">
        <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">
          Summary
        </p>
        <p className="text-sm text-[var(--color-text)] leading-relaxed">
          {analysisResult.summary}
        </p>
      </div>

      <Separator className="bg-[var(--color-divider)]" />

      {analysisResult.evidence_spans.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">
            Evidence in Text
          </p>
          <div className="px-3 py-2.5 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)] max-h-[180px] overflow-y-auto custom-scrollbar">
            <EvidenceText text={inputText} spans={analysisResult.evidence_spans} />
          </div>
          <div className="space-y-1">
            {analysisResult.evidence_spans.map((span, i) => (
              <div
                key={i}
                className="flex items-start gap-2 text-xs px-2 py-1.5 rounded-md bg-[var(--color-surface-offset)]"
              >
                <span className="text-[var(--color-text-faint)] font-mono mt-0.5">{i + 1}.</span>
                <div>
                  <span className="font-medium text-[var(--color-text)]">"{span.text}"</span>
                  <span className="text-[var(--color-text-faint)] ml-1.5">{span.label}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Memory panel ─────────────────────────────────────────────────────────────

function MemoryPanel() {
  const {
    memoryCandidates,
    savedMemories,
    isLoadingMemories,
    isAcceptingCandidate,
    acceptMemoryCandidate,
    rejectMemoryCandidate,
    removeMemory,
  } = useWorkspaceStore()

  if (isLoadingMemories) {
    return (
      <div className="p-4 space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton h-16 w-full rounded-lg" />
        ))}
      </div>
    )
  }

  return (
    <div className="p-4 space-y-5">

      {memoryCandidates.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">
            Suggested ({memoryCandidates.length})
          </p>
          <p className="text-xs text-[var(--color-text-faint)]">
            Review and accept relevant memories to save them.
          </p>
          <div className="space-y-2">
            {memoryCandidates.map((c, i) => {
              const Icon = MEMORY_ICONS[c.type]
              const isAccepting = isAcceptingCandidate === c.id
              return (
                <div key={c.id ?? i} className="panel-card px-3 py-2.5 space-y-1.5">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <Icon className="w-3.5 h-3.5 text-[var(--color-primary)] shrink-0" />
                      <span className="text-xs font-medium truncate">{c.title}</span>
                    </div>
                    <div className="flex gap-1 shrink-0">
                      <button
                        onClick={() => acceptMemoryCandidate(i)}
                        disabled={!!isAcceptingCandidate}
                        className="p-1 rounded-md hover:bg-emerald-100 dark:hover:bg-emerald-950/40 text-emerald-600 transition-colors disabled:opacity-50"
                        aria-label="Accept memory"
                      >
                        {isAccepting
                          ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          : <Check className="w-3.5 h-3.5" />
                        }
                      </button>
                      <button
                        onClick={() => rejectMemoryCandidate(i)}
                        disabled={!!isAcceptingCandidate}
                        className="p-1 rounded-md hover:bg-red-100 dark:hover:bg-red-950/40 text-red-500 transition-colors disabled:opacity-50"
                        aria-label="Reject memory"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                  <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">
                    {c.description}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-[var(--color-text-faint)] capitalize">
                      {MEMORY_TYPE_CONFIG[c.type].label}
                    </span>
                    <span className="text-xs font-mono text-[var(--color-text-faint)]">
                      {(c.confidence * 100).toFixed(0)}% conf.
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {memoryCandidates.length > 0 && savedMemories.length > 0 && (
        <Separator className="bg-[var(--color-divider)]" />
      )}

      {savedMemories.length > 0 ? (
        <div className="space-y-2">
          <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">
            Saved ({savedMemories.length})
          </p>
          <div className="space-y-1.5">
            {savedMemories.map((m) => {
              const Icon = MEMORY_ICONS[m.type]
              return (
                <div
                  key={m.id}
                  className="flex items-start gap-2 px-3 py-2 rounded-md bg-[var(--color-surface-offset)] group"
                >
                  <Icon className="w-3.5 h-3.5 text-[var(--color-primary)] mt-0.5 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-medium truncate">{m.title}</span>
                      {m.therapist_verified && (
                        <ShieldCheck className="w-3 h-3 text-emerald-500 shrink-0" />
                      )}
                    </div>
                    <p className="text-xs text-[var(--color-text-muted)] mt-0.5 leading-relaxed">
                      {m.description}
                    </p>
                  </div>
                  <button
                    onClick={() => removeMemory(m.id)}
                    className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-[var(--color-border)] text-[var(--color-text-faint)] transition-all"
                    aria-label="Remove memory"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              )
            })}
          </div>
        </div>
      ) : memoryCandidates.length === 0 && (
        <div className="flex flex-col items-center justify-center min-h-[200px] text-center px-6">
          <Shield className="w-7 h-7 text-[var(--color-text-faint)] mb-2" />
          <p className="text-sm font-medium text-[var(--color-text-muted)]">No memories yet</p>
          <p className="text-xs text-[var(--color-text-faint)] mt-1 max-w-[200px]">
            After analysis, suggested memories will appear here for review.
          </p>
        </div>
      )}
    </div>
  )
}

// ─── History panel ────────────────────────────────────────────────────────────

function HistoryPanel() {
  const { sessions, isLoadingSessions, loadSession } = useWorkspaceStore()

  if (isLoadingSessions) {
    return (
      <div className="p-4 space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton h-20 w-full rounded-lg" />
        ))}
      </div>
    )
  }

  if (sessions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[200px] text-center px-6 py-12">
        <Clock className="w-7 h-7 text-[var(--color-text-faint)] mb-2" />
        <p className="text-sm font-medium text-[var(--color-text-muted)]">No sessions yet</p>
        <p className="text-xs text-[var(--color-text-faint)] mt-1 max-w-[200px]">
          Each analysis you run will be saved here automatically.
        </p>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-2">
      <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">
        Sessions ({sessions.length})
      </p>
      {sessions.map((session) => {
        const cfg = session.analysis ? RISK_TIER_CONFIG[session.analysis.risk_tier] : null
        const date = new Date(session.created_at)
        const timeStr = date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })
        const dateStr = date.toLocaleDateString("en-US", { month: "short", day: "numeric" })

        return (
          <button
            key={session.id}
            onClick={() => loadSession(session)}
            className="panel-card w-full text-left px-3 py-2.5 space-y-1.5 hover:ring-2 hover:ring-[var(--color-primary)] transition-all group"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--color-text-faint)] group-hover:text-[var(--color-text-muted)] transition-colors">
                {dateStr} · {timeStr}
              </span>
              {cfg && (
                <span className={cn(
                  "text-xs px-2 py-0.5 rounded-full border font-medium",
                  cfg.bg, cfg.color, cfg.border
                )}>
                  {cfg.label}
                </span>
              )}
            </div>
            <p className="text-xs text-[var(--color-text-muted)] line-clamp-2 leading-relaxed">
              {session.raw_text}
            </p>
            {session.analysis && (
              <div className="flex gap-1.5 flex-wrap">
                {session.analysis.signal_labels.slice(0, 3).map((sig) => (
                  <span
                    key={sig}
                    className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--color-surface-offset)] border border-[var(--color-border)] text-[var(--color-text-faint)]"
                  >
                    {sig}
                  </span>
                ))}
              </div>
            )}
          </button>
        )
      })}
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function PatientWorkspacePage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = use(params)
  const { patients, getPatient, fetchPatients, isLoading: isPatientsLoading } = usePatientsStore()
  const patient = getPatient(id)

  const {
    inputText,
    setInputText,
    isAnalyzing,
    analysisResult,
    memoryCandidates,
    activeTab,
    setActiveTab,
    runAnalysis,
    resetWorkspace,
    setActivePatient,
    fetchSessions,
    fetchMemories,
    loadSession,
    sessions,
    isLoadingSessions,
  } = useWorkspaceStore()

  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (patients.length === 0) {
      fetchPatients()
    }
  }, [fetchPatients, patients.length])

  useEffect(() => {
    if (patient) {
      resetWorkspace()
      setActivePatient(patient)
      fetchSessions(id)
      fetchMemories(id)
      if (activeTab !== "chat") setTimeout(() => textareaRef.current?.focus(), 0)
    }
  }, [id, patient?.id])

  if (isPatientsLoading && !patient) {
    return (
      <div className="min-h-dvh flex items-center justify-center bg-[var(--color-bg)]">
        <Loader2 className="w-6 h-6 animate-spin text-[var(--color-primary)]" />
      </div>
    )
  }

  if (!patient && !isPatientsLoading) {
    return (
      <div className="min-h-dvh flex items-center justify-center bg-[var(--color-bg)]">
        <div className="text-center space-y-3">
          <p className="text-sm font-medium text-[var(--color-text-muted)]">Patient not found</p>
          <Link href="/" className="text-xs text-[var(--color-primary)] hover:underline">
            ← Back to dashboard
          </Link>
        </div>
      </div>
    )
  }

  // Cast patient to non-null for the rest of the component
  const p = patient!

  const hasPendingMemories = memoryCandidates.length > 0
  const isChatMode = activeTab === "chat"

  return (
    <TooltipProvider>
      <div className="workspace-grid bg-[var(--color-bg)]">

        {/* ── Col 1: Sidebar ──────────────────────────────────────────── */}
        <aside className="flex flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)] overflow-y-auto custom-scrollbar">

          <div className="px-4 py-3 border-b border-[var(--color-divider)] flex items-center gap-2">
            <Link
              href="/"
              className="p-1.5 rounded-md hover:bg-[var(--color-surface-offset)] transition-colors"
              aria-label="Back to dashboard"
            >
              <ChevronLeft className="w-4 h-4 text-[var(--color-text-muted)]" />
            </Link>
            <div className="flex items-center gap-2 flex-1">
              <div className="w-6 h-6 rounded bg-[var(--color-primary)] flex items-center justify-center">
                <Brain className="w-3.5 h-3.5 text-white" />
              </div>
              <span className="text-xs font-semibold tracking-tight">Therapist Assistant</span>
            </div>
            <ApiStatus />
            <ThemeToggle />
          </div>

          <div className="px-4 py-4 border-b border-[var(--color-divider)]">
            <div className="w-10 h-10 rounded-full bg-[var(--color-primary-highlight)] flex items-center justify-center text-sm font-semibold text-[var(--color-primary)] mb-3">
              {p.display_name.split(" ")[1]?.[0] ?? p.display_name[0]}
            </div>
            <p className="text-sm font-semibold">{p.display_name}</p>
            {p.notes && (
              <p className="text-xs text-[var(--color-text-muted)] mt-1 leading-relaxed">
                {p.notes}
              </p>
            )}
            {p.last_session && (
              <div className="flex items-center gap-1.5 mt-3">
                <Clock className="w-3 h-3 text-[var(--color-text-faint)]" />
                <span className="text-xs text-[var(--color-text-faint)]">
                  Last session:{" "}
                  {new Date(p.last_session).toLocaleDateString("en-US", {
                    month: "short", day: "numeric", year: "numeric",
                  })}
                </span>
              </div>
            )}
          </div>

          {/* Past sessions in sidebar — real DB data */}
          <div className="px-4 py-3 flex-1 overflow-y-auto custom-scrollbar">
            <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide mb-2">
              Past Sessions
            </p>
            {isLoadingSessions ? (
              <div className="space-y-1.5">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="skeleton h-10 w-full rounded-md" />
                ))}
              </div>
            ) : sessions.length === 0 ? (
              <p className="text-xs text-[var(--color-text-faint)]">No sessions yet.</p>
            ) : (
              sessions.slice(0, 8).map((s) => {
                const cfg = s.analysis ? RISK_TIER_CONFIG[s.analysis.risk_tier] : null
                return (
                  <button
                    key={s.id}
                    onClick={() => loadSession(s)}
                    className="w-full text-left px-2 py-2 rounded-md hover:bg-[var(--color-surface-offset)] hover:ring-1 hover:ring-[var(--color-primary)] transition-all mb-1 group"
                  >
                    <div className="flex items-center justify-between gap-1">
                      <p className="text-xs font-medium text-[var(--color-text)] truncate">
                        {new Date(s.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                      </p>
                      {cfg && (
                        <span className={cn("text-[10px] px-1.5 py-0.5 rounded-full shrink-0", cfg.bg, cfg.color)}>
                          {s.analysis?.risk_tier}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-[var(--color-text-faint)] truncate mt-0.5 group-hover:text-[var(--color-text-muted)]">
                      {s.raw_text}
                    </p>
                  </button>
                )
              })
            )}
          </div>
        </aside>

        {/* ── Col 2: Input workspace or Chat ───────────────────────────── */}
        <main className="flex flex-col overflow-hidden">
          {isChatMode ? (
            <ChatPanel patientId={id} patientName={p.display_name} />
          ) : (
            <>
              <div className="px-5 py-3 border-b border-[var(--color-border)] bg-[var(--color-surface)] flex items-center justify-between shrink-0">
                <div>
                  <p className="text-sm font-semibold">{p.display_name}</p>
                  <p className="text-xs text-[var(--color-text-muted)]">Session workspace</p>
                </div>
                {analysisResult && (
                  <div className={cn(
                    "text-xs font-medium px-2.5 py-1 rounded-full border",
                    RISK_TIER_CONFIG[analysisResult.risk_tier].bg,
                    RISK_TIER_CONFIG[analysisResult.risk_tier].color,
                    RISK_TIER_CONFIG[analysisResult.risk_tier].border,
                  )}>
                    {RISK_TIER_CONFIG[analysisResult.risk_tier].label}
                  </div>
                )}
              </div>

              <div className="flex-1 overflow-y-auto custom-scrollbar px-5 py-4">
                <textarea
                  ref={textareaRef}
                  className="w-full h-full min-h-[300px] resize-none bg-transparent text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-faint)] focus:outline-none leading-relaxed"
                  placeholder={`Paste a session note, message, or journal entry for ${p.display_name}...\n\nPress ⌘ + Enter to analyze.`}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runAnalysis()
                  }}
                />
              </div>

              <div className="px-5 py-3 border-t border-[var(--color-border)] bg-[var(--color-surface)] flex items-center justify-between shrink-0">
                <p className="text-xs text-[var(--color-text-faint)]">
                  {wordCount(inputText)} words
                </p>
                <Button
                  size="sm"
                  className="gap-1.5 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white"
                  disabled={!inputText.trim() || isAnalyzing}
                  onClick={runAnalysis}
                >
                  {isAnalyzing
                    ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Analyzing...</>
                    : <><Send className="w-3.5 h-3.5" /> Analyze</>
                  }
                </Button>
              </div>
            </>
          )}
        </main>

        {/* ── Col 3: Right panel tabs ──────────────────────────────────── */}
        <aside className="flex flex-col border-l border-[var(--color-border)] bg-[var(--color-surface)] overflow-hidden">
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as typeof activeTab)}
            className="flex flex-col h-full"
          >
            <div className="px-3 pt-3 border-b border-[var(--color-divider)] shrink-0">
              <TabsList className="w-full grid grid-cols-4 h-8">
                <TabsTrigger value="analysis" className="text-xs">
                  Analysis
                </TabsTrigger>
                <TabsTrigger value="memory" className="text-xs relative">
                  Memory
                  {hasPendingMemories && (
                    <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-[var(--color-primary)] text-white text-[10px] flex items-center justify-center font-medium">
                      {memoryCandidates.length}
                    </span>
                  )}
                </TabsTrigger>
                <TabsTrigger value="history" className="text-xs">
                  History
                </TabsTrigger>
                <TabsTrigger value="chat" className="text-xs flex items-center gap-1">
                  <MessageSquare className="w-3 h-3" />
                  Chat
                </TabsTrigger>
              </TabsList>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar">
              <TabsContent value="analysis" className="mt-0 h-full">
                <AnalysisPanel />
              </TabsContent>
              <TabsContent value="memory" className="mt-0 h-full">
                <MemoryPanel />
              </TabsContent>
              <TabsContent value="history" className="mt-0">
                <HistoryPanel />
              </TabsContent>
              <TabsContent value="chat" className="mt-0 h-full">
                <ChatInsights patientId={id} />
              </TabsContent>
            </div>
          </Tabs>
        </aside>

      </div>
    </TooltipProvider>
  )
}