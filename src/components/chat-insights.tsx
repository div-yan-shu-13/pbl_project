"use client"

import { useChatStore } from "@/store/chat"
import { useWorkspaceStore } from "@/store/workspace"
import { RISK_TIER_CONFIG, MEMORY_TYPE_CONFIG } from "@/types"
import { cn } from "@/lib/utils"
import { AlertTriangle, Brain, Check, X, Zap } from "lucide-react"
import { Separator } from "@/components/ui/separator"

const SIGNAL_COLORS: Record<string, string> = {
  hopelessness:     "var(--color-error)",
  grief:            "var(--color-blue)",
  isolation:        "var(--color-warning)",
  anxiety:          "var(--color-gold)",
  "passive-SI":     "var(--color-error)",
  "coping skills":  "var(--color-success)",
  default:          "var(--color-primary)",
}

export function ChatInsights({ patientId }: { patientId: string }) {
  const {
    chatAnalysis,
    chatCandidates,
    isAnalyzing,
    analysisError,
    autoAnalyze,
  } = useChatStore()

  const { acceptMemoryCandidate: _accept, savedMemories } = useWorkspaceStore()

  // We wire candidate acceptance into the workspace store so memories persist
  const acceptCandidate = async (idx: number) => {
    const candidate = chatCandidates[idx]
    if (!candidate) return
    
    try {
      const { saveMemoryFromCandidate } = useWorkspaceStore.getState()
      await saveMemoryFromCandidate(candidate)
      
      // On success, remove from local chat state
      useChatStore.setState((s) => ({
        chatCandidates: s.chatCandidates.filter((_, i) => i !== idx),
      }))
    } catch (err) {
      // Error handled by store
    }
  }

  // ── Analyzing skeleton ──────────────────────────────────────────────────
  if (isAnalyzing) {
    return (
      <div className="p-4 space-y-3">
        <div className="skeleton h-14 rounded-lg w-full" />
        <div className="space-y-2">
          <div className="skeleton h-3 w-24 rounded" />
          <div className="skeleton h-3 w-full rounded" />
          <div className="skeleton h-3 w-5/6 rounded" />
        </div>
        <div className="flex gap-2">
          <div className="skeleton h-5 w-16 rounded-full" />
          <div className="skeleton h-5 w-20 rounded-full" />
          <div className="skeleton h-5 w-14 rounded-full" />
        </div>
        <div className="space-y-1.5">
          {[1, 2, 3].map((i) => (
            <div key={i} className="skeleton h-8 rounded-md w-full" />
          ))}
        </div>
      </div>
    )
  }

  // ── Error ───────────────────────────────────────────────────────────────
  if (analysisError) {
    return (
      <div className="m-4 px-4 py-3 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800">
        <p className="text-sm text-red-600 dark:text-red-400 font-medium">Analysis failed</p>
        <p className="text-xs text-red-500 dark:text-red-400 mt-1">{analysisError}</p>
      </div>
    )
  }

  // ── Idle ────────────────────────────────────────────────────────────────
  if (!chatAnalysis) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[240px] text-center px-6 py-12 gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-surface-offset)] flex items-center justify-center">
          <Brain className="w-5 h-5 text-[var(--color-text-faint)]" />
        </div>
        <p className="text-sm font-medium text-[var(--color-text-muted)]">Waiting for messages</p>
        <p className="text-xs text-[var(--color-text-faint)] max-w-[22ch] leading-relaxed">
          {autoAnalyze
            ? "Insights will appear automatically after each patient message."
            : "Hit 'Analyze session' below the chat to see live insights."}
        </p>
        {autoAnalyze && (
          <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-primary)]">
            <Zap className="w-3 h-3" /> Auto-Analyze on
          </div>
        )}
      </div>
    )
  }

  const cfg = RISK_TIER_CONFIG[chatAnalysis.risk_tier]

  return (
    <div className="p-4 space-y-4">

      {/* High/Crisis alert */}
      {(chatAnalysis.risk_tier === "High" || chatAnalysis.risk_tier === "Crisis") && (
        <div className={cn(
          "flex items-center gap-2 px-3 py-2.5 rounded-lg border text-xs font-medium",
          cfg.bg, cfg.border, cfg.color
        )}>
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          {chatAnalysis.risk_tier === "Crisis"
            ? "Crisis-level risk detected. Review immediately."
            : "High-risk indicators in session."}
        </div>
      )}

      {/* Risk block */}
      <div className={cn(
        "flex items-center justify-between px-4 py-3 rounded-lg border",
        cfg.bg, cfg.border
      )}>
        <div>
          <p className={cn("text-sm font-bold", cfg.color)}>{cfg.label}</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-0.5 capitalize">
            {chatAnalysis.confidence} confidence
          </p>
        </div>
        <span className="text-xs font-mono text-[var(--color-text-faint)]">
          {chatAnalysis.raw_score.toFixed(3)}
        </span>
      </div>

      {/* Summary */}
      <div className="space-y-1">
        <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">Summary</p>
        <p className="text-xs text-[var(--color-text)] leading-relaxed">{chatAnalysis.summary}</p>
      </div>

      <Separator className="bg-[var(--color-divider)]" />

      {/* Signals */}
      {chatAnalysis.signal_labels.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">Signals</p>
          <div className="flex flex-wrap gap-1.5">
            {chatAnalysis.signal_labels.map((sig) => (
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

      {/* Evidence spans */}
      {chatAnalysis.evidence_spans.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">Key phrases</p>
          <div className="space-y-1.5">
            {chatAnalysis.evidence_spans.map((span, i) => {
              const color = SIGNAL_COLORS[span.label] ?? SIGNAL_COLORS.default
              return (
                <div key={i} className="flex items-start gap-2 text-xs px-2 py-1.5 rounded-md bg-[var(--color-surface-offset)]">
                  <div className="w-0.5 self-stretch rounded-full shrink-0 mt-0.5" style={{ background: color }} />
                  <div className="min-w-0 flex-1">
                    <span className="font-medium text-[var(--color-text)] italic">"{span.text}"</span>
                    <div className="flex items-center justify-between mt-0.5">
                      <span className="text-[var(--color-text-faint)]">{span.label}</span>
                      <span className="font-mono text-[var(--color-text-muted)]">{(span.score * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Memory candidates */}
      {chatCandidates.length > 0 && (
        <>
          <Separator className="bg-[var(--color-divider)]" />
          <div className="space-y-2">
            <p className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wide">
              Memory candidates ({chatCandidates.length})
            </p>
            <div className="space-y-2">
              {chatCandidates.map((c, i) => (
                <div key={i} className="panel-card px-3 py-2.5 space-y-1.5">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ background: MEMORY_TYPE_CONFIG[c.type]?.color ?? "var(--color-primary)" }}
                      />
                      <span className="text-xs font-medium truncate">{c.title}</span>
                    </div>
                    <div className="flex gap-1 shrink-0">
                      <button
                        onClick={() => acceptCandidate(i)}
                        className="p-1 rounded-md hover:bg-emerald-100 dark:hover:bg-emerald-950/40 text-emerald-600 transition-colors"
                        aria-label="Accept memory"
                      >
                        <Check className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => useChatStore.setState((s) => ({
                          chatCandidates: s.chatCandidates.filter((_, idx) => idx !== i),
                        }))}
                        className="p-1 rounded-md hover:bg-red-100 dark:hover:bg-red-950/40 text-red-500 transition-colors"
                        aria-label="Reject memory"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                  <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">{c.description}</p>
                  <span className="text-[11px] text-[var(--color-text-faint)] capitalize">
                    {MEMORY_TYPE_CONFIG[c.type]?.label ?? c.type.replace(/_/g, " ")}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}