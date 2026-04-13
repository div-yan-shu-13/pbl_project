"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { useChatStore, type ChatRole } from "@/store/chat"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Send, Zap, ZapOff } from "lucide-react"

interface ChatPanelProps {
  patientId:   string
  patientName: string
}

// Auto-analyze fires 2.5s after the last patient message when enabled
const AUTO_ANALYZE_DELAY = 2500

export function ChatPanel({ patientId, patientName }: ChatPanelProps) {
  const {
    messages,
    inputMode,
    autoAnalyze,
    analyzeCount,
    isAnalyzing,
    setInputMode,
    toggleAutoAnalyze,
    addMessage,
    runChatAnalysis,
    clearChat,
  } = useChatStore()

  const [draft, setDraft]   = useState("")
  const bottomRef           = useRef<HTMLDivElement>(null)
  const textareaRef         = useRef<HTMLTextAreaElement>(null)
  const autoTimer           = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Auto-grow textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setDraft(e.target.value)
    e.target.style.height = "auto"
    e.target.style.height = Math.min(e.target.scrollHeight, 100) + "px"
  }

  const scheduleAutoAnalyze = useCallback(() => {
    if (!autoAnalyze) return
    if (autoTimer.current) clearTimeout(autoTimer.current)
    autoTimer.current = setTimeout(() => runChatAnalysis(patientId), AUTO_ANALYZE_DELAY)
  }, [autoAnalyze, patientId, runChatAnalysis])

  const send = () => {
    const text = draft.trim()
    if (!text) return
    addMessage(inputMode, text)
    setDraft("")
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
    if (inputMode === "patient") scheduleAutoAnalyze()
  }

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() }
  }

  const initials = patientName.split(" ").map((n) => n[0]).join("").slice(0, 2)

  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* Header */}
      <div className="px-5 py-3 border-b border-[var(--color-border)] bg-[var(--color-surface)] flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded-full bg-[var(--color-primary-subtle)] flex items-center justify-center text-xs font-bold text-[var(--color-primary)]">
          {initials}
        </div>
        <div>
          <p className="text-sm font-semibold">{patientName}</p>
          <p className="text-xs text-[var(--color-text-muted)]">Live session · Chat mode</p>
        </div>

        <div className="ml-auto flex items-center gap-2">
          {/* Analyze count badge */}
          {analyzeCount > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--color-surface-offset)] border border-[var(--color-border)] text-[var(--color-text-faint)] font-mono tabular-nums">
              {analyzeCount} {analyzeCount === 1 ? "analysis" : "analyses"}
            </span>
          )}

          {/* Auto-analyze toggle */}
          <button
            onClick={toggleAutoAnalyze}
            className={cn(
              "flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors",
              autoAnalyze
                ? "bg-[var(--color-primary-highlight)] text-[var(--color-primary)] border-transparent"
                : "bg-transparent text-[var(--color-text-muted)] border-[var(--color-border)] hover:bg-[var(--color-surface-offset)]"
            )}
            title={autoAnalyze ? "Auto-Analyze on" : "Auto-Analyze off"}
          >
            {autoAnalyze
              ? <Zap    className="w-3 h-3" />
              : <ZapOff className="w-3 h-3" />
            }
            Auto
          </button>

          {/* Clear */}
          <button
            onClick={clearChat}
            className="text-xs text-[var(--color-text-faint)] hover:text-[var(--color-text-muted)] px-2 py-1 rounded-md hover:bg-[var(--color-surface-offset)] transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-5 py-4 flex flex-col gap-3">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center flex-1 text-center py-16 gap-3">
            <div className="w-10 h-10 rounded-xl bg-[var(--color-surface-offset)] flex items-center justify-center">
              <Send className="w-5 h-5 text-[var(--color-text-faint)]" />
            </div>
            <p className="text-sm font-medium text-[var(--color-text-muted)]">No messages yet</p>
            <p className="text-xs text-[var(--color-text-faint)] max-w-[22ch] leading-relaxed">
              Type a patient message below. The sidebar updates with live insights as you go.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={cn(
              "flex flex-col gap-0.5 max-w-[80%]",
              msg.role === "patient" ? "self-start" : "self-end items-end"
            )}
          >
            <span className="text-[11px] text-[var(--color-text-faint)] px-1 flex items-center gap-1.5">
              <span className={cn(
                "w-1.5 h-1.5 rounded-full",
                msg.role === "patient" ? "bg-[var(--color-blue)]" : "bg-[var(--color-primary)]"
              )} />
              {msg.role === "patient" ? patientName : "Dr. (You)"}
            </span>

            <div className={cn(
              "px-3 py-2 text-sm leading-relaxed",
              msg.role === "patient"
                ? cn(
                    "bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg rounded-tl-sm text-[var(--color-text)]",
                    msg.flagged && "ring-2 ring-[var(--color-warning)] border-[var(--color-warning)]"
                  )
                : "bg-[var(--color-primary-highlight)] text-[var(--color-primary)] rounded-lg rounded-tr-sm"
            )}>
              {msg.text}
              {msg.flagged && (
                <span className="ml-1.5 inline-flex items-center text-[var(--color-warning)]" title="Flagged by analysis">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/>
                    <line x1="4" y1="22" x2="4" y2="15" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </span>
              )}
            </div>

            <span className="text-[11px] text-[var(--color-text-faint)] px-1">
              {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="px-5 py-3 border-t border-[var(--color-border)] bg-[var(--color-surface)] flex flex-col gap-2 shrink-0">

        {/* Mode tabs */}
        <div className="flex gap-1">
          {(["patient", "therapist"] as ChatRole[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setInputMode(mode)}
              className={cn(
                "px-3 py-1 rounded-full text-[11px] font-medium border capitalize transition-colors",
                inputMode === mode
                  ? "bg-[var(--color-surface-dynamic)] border-transparent text-[var(--color-text)]"
                  : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:bg-[var(--color-surface-offset)]"
              )}
            >
              {mode}
            </button>
          ))}
        </div>

        {/* Textarea + send */}
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={handleInput}
            onKeyDown={handleKey}
            rows={1}
            placeholder={`Type as ${inputMode === "patient" ? patientName : "therapist"}…`}
            className="flex-1 resize-none bg-[var(--color-surface-offset)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-faint)] focus:outline-none focus:border-[var(--color-primary)] focus:ring-2 focus:ring-[var(--color-primary)]/15 min-h-[40px] max-h-[100px] leading-relaxed transition-colors"
          />
          <Button
            size="icon"
            className="h-9 w-9 shrink-0 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white"
            onClick={send}
            disabled={!draft.trim()}
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>

        {/* Bottom row */}
        <div className="flex items-center justify-between">
          <button
            onClick={() => runChatAnalysis(patientId)}
            disabled={isAnalyzing || messages.filter((m) => m.role === "patient").length === 0}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs border transition-colors",
              isAnalyzing
                ? "bg-[var(--color-primary-highlight)] text-[var(--color-primary)] border-transparent cursor-wait"
                : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:bg-[var(--color-primary-highlight)] hover:text-[var(--color-primary)] hover:border-transparent disabled:opacity-40 disabled:cursor-not-allowed"
            )}
          >
            {isAnalyzing ? (
              <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-3-6.7"/>
              </svg>
            ) : (
              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
            )}
            {isAnalyzing ? "Analyzing…" : "Analyze session"}
          </button>
          <span className="text-[11px] text-[var(--color-text-faint)]">↵ send · Shift↵ new line</span>
        </div>
      </div>
    </div>
  )
}