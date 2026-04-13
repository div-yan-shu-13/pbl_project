"use client"

import { useEffect, useState } from "react"
import { checkHealth, type HealthResponse } from "@/lib/api"

type HealthState = "checking" | "ok" | "model-not-loaded" | "unreachable"

export function useApiHealth() {
  const [state,   setState]   = useState<HealthState>("checking")
  const [version, setVersion] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    checkHealth()
      .then((h: HealthResponse) => {
        if (cancelled) return
        setState(h.model_loaded ? "ok" : "model-not-loaded")
        setVersion(h.version)
      })
      .catch(() => {
        if (!cancelled) setState("unreachable")
      })

    return () => { cancelled = true }
  }, [])

  return { state, version }
}