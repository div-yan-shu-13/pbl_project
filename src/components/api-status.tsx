"use client"

import { useApiHealth } from "@/lib/api-health"
import { cn } from "@/lib/utils"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export function ApiStatus() {
  const { state, version } = useApiHealth()

  const config = {
    checking:          { dot: "bg-[var(--color-text-faint)] animate-pulse", label: "Connecting..."    },
    ok:                { dot: "bg-emerald-500",                              label: `API ready · v${version}` },
    "model-not-loaded": { dot: "bg-orange-400",                              label: "Model not loaded" },
    unreachable:       { dot: "bg-red-500",                                  label: "Backend offline"  },
  }[state]

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-1.5 cursor-default">
            <span className={cn("w-2 h-2 rounded-full shrink-0", config.dot)} />
            <span className="text-xs text-[var(--color-text-faint)] hidden sm:block">
              {state === "ok" ? "API" : config.label}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent className="text-xs">{config.label}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}