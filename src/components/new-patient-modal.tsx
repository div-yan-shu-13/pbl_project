"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { usePatientsStore } from "@/store/patients"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Loader2 } from "lucide-react"

interface NewPatientModalProps {
  open:      boolean
  onClose:   () => void
}

export function NewPatientModal({ open, onClose }: NewPatientModalProps) {
  const router      = useRouter()
  const addPatient  = usePatientsStore((s) => s.addPatient)

  const [displayName, setDisplayName] = useState("")
  const [notes, setNotes]             = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError]             = useState<string | null>(null)

  function handleClose() {
    if (isSubmitting) return
    setDisplayName("")
    setNotes("")
    setError(null)
    onClose()
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const name = displayName.trim()
    if (!name) {
      setError("Patient name is required.")
      return
    }
    if (name.length < 2) {
      setError("Name must be at least 2 characters.")
      return
    }

    setIsSubmitting(true)
    setError(null)

    // Simulate brief async (replace with API call later)
    await new Promise((r) => setTimeout(r, 300))

    const patient = await addPatient({
      displayName: name,
      notes:        notes.trim() || undefined,
    })

    setIsSubmitting(false)
    handleClose()
    router.push(`/patients/${patient.id}`)
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="sm:max-w-[420px] bg-[var(--color-surface)] border-[var(--color-border)]">
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">New Patient</DialogTitle>
          <DialogDescription className="text-xs text-[var(--color-text-muted)]">
            Add a new patient to your workspace. You can edit details later.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4 pt-1">

          {/* Name */}
          <div className="space-y-1.5">
            <Label htmlFor="display-name" className="text-xs font-medium">
              Patient Name / Alias <span className="text-red-500">*</span>
            </Label>
            <Input
              id="display-name"
              placeholder="e.g. Patient E, or a pseudonym"
              value={displayName}
              onChange={(e) => {
                setDisplayName(e.target.value)
                if (error) setError(null)
              }}
              autoFocus
              className="text-sm bg-[var(--color-surface-2)] border-[var(--color-border)] focus:border-[var(--color-primary)]"
            />
            {error && (
              <p className="text-xs text-red-500">{error}</p>
            )}
          </div>

          {/* Notes */}
          <div className="space-y-1.5">
            <Label htmlFor="notes" className="text-xs font-medium">
              Clinical Notes
              <span className="text-[var(--color-text-faint)] font-normal ml-1">(optional)</span>
            </Label>
            <Textarea
              id="notes"
              placeholder="Age, referral reason, relevant background..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              className="text-sm resize-none bg-[var(--color-surface-2)] border-[var(--color-border)] focus:border-[var(--color-primary)]"
            />
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-1">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleClose}
              disabled={isSubmitting}
              className="text-[var(--color-text-muted)]"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              size="sm"
              disabled={!displayName.trim() || isSubmitting}
              className="bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white gap-1.5"
            >
              {isSubmitting
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Creating...</>
                : "Create Patient"
              }
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}