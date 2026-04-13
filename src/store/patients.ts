// src/store/patients.ts
// No more seed data. No more localStorage.
// State is fetched from /api/patients and mutated via API calls.
// Zustand is purely an in-memory UI cache here.

import { create } from "zustand"
import type { Patient } from "@/types"

interface PatientsStore {
  patients: Patient[]
  isLoading: boolean
  error: string | null

  // Fetch full list from DB
  fetchPatients: () => Promise<void>

  // Create a new patient via API, then add to local cache
  addPatient: (data: { displayName: string; notes?: string }) => Promise<Patient>

  // Update a patient via API, then sync local cache
  updatePatient: (id: string, updates: Partial<Pick<Patient, "display_name" | "notes">>) => Promise<void>

  // Delete a patient via API, then remove from local cache
  removePatient: (id: string) => Promise<void>

  // Local-only read — no fetch
  getPatient: (id: string) => Patient | undefined
}

export const usePatientsStore = create<PatientsStore>()((set, get) => ({
  patients: [],
  isLoading: false,
  error: null,

  // ── fetchPatients ───────────────────────────────────────────────────────────
  fetchPatients: async () => {
    set({ isLoading: true, error: null })
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 30000)

      const res = await fetch("/api/patients", { signal: controller.signal })
      clearTimeout(timeout)

      if (!res.ok) throw new Error(`Failed to fetch patients (${res.status})`)
      const data = await res.json()
      const list = Array.isArray(data) ? data : []

      set({
        patients: list.map(normalisePatient),
        isLoading: false,
      })
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Unknown error",
        isLoading: false,
      })
    }
  },

  // ── addPatient ──────────────────────────────────────────────────────────────
  addPatient: async ({ displayName, notes }) => {
    const res = await fetch("/api/patients", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName, notes }),
    })
    if (!res.ok) {
      const { error } = await res.json()
      throw new Error(error ?? "Failed to create patient")
    }
    const patient = await res.json()
    const normalised = normalisePatient(patient)
    set((state) => ({ patients: [normalised, ...state.patients] }))
    return normalised
  },

  // ── updatePatient ───────────────────────────────────────────────────────────
  updatePatient: async (id, updates) => {
    const body: Record<string, string> = {}
    if (updates.display_name) body.display_name = updates.display_name
    if (updates.notes) body.notes = updates.notes

    const res = await fetch(`/api/patients/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const { error } = await res.json()
      throw new Error(error ?? "Failed to update patient")
    }
    const patient = await res.json()
    const normalised = normalisePatient(patient)
    set((state) => ({
      patients: state.patients.map((p) => (p.id === id ? normalised : p)),
    }))
  },

  // ── removePatient ───────────────────────────────────────────────────────────
  removePatient: async (id) => {
    const res = await fetch(`/api/patients/${id}`, { method: "DELETE" })
    if (!res.ok) {
      const { error } = await res.json()
      throw new Error(error ?? "Failed to delete patient")
    }
    set((state) => ({
      patients: state.patients.filter((p) => p.id !== id),
    }))
  },

  // ── getPatient ──────────────────────────────────────────────────────────────
  getPatient: (id) => get().patients.find((p) => p.id === id),
}))

// ─── Normalise DB row → UI Patient shape ──────────────────────────────────────
// The DB uses camelCase (displayName, lastSession).
// The existing UI type uses snake_case (display_name, last_session).
// We normalise here so zero component code changes are needed.

function normalisePatient(p: any): Patient {
  return {
    id: p.id,
    display_name: p.display_name,
    notes: p.notes ?? null,
    created_at: p.created_at,
    last_session: p.last_session ?? null,
    last_risk_tier: (p.sessions?.[0]?.analysis?.risk_tier as any) ?? "Low",
  }
}