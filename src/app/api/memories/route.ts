// src/app/api/memories/route.ts

import { NextRequest, NextResponse } from "next/server"
import { db } from "@/lib/db"

// ─── GET /api/memories?patientId=... ─────────────────────────────────────────
// Returns all accepted (saved) memories for a patient, newest first.

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const patientId        = searchParams.get("patientId")

    if (!patientId) {
      return NextResponse.json(
        { error: "patientId query param is required" },
        { status: 400 }
      )
    }

    const memories = await db.memory.findMany({
      where:   { patient_id: patientId, status: "active" },
      orderBy: { created_at: "desc" },
    })

    return NextResponse.json({ memories })
  } catch (err) {
    console.error("[GET /api/memories]", err)
    return NextResponse.json(
      { error: "Failed to fetch memories" },
      { status: 500 }
    )
  }
}

// ─── POST /api/memories ───────────────────────────────────────────────────────
// Accepts a memory candidate — promotes it to a permanent Memory row
// and marks the MemoryCandidate as accepted.
// Body: { candidateId: string }

export async function POST(req: NextRequest) {
  try {
    const body        = await req.json()
    const { candidateId } = body

    if (!candidateId || typeof candidateId !== "string") {
      return NextResponse.json(
        { error: "candidateId is required" },
        { status: 400 }
      )
    }

    // Load candidate — need patientId, sessionId, and content
    const candidate = await db.memoryCandidate.findUnique({
      where: { id: candidateId },
    })

    if (!candidate) {
      return NextResponse.json({ error: "Candidate not found" }, { status: 404 })
    }

    if (candidate.status !== "pending") {
      return NextResponse.json(
        { error: `Candidate is already ${candidate.status}` },
        { status: 409 }
      )
    }

    // Promote candidate → Memory and mark candidate accepted in one transaction
    const { memory } = await db.$transaction(async (tx) => {

      const memory = await tx.memory.create({
        data: {
          patient_id:            candidate.patient_id,
          type:                  candidate.type,
          title:                 candidate.title,
          description:           candidate.description,
          therapist_verified:    true,
          first_seen_session_id: candidate.session_id,
          last_seen_session_id:  candidate.session_id,
        },
      })

      await tx.memoryCandidate.update({
        where: { id: candidateId },
        data:  { status: "accepted" },
      })

      return { memory }
    })

    return NextResponse.json({ memory }, { status: 201 })
  } catch (err) {
    console.error("[POST /api/memories]", err)
    return NextResponse.json(
      { error: "Failed to accept memory" },
      { status: 500 }
    )
  }
}