// src/app/api/patients/[id]/route.ts

import { NextRequest, NextResponse } from "next/server"
import { db } from "@/lib/db"

type Params = { params: Promise<{ id: string }> }

// ─── GET /api/patients/:id ────────────────────────────────────────────────────
// Returns a single patient with their session count and memory count.

export async function GET(_req: NextRequest, { params }: Params) {
  try {
    const { id } = await params

    const patient = await db.patient.findUnique({
      where: { id },
      include: {
        _count: {
          select: { sessions: true, memories: true, memoryCandidates: true },
        },
      },
    })

    if (!patient) {
      return NextResponse.json({ error: "Patient not found" }, { status: 404 })
    }

    return NextResponse.json(patient)
  } catch (err) {
    console.error("[GET /api/patients/:id]", err)
    return NextResponse.json(
      { error: "Failed to fetch patient" },
      { status: 500 }
    )
  }
}

// ─── PATCH /api/patients/:id ──────────────────────────────────────────────────
// Partial update. Body: { displayName?: string, notes?: string }

export async function PATCH(req: NextRequest, { params }: Params) {
  try {
    const { id }  = await params
    const body    = await req.json()

    const data: Record<string, unknown> = {}
    if (typeof body.display_name === "string") data.display_name = body.display_name.trim()
    if (typeof body.notes        === "string") data.notes        = body.notes.trim()
    if (typeof body.last_session === "string") data.last_session = new Date(body.last_session)

    if (Object.keys(data).length === 0) {
      return NextResponse.json({ error: "No valid fields to update" }, { status: 400 })
    }

    const patient = await db.patient.update({
      where: { id },
      data,
    })

    return NextResponse.json(patient)
  } catch (err: any) {
    if (err?.code === "P2025") {
      return NextResponse.json({ error: "Patient not found" }, { status: 404 })
    }
    console.error("[PATCH /api/patients/:id]", err)
    return NextResponse.json(
      { error: "Failed to update patient" },
      { status: 500 }
    )
  }
}

// ─── DELETE /api/patients/:id ─────────────────────────────────────────────────
// Hard delete — cascades to sessions, analyses, evidence spans,
// memories, and candidates via the schema's onDelete: Cascade rules.

export async function DELETE(_req: NextRequest, { params }: Params) {
  try {
    const { id } = await params

    await db.patient.delete({ where: { id } })

    return NextResponse.json({ success: true })
  } catch (err: any) {
    if (err?.code === "P2025") {
      return NextResponse.json({ error: "Patient not found" }, { status: 404 })
    }
    console.error("[DELETE /api/patients/:id]", err)
    return NextResponse.json(
      { error: "Failed to delete patient" },
      { status: 500 }
    )
  }
}