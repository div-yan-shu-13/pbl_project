// src/app/api/memories/[id]/route.ts

import { NextRequest, NextResponse } from "next/server"
import { db } from "@/lib/db"

type Params = { params: Promise<{ id: string }> }

// ─── DELETE /api/memories/:id ─────────────────────────────────────────────────
// Soft-deletes a memory by setting status = "archived".
// We keep the row so session history still has a reference to it.

export async function DELETE(_req: NextRequest, { params }: Params) {
  try {
    const { id } = await params

    const memory = await db.memory.update({
      where: { id },
      data:  { status: "archived" },
    })

    return NextResponse.json({ memory })
  } catch (err: any) {
    if (err?.code === "P2025") {
      return NextResponse.json({ error: "Memory not found" }, { status: 404 })
    }
    console.error("[DELETE /api/memories/:id]", err)
    return NextResponse.json(
      { error: "Failed to delete memory" },
      { status: 500 }
    )
  }
}

// ─── PATCH /api/memories/:id ──────────────────────────────────────────────────
// Update memory title or description if therapist wants to edit it.
// Body: { title?: string, description?: string }

export async function PATCH(req: NextRequest, { params }: Params) {
  try {
    const { id } = await params
    const body   = await req.json()

    const data: Record<string, unknown> = {}
    if (typeof body.title       === "string") data.title       = body.title.trim()
    if (typeof body.description === "string") data.description = body.description.trim()

    if (Object.keys(data).length === 0) {
      return NextResponse.json({ error: "No valid fields to update" }, { status: 400 })
    }

    const memory = await db.memory.update({
      where: { id },
      data,
    })

    return NextResponse.json({ memory })
  } catch (err: any) {
    if (err?.code === "P2025") {
      return NextResponse.json({ error: "Memory not found" }, { status: 404 })
    }
    console.error("[PATCH /api/memories/:id]", err)
    return NextResponse.json(
      { error: "Failed to update memory" },
      { status: 500 }
    )
  }
}