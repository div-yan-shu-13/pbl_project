import { NextResponse } from "next/server"
import { db } from "@/lib/db"

export async function GET() {
  try {
    const patients = await db.patient.findMany({
      orderBy: {
        last_session: {
          sort: "desc",
          nulls: "last",
        },
      },
      select: {
        id: true,
        display_name: true,
        notes: true,
        created_at: true,
        last_session: true,
        sessions: {
          orderBy: { created_at: "desc" },
          take: 1,
          select: {
            analysis: {
              select: {
                risk_tier: true,
              },
            },
          },
        },
        _count: {
          select: {
            sessions: true,
            memories: true,
            memoryCandidates: true,
          },
        },
      },
    })

    return NextResponse.json(patients)
  } catch (error: any) {
    console.error("[GET /api/patients] Error", error)
    return NextResponse.json(
      { error: "Failed to fetch patients", details: error?.message || String(error) },
      { status: 500 }
    )
  }
}

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const display_name = String(body.display_name ?? "").trim()
    const notes =
      typeof body.notes === "string" && body.notes.trim().length > 0
        ? body.notes.trim()
        : null

    if (!display_name) {
      return NextResponse.json(
        { error: "display_name is required" },
        { status: 400 }
      )
    }

    const patient = await db.patient.create({
      data: {
        display_name,
        notes,
      },
      select: {
        id: true,
        display_name: true,
        notes: true,
        created_at: true,
        last_session: true,
        _count: {
          select: {
            sessions: true,
            memories: true,
            memoryCandidates: true,
          },
        },
      },
    })

    return NextResponse.json(patient, { status: 201 })
  } catch (error) {
    console.error("[POST /api/patients] Error", error)
    return NextResponse.json(
      { error: "Failed to create patient" },
      { status: 500 }
    )
  }
}