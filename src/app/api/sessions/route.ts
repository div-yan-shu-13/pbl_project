import { NextResponse } from "next/server"
import { db } from "@/lib/db"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const patientId = searchParams.get("patientId")

    const sessions = await db.session.findMany({
      where: patientId ? { patient_id: patientId } : {},
      orderBy: { created_at: "desc" },
      take: patientId ? undefined : 10,
      include: {
        analysis: true,
      },
    })

    return NextResponse.json({ sessions })
  } catch (error: any) {
    console.error("[GET /api/sessions] Error", error)
    return NextResponse.json(
      { error: "Failed to fetch global sessions", details: error?.message || String(error) },
      { status: 500 }
    )
  }
}