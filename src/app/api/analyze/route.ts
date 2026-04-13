import { NextResponse } from "next/server"
import { db } from "@/lib/db"

const FASTAPI_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

type FastApiEvidenceSpan = {
  text: string
  label: string
  score: number
  start_idx: number | null
  end_idx: number | null
}

type FastApiMemoryCandidate = {
  type: string
  title: string
  description: string
  confidence: number
}

type FastApiAnalyzeResponse = {
  risk_tier: string
  context_label: string
  signal_labels: string[]
  confidence: string
  summary: string
  evidence_spans: FastApiEvidenceSpan[]
  raw_class: string
  raw_score: number
}

type FastApiMemoryResponse = {
  candidates: FastApiMemoryCandidate[]
  count: number
}

async function postJson<T>(url: string, payload: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const body = await res.json()
      message = body?.detail ?? body?.error ?? message
    } catch { }
    throw new Error(message)
  }

  return res.json() as Promise<T>
}

export async function POST(req: Request) {
  try {
    const body = await req.json()
    console.log("[POST /api/analyze] Received body:", body)

    const text = String(body.text ?? body.content ?? body.message ?? "").trim()
    const patient_id = String(body.patient_id ?? "").trim()
    const session_id = String(body.session_id ?? "").trim()
    const mode = String(body.mode ?? "paste").trim() || "paste"

    if (!text) {
      return NextResponse.json(
        { error: "text is required" },
        { status: 400 }
      )
    }

    if (!patient_id) {
      return NextResponse.json(
        { error: "patient_id is required" },
        { status: 400 }
      )
    }

    const patient = await db.patient.findUnique({
      where: { id: patient_id },
      select: { id: true },
    })

    if (!patient) {
      return NextResponse.json(
        { error: "Patient not found" },
        { status: 404 }
      )
    }

    const [analysisRes, memoryRes] = await Promise.allSettled([
      postJson<FastApiAnalyzeResponse>(`${FASTAPI_BASE_URL}/analyze`, {
        text,
        patient_id,
        mode,
      }),
      postJson<FastApiMemoryResponse>(`${FASTAPI_BASE_URL}/extract-memory`, {
        text,
        patient_id,
      }),
    ])

    if (analysisRes.status === "rejected") {
      throw analysisRes.reason
    }

    const analysisData = analysisRes.value
    const memoryData =
      memoryRes.status === "fulfilled"
        ? memoryRes.value
        : { candidates: [], count: 0 }

    const result = await db.$transaction(async (tx) => {
      let session;

      if (session_id && mode === "chat") {
        session = await tx.session.findUnique({
          where: { id: session_id },
          select: { id: true, patient_id: true, created_at: true }
        })

        if (session) {
          // Update the session text
          await tx.session.update({
            where: { id: session_id },
            data: { raw_text: text }
          })

          // Clear old analysis related data to overwrite
          await tx.evidenceSpan.deleteMany({
            where: { analysis: { session_id } }
          })
          
          // Delete ONLY pending candidates for this session.
          // Accepted memories (Memory table) are preserved.
          await tx.memoryCandidate.deleteMany({
            where: { session_id, status: "pending" }
          })
        }
      }

      if (!session) {
        session = await tx.session.create({
          data: {
            patient_id,
            source_type: mode,
            raw_text: text,
          },
          select: {
            id: true,
            patient_id: true,
            raw_text: true,
            source_type: true,
            created_at: true,
          },
        })
      }

      const analysis = await tx.analysis.upsert({
        where: { session_id: session.id },
        update: {
          risk_tier: analysisData.risk_tier,
          context_label: analysisData.context_label,
          signal_labels: analysisData.signal_labels,
          confidence: analysisData.confidence,
          summary: analysisData.summary,
          raw_class: analysisData.raw_class,
          raw_score: analysisData.raw_score,
          evidenceSpans: {
            create: analysisData.evidence_spans.map((span) => ({
              text: span.text,
              label: span.label,
              score: span.score,
              start_idx: span.start_idx,
              end_idx: span.end_idx,
            })),
          },
        },
        create: {
          session_id: session.id,
          risk_tier: analysisData.risk_tier,
          context_label: analysisData.context_label,
          signal_labels: analysisData.signal_labels,
          confidence: analysisData.confidence,
          summary: analysisData.summary,
          raw_class: analysisData.raw_class,
          raw_score: analysisData.raw_score,
          evidenceSpans: {
            create: analysisData.evidence_spans.map((span) => ({
              text: span.text,
              label: span.label,
              score: span.score,
              start_idx: span.start_idx,
              end_idx: span.end_idx,
            })),
          },
        },
        include: {
          evidenceSpans: true,
        },
      })

      let createdCandidates: Array<{
        id: string
        patient_id: string
        session_id: string
        type: string
        title: string
        description: string
        confidence: number
        status: string
      }> = []

      if (memoryData.candidates.length > 0) {
        createdCandidates = await Promise.all(
          memoryData.candidates.map((candidate) =>
            tx.memoryCandidate.create({
              data: {
                patient_id,
                session_id: session.id,
                type: candidate.type,
                title: candidate.title,
                description: candidate.description,
                confidence: candidate.confidence,
                status: "pending",
              },
            })
          )
        )
      }

      await tx.patient.update({
        where: { id: patient_id },
        data: {
          last_session: session.created_at,
        },
      })

      return {
        session,
        analysis,
        candidates: createdCandidates,
      }
    })

    return NextResponse.json({
      risk_tier: analysisData.risk_tier,
      context_label: analysisData.context_label,
      signal_labels: analysisData.signal_labels,
      confidence: analysisData.confidence,
      summary: analysisData.summary,
      evidence_spans: result.analysis.evidenceSpans.map((span) => ({
        text: span.text,
        label: span.label,
        score: span.score,
        start_idx: span.start_idx,
        end_idx: span.end_idx,
      })),
      raw_class: analysisData.raw_class,
      raw_score: analysisData.raw_score,
      session_id: result.session.id,
      analysis_id: result.analysis.id,
      candidates: result.candidates.map((candidate) => ({
        id: candidate.id,
        type: candidate.type,
        title: candidate.title,
        description: candidate.description,
        confidence: candidate.confidence,
      })),
    })
  } catch (error) {
    console.error("[POST /api/analyze] Error", error)
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to analyze text",
      },
      { status: 500 }
    )
  }
}