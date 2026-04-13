import "dotenv/config"
import { PrismaClient } from "@prisma/client"
import { Pool } from "pg"
import { PrismaPg } from "@prisma/adapter-pg"

const pool = new Pool({
  connectionString: process.env.DATABASE_URL
})
const adapter = new PrismaPg(pool)

const prisma = new PrismaClient({ adapter })

async function main() {
    console.log("🌱 Seeding database...")

    // Delete in dependency order
    await prisma.evidenceSpan.deleteMany()
    await prisma.analysis.deleteMany()
    await prisma.memoryCandidate.deleteMany()
    await prisma.memory.deleteMany()
    await prisma.session.deleteMany()
    await prisma.patient.deleteMany()

    // ────────────────────────────────────────────────────────────────────────────
    // Patient 1
    // ────────────────────────────────────────────────────────────────────────────
    const p1 = await prisma.patient.create({
        data: {
            display_name: "Marcus Webb",
            notes: "Sleep issues, social withdrawal, work-related stress.",
            created_at: new Date("2026-04-01T09:30:00Z"),
            last_session: new Date("2026-04-12T15:10:00Z"),
        },
    })

    const p1s1 = await prisma.session.create({
        data: {
            patient_id: p1.id,
            source_type: "paste",
            raw_text:
                "I haven't really been sleeping. I feel exhausted all day and I keep canceling plans with people.",
            therapist_notes: "Initial note from intake reflection.",
            created_at: new Date("2026-04-10T14:00:00Z"),
        },
    })

    await prisma.analysis.create({
        data: {
            session_id: p1s1.id,
            risk_tier: "Moderate",
            context_label: "self-directed",
            signal_labels: ["insomnia", "social withdrawal", "fatigue"],
            confidence: "medium",
            summary:
                "Patient reports poor sleep, low energy, and pulling away socially, suggesting moderate distress.",
            raw_class: "moderate_distress",
            raw_score: 0.71,
            evidenceSpans: {
                create: [
                    {
                        text: "haven't really been sleeping",
                        label: "insomnia",
                        score: 0.83,
                        start_idx: 7,
                        end_idx: 35,
                    },
                    {
                        text: "canceling plans with people",
                        label: "social withdrawal",
                        score: 0.76,
                        start_idx: 74,
                        end_idx: 101,
                    },
                ],
            },
        },
    })

    const p1s2 = await prisma.session.create({
        data: {
            patient_id: p1.id,
            source_type: "paste",
            raw_text:
                "Some days it feels like nothing will get better. I'm not planning to do anything, but I feel very low.",
            therapist_notes: "Follow-up after difficult week.",
            created_at: new Date("2026-04-12T15:10:00Z"),
        },
    })

    await prisma.analysis.create({
        data: {
            session_id: p1s2.id,
            risk_tier: "High",
            context_label: "self-directed",
            signal_labels: ["hopelessness", "depressive language"],
            confidence: "strong",
            summary:
                "Patient expresses hopelessness and sustained low mood. Direct review is recommended.",
            raw_class: "high_risk_language",
            raw_score: 0.89,
            evidenceSpans: {
                create: [
                    {
                        text: "nothing will get better",
                        label: "hopelessness",
                        score: 0.91,
                        start_idx: 24,
                        end_idx: 47,
                    },
                ],
            },
        },
    })

    await prisma.memory.createMany({
        data: [
            {
                patient_id: p1.id,
                type: "recurring_theme",
                title: "Sleep difficulties",
                description: "Ongoing trouble sleeping with daytime exhaustion.",
                first_seen_session_id: p1s1.id,
                last_seen_session_id: p1s2.id,
                therapist_verified: true,
                status: "active",
                created_at: new Date("2026-04-10T14:02:00Z"),
            },
            {
                patient_id: p1.id,
                type: "protective_factor",
                title: "Denies immediate intent",
                description: "States no immediate plan to act despite feeling very low.",
                first_seen_session_id: p1s2.id,
                last_seen_session_id: p1s2.id,
                therapist_verified: true,
                status: "active",
                created_at: new Date("2026-04-12T15:12:00Z"),
            },
        ],
    })

    await prisma.memoryCandidate.create({
        data: {
            patient_id: p1.id,
            session_id: p1s2.id,
            type: "recurring_theme",
            title: "Social isolation pattern",
            description: "Repeatedly mentions cancelling plans and withdrawing from others.",
            confidence: 0.84,
            status: "pending",
        },
    })

    // ────────────────────────────────────────────────────────────────────────────
    // Patient 2
    // ────────────────────────────────────────────────────────────────────────────
    const p2 = await prisma.patient.create({
        data: {
            display_name: "Aisha Khan",
            notes: "Academic stress, perfectionism, anxiety around deadlines.",
            created_at: new Date("2026-04-03T10:00:00Z"),
            last_session: new Date("2026-04-11T11:30:00Z"),
        },
    })

    const p2s1 = await prisma.session.create({
        data: {
            patient_id: p2.id,
            source_type: "paste",
            raw_text:
                "I keep thinking if I don't do everything perfectly, I'll disappoint everyone.",
            therapist_notes: "Concerned about grades and expectations.",
            created_at: new Date("2026-04-11T11:30:00Z"),
        },
    })

    await prisma.analysis.create({
        data: {
            session_id: p2s1.id,
            risk_tier: "Low",
            context_label: "self-directed",
            signal_labels: ["anxiety", "perfectionism"],
            confidence: "medium",
            summary:
                "Patient shows anxious thinking and perfectionism tied to fear of disappointing others.",
            raw_class: "anxious_cognition",
            raw_score: 0.42,
            evidenceSpans: {
                create: [
                    {
                        text: "do everything perfectly",
                        label: "perfectionism",
                        score: 0.79,
                        start_idx: 32,
                        end_idx: 55,
                    },
                ],
            },
        },
    })

    await prisma.memory.create({
        data: {
            patient_id: p2.id,
            type: "recurring_theme",
            title: "Fear of disappointing others",
            description: "Self-worth appears closely tied to performance and others' approval.",
            first_seen_session_id: p2s1.id,
            last_seen_session_id: p2s1.id,
            therapist_verified: true,
            status: "active",
        },
    })

    await prisma.memoryCandidate.create({
        data: {
            patient_id: p2.id,
            session_id: p2s1.id,
            type: "protective_factor",
            title: "Strong academic motivation",
            description: "Despite anxiety, remains engaged and future-oriented academically.",
            confidence: 0.73,
            status: "pending",
        },
    })

    // ────────────────────────────────────────────────────────────────────────────
    // Patient 3
    // ────────────────────────────────────────────────────────────────────────────
    const p3 = await prisma.patient.create({
        data: {
            display_name: "Rohan Mehta",
            notes: "Breakup-related grief, low appetite, reduced concentration.",
            created_at: new Date("2026-04-05T08:45:00Z"),
            last_session: new Date("2026-04-13T09:15:00Z"),
        },
    })

    const p3s1 = await prisma.session.create({
        data: {
            patient_id: p3.id,
            source_type: "paste",
            raw_text:
                "Ever since the breakup I can't focus on anything and I barely feel like eating.",
            therapist_notes: "Recent breakup seems central to current symptoms.",
            created_at: new Date("2026-04-13T09:15:00Z"),
        },
    })

    await prisma.analysis.create({
        data: {
            session_id: p3s1.id,
            risk_tier: "Moderate",
            context_label: "self-directed",
            signal_labels: ["grief", "appetite change", "poor concentration"],
            confidence: "medium",
            summary:
                "Patient describes grief symptoms with concentration problems and reduced appetite.",
            raw_class: "moderate_distress",
            raw_score: 0.63,
            evidenceSpans: {
                create: [
                    {
                        text: "can't focus on anything",
                        label: "poor concentration",
                        score: 0.74,
                        start_idx: 27,
                        end_idx: 50,
                    },
                    {
                        text: "barely feel like eating",
                        label: "appetite change",
                        score: 0.77,
                        start_idx: 57,
                        end_idx: 80,
                    },
                ],
            },
        },
    })

    await prisma.memory.create({
        data: {
            patient_id: p3.id,
            type: "life_event",
            title: "Recent breakup",
            description: "Breakup appears central to current emotional and functional change.",
            first_seen_session_id: p3s1.id,
            last_seen_session_id: p3s1.id,
            therapist_verified: true,
            status: "active",
        },
    })

    await prisma.memoryCandidate.create({
        data: {
            patient_id: p3.id,
            session_id: p3s1.id,
            type: "life_event",
            title: "Relationship loss affecting daily functioning",
            description: "Breakup is linked to concentration issues and appetite change.",
            confidence: 0.9,
            status: "pending",
        },
    })

    console.log("✅ Seed complete")
}

main()
    .catch((e) => {
        console.error("❌ Seed failed")
        console.error(e)
        process.exit(1)
    })
    .finally(async () => {
        await prisma.$disconnect()
    })