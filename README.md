# Therapist Assistant

An explainable AI-assisted mental health risk triage system for therapists and clinical reviewers.  
It analyzes patient-written text, highlights risk signals, explains predictions, and helps manage structured patient memory across sessions.

## Features

- Mental health risk classification from text
- Risk tier mapping: Low / Moderate / High / Crisis
- Context detection: self-directed, third-person, support-seeking, ambiguous
- Evidence highlights for explainable predictions
- Patient workspace with session history
- Structured memory extraction and review
- Human-in-the-loop workflow for therapist approval

## Tech Stack

**Frontend**
- Next.js
- TypeScript
- Tailwind CSS
- Zustand
- shadcn/ui

**Backend**
- FastAPI
- Python
- Pydantic

**ML / NLP**
- BERT
- PyTorch
- scikit-learn
- SHAP
- LIME

**Database / Auth**
- Prisma
- Neon DB
- Clerk

## Model

The classifier is trained on Reddit mental health text across 5 classes:

- Control
- Anxiety
- Depression
- BPD
- SuicideWatch

Mapped risk tiers:

| Class | Risk Tier |
|---|---|
| Control | Low |
| Anxiety | Moderate |
| Depression | High |
| BPD | High |
| SuicideWatch | Crisis |

## Workflow

1. Create or select a patient
2. Paste patient text or use chat mode
3. Run analysis
4. Review risk tier, signals, and evidence
5. Accept/edit/reject memory candidates
6. Save session and patient context

## Project Structure

```bash
therapist-assistant/
├── app/
├── components/
├── prisma/
├── lib/
├── therapist-assistant-api/
└── README.md
```

## Run Locally

### Frontend
```bash
npm install
npm run dev
```

### Backend
```bash
cd therapist-assistant-api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Database
```bash
npx prisma generate
npx prisma migrate dev
```

## Environment Variables

```env
DATABASE_URL=
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Status

Built as a PBL project at Manipal University Jaipur.  
Current focus: explainable mental health text analysis, therapist workflow support, and deployment-ready architecture.