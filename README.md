# TRAXR — AI Resume Skill Gap Analyzer

<p align="center">
  <b>Track. Prove. Advance.</b><br/>
  An AI-assisted system for turning resumes and job descriptions into explainable, actionable skill-gap analysis.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-0E1117?style=for-the-badge&logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/UI-Dark%20Theme-111827?style=for-the-badge" alt="UI" />
</p>

<p align="center">
  <a href="https://traxr-ai.streamlit.app/">
    <img src="https://img.shields.io/badge/%E2%96%B6%20Launch-Live%20App-FF4B4B?style=for-the-badge" alt="Launch Live App" />
  </a>
</p>

---

## ✨ Overview

**TRAXR** is a resume-to-role intelligence app that evaluates how well a candidate aligns with a target opportunity.

Instead of offering generic resume advice, TRAXR is built around a sharper product loop:

- Understand the target role
- Compare resume evidence against job expectations
- Surface matched and missing skills
- Translate the gap into a practical action roadmap

The experience is designed to feel fast, explainable, and demo-ready while maintaining a modular architecture behind the interface.

---

## 🎯 Product Lens

TRAXR is built around a simple principle:

> A strong analysis system should not only show users **where they stand**, but also explain **why** they scored that way and **what to improve next**.

That principle is reflected in four core outputs:

- **Role understanding** — what the job is actually asking for
- **Score interpretation** — how the resume aligns with that role
- **Proof and evidence** — what support exists inside the profile
- **Action roadmap** — what to improve next

---

## 🧩 Component Responsibilities

| Component | Responsibility |
|---|---|
| `app.py` | Main Streamlit entry point. Handles the interface, layout, state flow, dashboard rendering, visual components, and overall orchestration. |
| `llmclient.py` | AI integration layer. Responsible for model-backed tasks such as structured extraction, generation, and LLM-assisted outputs. |
| `models.py` | Schema layer. Defines structured data objects so information moves consistently between parsing, scoring, and rendering. |
| `scorer.py` | Scoring engine. Computes readiness signals, gap severity, and overall score logic. |
| `nlppipeline.py` | Text-processing layer. Cleans, extracts, and normalizes resume and job-description content before it reaches the scoring system. |
| `skilltaxonomy.py` | Skills knowledge base. Stores canonical skills, aliases, categories, and interpretation rules used across analysis. |
| `demodata.py` | Demo fallback layer. Provides stable sample data for demo mode, testing, and safe fallback behavior. |
| `requirements.txt` | Dependency manifest for the project runtime. |
| `.streamlit/` | Streamlit-specific configuration and secret handling for local and cloud deployment contexts. |
| `.env` | Local-only environment configuration for secrets and private keys. |
| `sample_data/` | Controlled sample inputs used for testing, demos, and reference workflows. |

---

## 🏗️ Architecture

TRAXR follows a modular application flow:

```text
User Input
   ↓
app.py
   ↓
nlppipeline.py
   ↓
skilltaxonomy.py
   ↓
scorer.py
   ↓
llmclient.py
   ↓
Dashboard Output

models.py   → shared schemas across parsing, scoring, and AI layers
demodata.py → fallback/demo support across the app
```

### Layer Roles

- **Presentation Layer** — `app.py`
- **Parsing Layer** — `nlppipeline.py`
- **Knowledge Layer** — `skilltaxonomy.py`
- **Scoring Layer** — `scorer.py`
- **AI Layer** — `llmclient.py`
- **Schema Layer** — `models.py`
- **Fallback Layer** — `demodata.py`

This separation keeps the UI expressive while making the analysis logic easier to debug, extend, and refine.

---

## 🎨 Design Intent

TRAXR is designed to feel like a focused AI product rather than a plain internal utility.

Its direction emphasizes:

- Dark, premium visual language
- Explainable analysis instead of black-box outputs
- Role-specific evaluation instead of generic resume tips
- Actionable gap framing instead of passive feedback
- Modular code organization instead of monolithic app logic

The goal is to make the user feel like they are entering an analysis system, not just filling out a form.

---

## 🚀 What Makes It Different

Many resume tools stop at formatting advice or basic keyword checks.

TRAXR pushes further by combining:

- **Resume parsing**
- **Job-description interpretation**
- **Skill taxonomy matching**
- **Score reasoning**
- **Gap visibility**
- **Guided next actions**

That makes it useful not only as a demoable product, but as a foundation for a more serious career-readiness platform.

---

## 🌐 Live Experience

The deployed app is available here:

**[https://traxr-ai.streamlit.app](https://traxr-ai.streamlit.app/)**

---

## 🔤 Project Identity

**TRAXR** stands for:

**Track · Analyze · eXecute**

It reflects the product’s core flow: identify the mismatch, analyze the evidence, and execute on improvement.

---

## 👨‍💻 Author

**Tushar Nipane**

B.Tech CSE student, builder, and systems-focused developer working on AI-first product experiences with a strong emphasis on clarity, utility, and polish.
