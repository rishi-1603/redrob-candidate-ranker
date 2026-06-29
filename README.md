# Redrob Intelligent Candidate Ranking System

**Challenge:** Intelligent Candidate Discovery & Ranking — India Runs Data & AI Challenge  
**Role:** Senior AI Engineer — Founding Team @ Redrob AI  
**Approach:** Multi-signal hybrid scorer with behavioral multiplier

---

## How to reproduce the submission

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ranker
python rank.py --candidates candidates.jsonl --out submission.csv

# 3. Validate format
python validate_submission.py submission.csv
```

**Runtime:** ~15–30 seconds on CPU (16 GB RAM). No GPU, no network required during ranking.

---

## Architecture

The system scores each candidate across **five signals**, then applies a **behavioral multiplier**.

### Signal Weights

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Career | 35% | Title match, product-company AI experience, ranking/retrieval in work history |
| Skills | 30% | Core AI/retrieval skills with proficiency × duration trust score |
| Experience | 15% | Years (JD sweet spot 5–9y), education tier |
| Logistics | 20% | Location (Pune/Noida preferred), notice period, relocation |
| **Behavioral** | **multiplier** | Platform engagement, response rate, GitHub activity, recency |

### Final Score Formula

```
final_score = (career×0.35 + skills×0.30 + exp×0.15 + logistics×0.20) × behavioral_multiplier
```

Behavioral multiplier range: **0.30 – 1.30**

---

## Key Design Decisions

### 1. Career Signal (35%) — the decisive signal
The JD explicitly warns against keyword matching. A Marketing Manager with 9 AI skills listed is not a fit. The career component:
- Rewards **strong AI/ML job titles** (ML Engineer, Search Engineer, NLP Engineer, etc.)
- Scans **career history descriptions** for production AI evidence (ranking, retrieval, embedding, A/B test, model serving)
- **Penalizes pure consulting careers** (TCS, Infosys, Wipro, etc.) — explicit JD disqualifier
- Rewards **product-company AI months** over consulting AI months

### 2. Skills Signal (30%) — with anti-stuffing trust multiplier
Skills are not just keyword-matched. Each skill is weighted by:
- **Proficiency level** (beginner → expert: 0.3 → 1.0)
- **Duration trust** (0 months used → 30% weight; 18+ months → 100%)  
- **Platform assessment scores** override proficiency when available (more objective)

This catches keyword stuffers: listing "Pinecone | expert" with 0 months gets only 30% credit.

### 3. Behavioral Multiplier — availability is not optional
JD quote: *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is not actually available."*

Signals used:
- Days since last active (inactive 180d+ → 0.55× multiplier)
- Recruiter response rate (< 20% → 0.75×; > 70% → 1.08×)
- Open-to-work flag (not open → 0.80×)
- GitHub activity score (≥60 → 1.10×)
- Interview completion rate, profile completeness, saved-by-recruiters

### 4. Honeypot Detection
The dataset contains ~80 honeypots with impossible signals:
- Job duration > total years of experience
- 3+ "expert" skills with 0 months used
- Sum of all job durations > 1.5× total YOE

Detected honeypots receive score = 0.001, ensuring they never appear in top 100.

### 5. Logistics (20%) — location and availability
- Pune/Noida candidates get full location credit
- Hyderabad, Mumbai, Bangalore, Delhi NCR: good credit
- Outside India but willing to relocate: partial
- Notice period >90 days: penalized (JD explicitly flags this)

---

## What this system is NOT doing

- **Not keyword matching** — a candidate with "Python, RAG, Pinecone" in skills but a Marketing Manager career does not rank high
- **Not embedding-based** — the submission spec prohibits API calls and GPU during ranking. A rule-based system with explicit JD reasoning is both faster and more interpretable for Stage 4 evaluation
- **Not over-indexing on pure AI title** — candidates with data engineering backgrounds who demonstrably built production retrieval systems in their descriptions score well even if titled "Backend Engineer"

---

## Repository Structure

```
redrob_ranker/
├── rank.py                    # Main ranker — produces submission.csv
├── validate_submission.py     # Official format validator (from hackathon bundle)
├── requirements.txt           # Python dependencies
├── submission_metadata.yaml   # Team metadata
├── README.md                  # This file
└── candidates.jsonl           # Dataset (not committed — place here before running)
```

---

## Compute Specs

- Platform: CPU only
- RAM: 16 GB sufficient
- Python: 3.10+
- Runtime: ~15–30s for 100K candidates
- No GPU, no network during ranking step

---

## Scoring Logic — Quick Reference

```python
# Career score drivers
STRONG_TITLES = {"ml engineer", "search engineer", "ai engineer", ...}  # 0.40 base
GOOD_TITLES   = {"data scientist", "backend engineer", ...}              # 0.22 base

# Core skills (JD's "must-have" cluster)
CORE_SKILLS = {
    "embeddings", "sentence transformers", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "elasticsearch", "rag", "llms", "fine-tuning llms",
    "vector search", "semantic search", "information retrieval", "bm25",
    "learning to rank", "pgvector", "hugging face transformers", "opensearch"
}

# Behavioral multiplier cap: 0.30 – 1.30
```
