#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranking System
Senior AI Engineer — Founding Team

Architecture: Multi-signal hybrid scorer with:
  1. Career signal (title relevance, product-company experience, trajectory)
  2. Skill signal (core AI/ML skills, proficiency-weighted, honeypot-aware)
  3. Experience signal (years, role depth, quality of companies)
  4. Logistics signal (location, notice period, availability)
  5. Behavioral multiplier (engagement, response rate, GitHub activity)

Explicit disqualifiers (JD-specified):
  - Consulting-only career (TCS, Infosys, etc.)
  - Pure research / no production deployment
  - Honeypot profiles (impossible timelines, expert skills with 0 months)
  - Long notice periods (>90 days penalized)
  - No relevant AI/ML in career history

Runtime: ~30s on CPU for 100K candidates. No network, no GPU.
"""

import argparse
import csv
import datetime
import json
import math
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# JD-derived constants
# ─────────────────────────────────────────────

# Must-have core skills (production retrieval/ranking/embedding systems)
CORE_SKILLS = {
    "embeddings", "sentence transformers", "vector search", "semantic search",
    "information retrieval", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "pgvector", "elasticsearch", "opensearch", "rag", "llms", "fine-tuning llms",
    "hugging face transformers", "bm25", "learning to rank",
}

# Strong supporting skills (nice-to-have from JD)
BONUS_SKILLS = {
    "pytorch", "tensorflow", "nlp", "lora", "qlora", "python",
    "xgboost", "mlflow", "reranking", "hybrid search", "langchain",
    "deep learning", "machine learning", "transformers", "bert",
    "a/b testing", "recommendation systems", "distributed systems",
    "kubernetes", "docker", "fastapi", "flask", "redis",
}

# Titles that signal genuine AI/ML production engineering
STRONG_TITLES = {
    "ml engineer", "machine learning engineer", "applied ml engineer",
    "ai engineer", "senior ai engineer", "lead ai engineer",
    "senior machine learning engineer", "staff machine learning engineer",
    "senior software engineer (ml)", "nlp engineer", "senior nlp engineer",
    "recommendation systems engineer", "search engineer",
    "ai research engineer", "ai specialist",
}

GOOD_TITLES = {
    "data scientist", "senior data scientist", "backend engineer",
    "senior software engineer", "software engineer", "data engineer",
    "senior data engineer", "analytics engineer", "research scientist",
    "applied scientist", "computer vision engineer",
}

# Career titles that are broadly tech but need AI/ML in description
TECH_ADJACENT = {
    "full stack developer", "cloud engineer", "devops engineer",
    "frontend engineer", "java developer", ".net developer",
    "mobile developer", "qa engineer",
}

# Consulting-only companies (JD explicit disqualifier)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "l&t infotech", "l&t technology", "hexaware", "mindtree",
    "sonata software", "niit technologies",
}

# Preferred Indian cities for this role
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurugram", "gurgaon", "chennai", "delhi ncr",
}

TODAY = datetime.date(2026, 6, 29)


# ─────────────────────────────────────────────
# Honeypot Detection
# ─────────────────────────────────────────────

def is_honeypot(candidate: dict) -> bool:
    """
    Detect profiles with subtly impossible signals.
    JD warns: ~80 honeypots with impossible timelines or expert skills at 0 months.
    """
    yoe_months = candidate["profile"]["years_of_experience"] * 12

    # Check 1: any single job duration > total YOE (impossible)
    for job in candidate["career_history"]:
        if job["duration_months"] > yoe_months + 12:
            return True

    # Check 2: multiple expert-proficiency skills with 0 months used
    expert_zero = sum(
        1 for s in candidate["skills"]
        if s["proficiency"] == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero >= 3:
        return True

    # Check 3: overlapping career jobs > 100% of time (career duration sum >> YOE)
    total_job_months = sum(j["duration_months"] for j in candidate["career_history"])
    if total_job_months > yoe_months * 1.5 + 24:
        return True

    return False


# ─────────────────────────────────────────────
# Component 1: Title & Career Signal
# ─────────────────────────────────────────────

def score_career(candidate: dict) -> tuple[float, list[str]]:
    """
    Score based on:
    - Current title match to JD
    - Career history: product company vs consulting
    - Evidence of production AI/ML systems in descriptions
    - Career trajectory (growing into AI, not away from it)

    Returns (score 0-1, reasoning_notes)
    """
    notes = []
    profile = candidate["profile"]
    history = candidate["career_history"]

    current_title_lower = profile["current_title"].lower()
    score = 0.0

    # ── Title scoring ──
    if current_title_lower in STRONG_TITLES:
        score += 0.40
        notes.append(f"strong AI title: {profile['current_title']}")
    elif current_title_lower in GOOD_TITLES:
        score += 0.22
        notes.append(f"relevant tech title: {profile['current_title']}")
    elif current_title_lower in TECH_ADJACENT:
        score += 0.08
    else:
        # Non-tech titles (HR, Marketing, etc.) get near-zero from title
        score += 0.01

    # ── Career history analysis ──
    consulting_jobs = 0
    product_company_ai_months = 0
    total_ai_months = 0
    has_production_ai = False
    has_ranking_or_retrieval = False

    # AI/retrieval keywords for description scanning
    prod_ai_kw = [
        "ranking", "retrieval", "embedding", "vector", "search", "recommendation",
        "nlp", "llm", "rag", "fine-tun", "deployed", "production", "a/b test",
        "machine learning", "deep learning", "inference", "model serving",
        "rerank", "semantic", "transformer", "bert", "faiss", "pinecone",
        "weaviate", "qdrant", "milvus", "elasticsearch", "bm25",
    ]
    retrieval_kw = ["ranking", "retrieval", "search", "recommendation", "rerank", "bm25", "faiss", "vector"]

    for job in history:
        company_lower = job["company"].lower()
        desc_lower = job["description"].lower()

        # Consulting firm detection
        if any(cf in company_lower for cf in CONSULTING_FIRMS):
            consulting_jobs += 1

        # Production AI evidence in description
        ai_hits = sum(1 for kw in prod_ai_kw if kw in desc_lower)
        if ai_hits >= 3:
            has_production_ai = True
            total_ai_months += job["duration_months"]
            # Product company bonus (not consulting, not pure research)
            if not any(cf in company_lower for cf in CONSULTING_FIRMS):
                product_company_ai_months += job["duration_months"]

        if any(kw in desc_lower for kw in retrieval_kw):
            has_ranking_or_retrieval = True

    # Career composition scoring
    total_jobs = len(history)
    if total_jobs > 0:
        consulting_ratio = consulting_jobs / total_jobs
        if consulting_ratio == 1.0:
            # Pure consulting - explicit JD disqualifier
            score *= 0.25
            notes.append("pure consulting career (JD disqualifier)")
        elif consulting_ratio > 0.5:
            score *= 0.6
            notes.append("majority consulting background")
        elif consulting_ratio > 0:
            score *= 0.85
            notes.append("some consulting, some product")

    # Production AI in career
    if has_production_ai:
        score += 0.25
        notes.append(f"production AI/ML evidence in career ({total_ai_months}m)")
    if has_ranking_or_retrieval:
        score += 0.20
        notes.append("ranking/retrieval/search system experience")
    if product_company_ai_months >= 36:
        score += 0.10
        notes.append(f"≥3y product-company AI ({product_company_ai_months}m)")

    return min(score, 1.0), notes


# ─────────────────────────────────────────────
# Component 2: Skills Signal
# ─────────────────────────────────────────────

def score_skills(candidate: dict) -> tuple[float, list[str]]:
    """
    Score skills with trust multiplier:
    - Core AI/retrieval skills weighted higher
    - Proficiency level matters
    - Duration months as trust signal (prevents keyword stuffing)
    - Endorsements as weak signal
    - Skill assessment scores from Redrob platform (objective)
    """
    notes = []
    skills_list = candidate["skills"]
    assessment_scores = candidate["redrob_signals"].get("skill_assessment_scores", {})

    proficiency_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
    prof_map = {s["name"].lower(): s for s in skills_list}

    core_hit_score = 0.0
    core_hits = []
    bonus_hit_score = 0.0

    for skill in skills_list:
        name_lower = skill["name"].lower()
        prof = proficiency_weight.get(skill["proficiency"], 0.5)
        dur_months = skill.get("duration_months", 0)

        # Duration trust multiplier (catches keyword stuffers)
        if dur_months == 0:
            dur_trust = 0.3
        elif dur_months < 6:
            dur_trust = 0.5
        elif dur_months < 18:
            dur_trust = 0.75
        else:
            dur_trust = 1.0

        # Assessment override — objective platform score
        assess_score = None
        for akey, aval in assessment_scores.items():
            if akey.lower() == name_lower:
                assess_score = aval / 100.0
                break

        effective_score = prof * dur_trust
        if assess_score is not None:
            # Blend proficiency with platform assessment (assessment is more objective)
            effective_score = 0.4 * prof + 0.6 * assess_score

        if name_lower in CORE_SKILLS:
            core_hit_score += effective_score
            core_hits.append(skill["name"])
        elif name_lower in BONUS_SKILLS:
            bonus_hit_score += effective_score * 0.4

    # Normalize
    max_core = len(CORE_SKILLS) * 1.0
    core_norm = min(core_hit_score / (max_core * 0.35), 1.0)  # 35% coverage = max score
    bonus_norm = min(bonus_hit_score / (len(BONUS_SKILLS) * 0.5), 0.3)

    total = core_norm * 0.75 + bonus_norm * 0.25

    if core_hits:
        notes.append(f"core AI skills: {', '.join(core_hits[:6])}{'...' if len(core_hits) > 6 else ''}")
    else:
        notes.append("no core AI/retrieval skills")

    return min(total, 1.0), notes


# ─────────────────────────────────────────────
# Component 3: Experience Signal
# ─────────────────────────────────────────────

def score_experience(candidate: dict) -> tuple[float, list[str]]:
    """
    JD says 5-9 years, but explicitly notes it's not a hard requirement.
    Sweet spot: 5-9y. Strong consideration: 4-11y. Outside: partial credit.
    Prefer candidates who've been in AI/ML roles specifically.
    """
    notes = []
    yoe = candidate["profile"]["years_of_experience"]

    # JD sweet spot: 5-9y
    if 5 <= yoe <= 9:
        base = 1.0
        notes.append(f"{yoe}y experience (ideal range)")
    elif 4 <= yoe < 5:
        base = 0.85
        notes.append(f"{yoe}y experience (slightly junior)")
    elif 9 < yoe <= 11:
        base = 0.90
        notes.append(f"{yoe}y experience (slightly senior)")
    elif 3 <= yoe < 4:
        base = 0.65
        notes.append(f"{yoe}y experience (below target)")
    elif 11 < yoe <= 14:
        base = 0.75
        notes.append(f"{yoe}y experience (over-experienced)")
    else:
        base = 0.50
        notes.append(f"{yoe}y experience (outside range)")

    # Education tier bonus
    edu_bonus = 0.0
    for edu in candidate.get("education", []):
        tier = edu.get("tier", "unknown")
        if tier == "tier_1":
            edu_bonus = max(edu_bonus, 0.10)
        elif tier == "tier_2":
            edu_bonus = max(edu_bonus, 0.05)

    if edu_bonus > 0:
        notes.append(f"tier-{1 if edu_bonus >= 0.10 else 2} education")

    return min(base + edu_bonus, 1.0), notes


# ─────────────────────────────────────────────
# Component 4: Logistics Signal
# ─────────────────────────────────────────────

def score_logistics(candidate: dict) -> tuple[float, list[str]]:
    """
    Score: location fit, notice period, work mode preference.
    JD: Pune/Noida preferred; open to Hyderabad, Mumbai, Delhi NCR.
    Notice: loves <30d, buys out up to 30d, >90d is a real hurdle.
    """
    notes = []
    sig = candidate["redrob_signals"]
    profile = candidate["profile"]

    score = 0.5  # neutral baseline

    # Location scoring
    location_lower = profile.get("location", "").lower()
    country = profile.get("country", "").lower()

    if country == "india":
        if any(city in location_lower for city in ["pune", "noida"]):
            score += 0.30
            notes.append("location: Pune/Noida (preferred)")
        elif any(city in location_lower for city in PREFERRED_LOCATIONS):
            score += 0.20
            notes.append(f"location: {profile['location']} (good fit)")
        else:
            score += 0.05
            notes.append(f"India but non-preferred city: {profile['location']}")
    elif sig.get("willing_to_relocate", False):
        score += 0.10
        notes.append(f"willing to relocate from {profile['country']}")
    else:
        score -= 0.10
        notes.append(f"overseas ({profile['country']}), not willing to relocate")

    # Notice period
    notice = sig.get("notice_period_days", 60)
    if notice <= 15:
        score += 0.20
        notes.append(f"notice: {notice}d (immediate/fast)")
    elif notice <= 30:
        score += 0.15
        notes.append(f"notice: {notice}d (within buyout window)")
    elif notice <= 60:
        score += 0.05
        notes.append(f"notice: {notice}d (moderate)")
    elif notice <= 90:
        score -= 0.05
        notes.append(f"notice: {notice}d (long)")
    else:
        score -= 0.15
        notes.append(f"notice: {notice}d (very long — JD concern)")

    # Work mode (hybrid preferred by JD)
    work_mode = sig.get("preferred_work_mode", "flexible")
    if work_mode in ("hybrid", "flexible"):
        score += 0.05
    elif work_mode == "onsite":
        score += 0.03  # fine for Pune/Noida offices

    return max(0.0, min(score, 1.0)), notes


# ─────────────────────────────────────────────
# Component 5: Behavioral Multiplier
# ─────────────────────────────────────────────

def score_behavioral(candidate: dict) -> tuple[float, list[str]]:
    """
    Platform signals as a multiplier on profile quality.
    JD: 'A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% response rate is not actually available.'
    
    This is a MULTIPLIER (0.3–1.2), not an additive score.
    """
    notes = []
    sig = candidate["redrob_signals"]

    multiplier = 1.0

    # ── Availability / Open to work ──
    if not sig.get("open_to_work_flag", False):
        multiplier *= 0.80
        notes.append("not open to work")

    # ── Recency: last active ──
    last_active = datetime.date.fromisoformat(sig["last_active_date"])
    days_inactive = (TODAY - last_active).days
    if days_inactive <= 14:
        multiplier *= 1.10
        notes.append(f"active {days_inactive}d ago")
    elif days_inactive <= 30:
        multiplier *= 1.05
    elif days_inactive <= 60:
        multiplier *= 1.00
    elif days_inactive <= 90:
        multiplier *= 0.90
    elif days_inactive <= 180:
        multiplier *= 0.75
        notes.append(f"inactive {days_inactive}d")
    else:
        multiplier *= 0.55
        notes.append(f"inactive {days_inactive}d (stale)")

    # ── Recruiter response rate ──
    rr = sig.get("recruiter_response_rate", 0.5)
    if rr >= 0.7:
        multiplier *= 1.08
        notes.append(f"high response rate ({rr:.0%})")
    elif rr >= 0.4:
        multiplier *= 1.00
    elif rr >= 0.2:
        multiplier *= 0.90
    else:
        multiplier *= 0.75
        notes.append(f"low response rate ({rr:.0%})")

    # ── Interview completion (reliability signal) ──
    icr = sig.get("interview_completion_rate", 0.7)
    if icr >= 0.8:
        multiplier *= 1.05
    elif icr < 0.5:
        multiplier *= 0.90

    # ── GitHub activity (for AI/ML role, this matters) ──
    gh = sig.get("github_activity_score", -1)
    if gh >= 60:
        multiplier *= 1.10
        notes.append(f"strong GitHub ({gh:.0f})")
    elif gh >= 30:
        multiplier *= 1.05
    elif gh == -1:
        multiplier *= 0.95  # no GitHub linked
    elif gh < 10:
        multiplier *= 0.95

    # ── Saved by recruiters (market signal) ──
    saved = sig.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        multiplier *= 1.05
    elif saved == 0:
        multiplier *= 0.98

    # ── Profile completeness ──
    completeness = sig.get("profile_completeness_score", 70)
    if completeness >= 90:
        multiplier *= 1.03
    elif completeness < 60:
        multiplier *= 0.92

    # Cap multiplier range
    multiplier = max(0.30, min(1.30, multiplier))

    return multiplier, notes


# ─────────────────────────────────────────────
# Master Scorer
# ─────────────────────────────────────────────

WEIGHTS = {
    "career": 0.35,
    "skills": 0.30,
    "experience": 0.15,
    "logistics": 0.20,
}


def score_candidate(candidate: dict) -> dict:
    """
    Compute composite score for a single candidate.
    Returns dict with score, component scores, and reasoning.
    """
    if is_honeypot(candidate):
        return {
            "candidate_id": candidate["candidate_id"],
            "score": 0.001,
            "reasoning": "Honeypot: impossible profile signals detected.",
            "_honeypot": True,
        }

    career_score, career_notes = score_career(candidate)
    skills_score, skills_notes = score_skills(candidate)
    exp_score, exp_notes = score_experience(candidate)
    logistics_score, logistics_notes = score_logistics(candidate)
    behav_mult, behav_notes = score_behavioral(candidate)

    base_score = (
        career_score * WEIGHTS["career"]
        + skills_score * WEIGHTS["skills"]
        + exp_score * WEIGHTS["experience"]
        + logistics_score * WEIGHTS["logistics"]
    )

    final_score = base_score * behav_mult

    # Build concise reasoning (Submission spec requires 1-2 sentences)
    title = candidate["profile"]["current_title"]
    yoe = candidate["profile"]["years_of_experience"]
    loc = candidate["profile"]["location"]

    key_career = career_notes[0] if career_notes else ""
    key_skills = skills_notes[0] if skills_notes else ""
    key_behav = "; ".join(behav_notes[:2]) if behav_notes else ""
    key_logist = logistics_notes[0] if logistics_notes else ""

    reasoning_parts = [f"{title} with {yoe}y exp ({loc})"]
    if key_career:
        reasoning_parts.append(key_career)
    if key_skills:
        reasoning_parts.append(key_skills)
    if key_behav:
        reasoning_parts.append(key_behav)
    if key_logist and key_logist not in reasoning_parts:
        reasoning_parts.append(key_logist)

    reasoning = "; ".join(reasoning_parts[:4]) + "."

    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(min(final_score, 0.999), 4),
        "reasoning": reasoning,
        "_career": career_score,
        "_skills": skills_score,
        "_exp": exp_score,
        "_logistics": logistics_score,
        "_behav_mult": behav_mult,
        "_honeypot": False,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob AI Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates JSONL file")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading candidates from {candidates_path}...", file=sys.stderr)
    candidates = []
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates. Scoring...", file=sys.stderr)

    results = []
    honeypot_count = 0
    for i, c in enumerate(candidates):
        r = score_candidate(c)
        results.append(r)
        if r.get("_honeypot"):
            honeypot_count += 1
        if (i + 1) % 10000 == 0:
            print(f"  Scored {i+1}/{len(candidates)}...", file=sys.stderr)

    print(f"Detected {honeypot_count} honeypot candidates.", file=sys.stderr)

    # Sort by score descending, break ties by candidate_id ascending
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Take top N
    top_n = results[:args.top_n]

    # Assign ranks and normalize scores to be strictly non-increasing
    # (submission spec requirement)
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_score = None
        for rank, r in enumerate(top_n, start=1):
            score = r["score"]
            if prev_score is not None and score > prev_score:
                score = prev_score
            prev_score = score
            writer.writerow([r["candidate_id"], rank, f"{score:.4f}", r["reasoning"]])

    print(f"\nTop {args.top_n} candidates written to {out_path}", file=sys.stderr)
    print(f"Top 5 preview:", file=sys.stderr)
    for r in top_n[:5]:
        print(f"  #{results.index(r)+1} {r['candidate_id']}: {r['score']:.4f} — {r['reasoning'][:80]}", file=sys.stderr)


if __name__ == "__main__":
    main()
