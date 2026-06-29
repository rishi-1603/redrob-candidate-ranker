import streamlit as st
import json, csv, io, sys
sys.path.insert(0, ".")
from rank import score_candidate

st.set_page_config(page_title="Redrob AI Ranker", page_icon="🎯")
st.title("🎯 Redrob AI Candidate Ranker")
st.write("Upload a candidates JSONL file (≤100 candidates) to get a ranked shortlist.")

uploaded = st.file_uploader("Upload candidates.jsonl", type=["jsonl", "json"])

if uploaded:
    lines = uploaded.read().decode("utf-8").strip().split("\n")
    candidates = [json.loads(line) for line in lines if line.strip()]
    st.info(f"Loaded {len(candidates)} candidates. Scoring...")

    results = [score_candidate(c) for c in candidates]
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top = results[:100]

    st.success(f"Done! Showing top {len(top)} candidates.")
    st.dataframe({
        "Rank": list(range(1, len(top)+1)),
        "Candidate ID": [r["candidate_id"] for r in top],
        "Score": [round(r["score"], 4) for r in top],
        "Reasoning": [r["reasoning"] for r in top],
    })

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for i, r in enumerate(top, 1):
        writer.writerow([r["candidate_id"], i, f"{r['score']:.4f}", r["reasoning"]])

    st.download_button("⬇️ Download submission.csv", out.getvalue(), "submission.csv", "text/csv")