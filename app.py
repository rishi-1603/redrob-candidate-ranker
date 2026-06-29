import streamlit as st
import json, csv, io, sys
sys.path.insert(0, ".")
from rank import score_candidate

st.set_page_config(page_title="Redrob AI Ranker", page_icon="🎯")
st.title("🎯 Redrob AI Candidate Ranker")
st.write("Upload a candidates JSONL file (≤100 candidates) to get a ranked shortlist.")

uploaded = st.file_uploader("Upload candidates.jsonl", type=["jsonl", "json"])

if uploaded:
    uploaded.seek(0)
    raw = uploaded.read().decode("utf-8").strip()

    if not raw:
        st.error("Uploaded file is empty.")
        st.stop()

    if uploaded.name.lower().endswith(".jsonl"):
        candidates = []
        for i, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON on line {i}: {e}")
                st.stop()
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {e}")
            st.stop()

        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = [data]
        else:
            st.error("JSON must be an object or a list of objects.")
            st.stop()

    if len(candidates) > 100:
        st.warning("More than 100 candidates uploaded. Only the first 100 will be scored.")
        candidates = candidates[:100]

    st.info(f"Loaded {len(candidates)} candidates. Scoring...")

    results = [score_candidate(c) for c in candidates]
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top = results[:100]

    st.success(f"Done! Showing top {len(top)} candidates.")
    st.dataframe({
        "Rank": list(range(1, len(top) + 1)),
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