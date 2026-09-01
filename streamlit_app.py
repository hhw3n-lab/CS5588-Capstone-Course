import json
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI-Guided Job Search", page_icon="🔎", layout="wide")

JOBS = [
    {"job_id":"J001","title":"Junior Data Scientist","company":"HealthAI Labs","location":"Kansas City, MO","work_mode":"Hybrid","min_experience":1,"required_skills":["python","sql","pandas","machine learning"],"preferred_skills":["healthcare","scikit-learn","git"],"description":"Build predictive models and analytics pipelines for healthcare data using Python, SQL, pandas, and machine learning."},
    {"job_id":"J002","title":"Data Analyst","company":"Metro Analytics","location":"Kansas City, MO","work_mode":"On-site","min_experience":1,"required_skills":["sql","excel","data visualization"],"preferred_skills":["python","tableau"],"description":"Analyze business data, create SQL reports, dashboards, and visualizations, and communicate findings to stakeholders."},
    {"job_id":"J003","title":"Machine Learning Engineer","company":"VectorWorks AI","location":"Remote","work_mode":"Remote","min_experience":3,"required_skills":["python","pytorch","docker","machine learning"],"preferred_skills":["aws","mlops","git"],"description":"Develop production machine learning systems, model training pipelines, APIs, containers, and cloud deployments."},
    {"job_id":"J004","title":"Research Data Scientist","company":"BioDiscovery Institute","location":"St. Louis, MO","work_mode":"Hybrid","min_experience":2,"required_skills":["python","statistics","machine learning"],"preferred_skills":["healthcare","nlp","pandas"],"description":"Apply statistical learning and AI methods to biomedical research datasets and collaborate with domain scientists."},
    {"job_id":"J005","title":"Business Intelligence Analyst","company":"Civic Data Group","location":"Remote","work_mode":"Remote","min_experience":2,"required_skills":["sql","power bi","data visualization"],"preferred_skills":["python","data modeling"],"description":"Build dashboards, semantic models, SQL transformations, and KPI reporting for distributed business teams."},
    {"job_id":"J006","title":"AI Software Intern","company":"Agentic Systems","location":"Remote","work_mode":"Remote","min_experience":0,"required_skills":["python","git"],"preferred_skills":["llm","hugging face","agents"],"description":"Prototype AI applications using Python, open-source models, GitHub workflows, and agentic development tools."},
    {"job_id":"J007","title":"Senior Data Scientist","company":"FinModel","location":"Chicago, IL","work_mode":"Hybrid","min_experience":5,"required_skills":["python","sql","statistics","machine learning"],"preferred_skills":["spark","aws","leadership"],"description":"Lead modeling projects, mentor data scientists, and design scalable machine learning solutions for financial products."},
    {"job_id":"J008","title":"Healthcare Data Analyst","company":"CareMetrics","location":"Remote","work_mode":"Remote","min_experience":1,"required_skills":["sql","python","data visualization"],"preferred_skills":["healthcare","pandas"],"description":"Analyze healthcare quality data with SQL and Python, prepare visual reports, and support clinical analytics teams."},
    {"job_id":"J009","title":"Data Engineering Associate","company":"CloudPipe","location":"Kansas City, MO","work_mode":"Hybrid","min_experience":2,"required_skills":["python","sql","etl"],"preferred_skills":["spark","aws","git"],"description":"Create ETL workflows, data quality checks, SQL transformations, and cloud-oriented data pipelines."},
    {"job_id":"J010","title":"NLP Research Assistant","company":"University AI Lab","location":"Kansas City, MO","work_mode":"On-site","min_experience":0,"required_skills":["python","machine learning"],"preferred_skills":["nlp","hugging face","research"],"description":"Support experiments in natural language processing, machine learning, Hugging Face models, and research evaluation."},
]

DEFAULT_PROFILE = {
    "name":"Sample Student",
    "skills":["python","sql","pandas","scikit-learn","data visualization","machine learning","git"],
    "years_experience":2,
    "interests":["data science","healthcare","ai"],
    "preferred_locations":["Kansas City, MO","Remote"],
    "remote_ok":True,
    "target_titles":["Data Scientist","Data Analyst","ML Engineer"],
}

def norm(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())

def skill_score(user_skills, required, preferred):
    user = {norm(x) for x in user_skills}
    req = {norm(x) for x in required}
    pref = {norm(x) for x in preferred}
    req_s = len(user & req) / max(len(req), 1)
    pref_s = len(user & pref) / max(len(pref), 1) if pref else 0.0
    return 0.8 * req_s + 0.2 * pref_s

def exp_score(user_years, min_years):
    return 1.0 if min_years <= 0 else min(user_years / min_years, 1.0)

def interest_score(interests, job):
    text = " ".join([job["title"], job["company"], job["description"], " ".join(job["required_skills"]), " ".join(job["preferred_skills"])]).lower()
    return 0.0 if not interests else sum(norm(i) in text for i in interests) / len(interests)

def constraint_score(profile, job):
    loc = job["location"] in profile["preferred_locations"]
    remote = profile["remote_ok"] and job["work_mode"].lower() == "remote"
    return 1.0 if loc or remote else 0.25

def match(profile, job, weights):
    parts = {
        "skill_score": skill_score(profile["skills"], job["required_skills"], job["preferred_skills"]),
        "experience_score": exp_score(profile["years_experience"], job["min_experience"]),
        "interest_score": interest_score(profile["interests"], job),
        "constraint_score": constraint_score(profile, job),
    }
    denom = max(sum(weights.values()), 1e-9)
    total = (
        weights["skills"] * parts["skill_score"] +
        weights["experience"] * parts["experience_score"] +
        weights["interests"] * parts["interest_score"] +
        weights["constraints"] * parts["constraint_score"]
    ) / denom
    parts["score"] = float(np.clip(total, 0, 1))
    return parts

def explain(profile, job, weights):
    user = {norm(x) for x in profile["skills"]}
    req = {norm(x) for x in job["required_skills"]}
    pref = {norm(x) for x in job["preferred_skills"]}
    return {
        **match(profile, job, weights),
        "matched_required": sorted(user & req),
        "missing_required": sorted(req - user),
        "matched_preferred": sorted(user & pref),
    }

def rank_jobs(profile, jobs, weights):
    rows = []
    for j in jobs:
        s = match(profile, j, weights)
        rows.append({"job_id":j["job_id"],"title":j["title"],"company":j["company"],"location":j["location"],"work_mode":j["work_mode"],"min_experience":j["min_experience"],**s})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "comparison" not in st.session_state:
    st.session_state.comparison = pd.DataFrame([
        {"Version":"Human baseline","Development time (min)":None,"Tests passed":4,"Top-5 quality (1-5)":None,"Human edits":"Baseline written manually","Main failure":"","What we learned":""},
        {"Version":"AI-guided","Development time (min)":None,"Tests passed":None,"Top-5 quality (1-5)":None,"Human edits":"","Main failure":"","What we learned":""},
        {"Version":"Human-AI co-designed","Development time (min)":None,"Tests passed":None,"Top-5 quality (1-5)":None,"Human edits":"","Main failure":"","What we learned":""},
    ])

st.sidebar.header("User profile")
name = st.sidebar.text_input("Name", DEFAULT_PROFILE["name"])
skills = [x.strip() for x in st.sidebar.text_area("Skills (comma separated)", ", ".join(DEFAULT_PROFILE["skills"]), height=120).split(",") if x.strip()]
years = st.sidebar.slider("Years of experience", 0, 10, DEFAULT_PROFILE["years_experience"])
interests = [x.strip() for x in st.sidebar.text_input("Interests (comma separated)", ", ".join(DEFAULT_PROFILE["interests"])).split(",") if x.strip()]
all_locations = sorted({j["location"] for j in JOBS})
preferred_locations = st.sidebar.multiselect("Preferred locations", all_locations, default=["Kansas City, MO","Remote"])
remote_ok = st.sidebar.checkbox("Remote is acceptable", True)
profile = {"name":name,"skills":skills,"years_experience":years,"interests":interests,"preferred_locations":preferred_locations,"remote_ok":remote_ok,"target_titles":DEFAULT_PROFILE["target_titles"]}

st.sidebar.divider()
st.sidebar.subheader("Explainable score weights")
weights = {
    "skills": st.sidebar.slider("Skills", 0.0, 1.0, 0.45, 0.05),
    "experience": st.sidebar.slider("Experience", 0.0, 1.0, 0.25, 0.05),
    "interests": st.sidebar.slider("Interests", 0.0, 1.0, 0.15, 0.05),
    "constraints": st.sidebar.slider("Constraints", 0.0, 1.0, 0.15, 0.05),
}

st.title("🔎 AI-Guided Job Search Application")
st.caption("CS 5588 Challenge 1 • Human Design → AI Design → Human–AI Co-Design → Feedback & Refinement")
st.info("Teaching demo: the starter uses synthetic job postings. Replace them only with instructor-approved data or APIs.")

t1, t2, t3, t4, t5 = st.tabs(["1. Search & Rank","2. Hugging Face","3. Feedback","4. Manual vs AI","5. Agentic AI + GitHub"])

with t1:
    st.subheader("Search, filter, rank, and explain")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        query = st.text_input("Search title/description", placeholder="e.g., data, healthcare, machine learning")
    with c2:
        loc_filter = st.selectbox("Location", ["All"] + all_locations)
    with c3:
        remote_only = st.checkbox("Remote only")
    filtered = []
    for j in JOBS:
        if query and query.lower() not in (j["title"] + " " + j["description"]).lower():
            continue
        if loc_filter != "All" and j["location"] != loc_filter:
            continue
        if remote_only and j["work_mode"].lower() != "remote":
            continue
        filtered.append(j)
    if not filtered:
        st.warning("No jobs match the current filters.")
    else:
        ranked = rank_jobs(profile, filtered, weights)
        m1, m2, m3 = st.columns(3)
        m1.metric("Jobs considered", len(ranked))
        m2.metric("Top match score", f'{ranked.iloc[0]["score"]:.1%}')
        m3.metric("Profile skills", len(profile["skills"]))
        st.bar_chart(ranked[["title","score"]].head(8).set_index("title"))
        show = ranked.copy()
        for c in ["score","skill_score","experience_score","interest_score","constraint_score"]:
            show[c] = (100 * show[c]).round(1)
        st.dataframe(show.rename(columns={"score":"Match %","skill_score":"Skills %","experience_score":"Experience %","interest_score":"Interests %","constraint_score":"Constraints %"}), use_container_width=True, hide_index=True)
        st.markdown("### Top job explanations")
        for _, row in ranked.head(5).iterrows():
            job = next(j for j in filtered if j["job_id"] == row["job_id"])
            d = explain(profile, job, weights)
            with st.expander(f'{job["title"]} — {job["company"]} • {row["score"]:.1%}'):
                st.write(job["description"])
                a, b = st.columns(2)
                with a:
                    st.markdown("**Matched required skills**")
                    st.write(", ".join(d["matched_required"]) or "None")
                    st.markdown("**Matched preferred skills**")
                    st.write(", ".join(d["matched_preferred"]) or "None")
                with b:
                    st.markdown("**Missing required skills**")
                    st.write(", ".join(d["missing_required"]) or "None")
                    st.markdown("**Experience**")
                    st.write(f'{profile["years_experience"]} years vs. {job["min_experience"]} required')
                    st.markdown("**Location / work mode**")
                    st.write(f'{job["location"]} • {job["work_mode"]}')

with t2:
    st.subheader("Hugging Face semantic matching")
    st.write("Optional semantic matching augments the transparent baseline rather than replacing it.")
    enable_hf = st.checkbox("Load sentence-transformers/all-MiniLM-L6-v2", False)
    if enable_hf:
        try:
            from sentence_transformers import SentenceTransformer
            @st.cache_resource(show_spinner="Loading Hugging Face model...")
            def load_model():
                return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            model = load_model()
            profile_text = " ".join(profile["skills"] + profile["interests"] + profile["target_titles"])
            job_texts = [" ".join([j["title"], j["description"], " ".join(j["required_skills"]), " ".join(j["preferred_skills"])]) for j in JOBS]
            p = model.encode([profile_text], normalize_embeddings=True)
            jemb = model.encode(job_texts, normalize_embeddings=True)
            semantic = (p @ jemb.T).flatten()
            base = rank_jobs(profile, JOBS, weights)
            base_map = dict(zip(base["job_id"], base["score"]))
            df = pd.DataFrame({"job_id":[j["job_id"] for j in JOBS],"title":[j["title"] for j in JOBS],"company":[j["company"] for j in JOBS],"baseline":[base_map[j["job_id"]] for j in JOBS],"semantic":semantic})
            df["hybrid"] = 0.70 * df["baseline"] + 0.30 * df["semantic"]
            df = df.sort_values("hybrid", ascending=False).reset_index(drop=True)
            for c in ["baseline","semantic","hybrid"]:
                df[c] = (100 * df[c]).round(1)
            st.success("Hybrid score = 70% transparent baseline + 30% semantic similarity.")
            st.dataframe(df.rename(columns={"baseline":"Baseline %","semantic":"Semantic %","hybrid":"Hybrid %"}), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error("Could not load the Hugging Face model. Install sentence-transformers and ensure model access is available.")
            st.caption(f"{type(e).__name__}: {e}")
    else:
        st.code("pip install sentence-transformers\nstreamlit run streamlit_app.py", language="bash")

with t3:
    st.subheader("Feedback & refinement")
    labels = {f'{j["job_id"]} — {j["title"]} @ {j["company"]}': j["job_id"] for j in JOBS}
    selected = st.selectbox("Job", list(labels.keys()))
    action = st.radio("Feedback", ["save","reject","needs review"], horizontal=True)
    note = st.text_input("Reason / correction", placeholder="e.g., experience requirement is too high")
    if st.button("Record feedback", type="primary"):
        st.session_state.feedback.append({"job_id":labels[selected],"action":action,"note":note,"profile":profile["name"]})
        st.success("Feedback recorded in this Streamlit session.")
    if st.session_state.feedback:
        fdf = pd.DataFrame(st.session_state.feedback)
        st.dataframe(fdf, use_container_width=True, hide_index=True)
        st.download_button("Download feedback CSV", fdf.to_csv(index=False), "feedback_log.csv", "text/csv")
    else:
        st.caption("No feedback recorded yet.")

with t4:
    st.subheader("Manual vs. AI-guided experiment")
    st.write("Do not force AI to win. Compare time, correctness, ranking quality, transparency, reproducibility, and human edits.")
    edited = st.data_editor(st.session_state.comparison, use_container_width=True, hide_index=True, num_rows="fixed")
    st.session_state.comparison = edited
    st.download_button("Download comparison CSV", edited.to_csv(index=False), "manual_vs_ai_comparison.csv", "text/csv")
    st.markdown("**Questions:** What became faster? What became harder to verify? What did the human change? Did the AI actually run tests?")

with t5:
    st.subheader("Agentic AI + GitHub workflow")
    st.markdown("**GOAL → PLAN → TOOLS → OBSERVE → REVISE → VERIFY**")
    prompt = '''We are building a teaching application for AI-guided job search.
Before changing code, propose a short plan.

Improve only the baseline_match() and explain_match() parts of the project.

Requirements:
1. Keep every score in [0,1].
2. Preserve separate evidence for skills, experience, interests, and constraints.
3. Add at least three meaningful tests.
4. Do not invent job facts not present in the input data.
5. After editing, execute the tests and show the outputs.
6. Summarize every change.
7. List decisions that still require human approval.
8. Do not rewrite unrelated components.

After implementation, compare the result with the human baseline and document
one improvement, one failure, and one human correction.'''
    st.text_area("Google Antigravity prompt", prompt, height=310)
    st.markdown("#### GitHub experiment workflow")
    st.code('''git checkout -b human-baseline
git add .
git commit -m "HUMAN: add transparent baseline matcher"

git checkout -b ai-guided-matcher
# Apply agent changes, run tests, inspect diff
git add .
git commit -m "AI-GENERATED: propose improved matching logic"

# Human correction and verification
git add .
git commit -m "CO-DESIGNED: verify and correct matching logic"''', language="bash")
    st.write("Issue → Branch → Small commits → PR/review → README → comparison evidence")

st.divider()
st.caption("CS 5588 Challenge 1 • Teaching prototype • Keep human review in the loop.")
