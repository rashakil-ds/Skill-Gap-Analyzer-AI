import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.parsing import extract_text_from_upload
from src.skills import extract_skills_with_evidence, normalize_role_name
from src.rag import get_or_build_vectordb, rag_retrieve
from src.roadmap import build_roadmap
from src.utils import now_ts

load_dotenv()

st.set_page_config(page_title="CV Skill Gap Analyzer", layout="centered")
st.title("Skill Gap Analyzer AI")
st.caption(
    "Upload your CV and target role to get an intelligent skill gap assessment and a customized AI-focused learning roadmap powered by Generative AI."
)

COMMON_ROLES = [
    "Select a role",
    "Data/BI Analyst",
    "Data Scientist",
    "Data Engineer",
    "ML Engineer",
    "AI Engineer",
    "GenAI/NLP Engineer",
]

st.subheader("Inputs")

cv_file = st.file_uploader("Upload your CV *", type=["pdf", "docx"])

selected_role = st.selectbox("Choose a common role (optional)", COMMON_ROLES)
custom_role = st.text_input("Or, Enter a custom job position", placeholder="e.g., Software Engineer (Machine Learning) / Analytics Engineer")

use_jd = st.checkbox("I have a job description (recommended)")
job_description = ""
if use_jd:
    job_description = st.text_area("Job description from job post:", height=220, placeholder="Paste full job description...")

def resolve_target_role(selected, custom):
    if custom.strip():
        return custom.strip()
    if selected != "Select a role":
        return selected
    return ""

target_role = resolve_target_role(selected_role, custom_role)
st.caption(f"Analyzing for role: **{target_role if target_role else '—'}**")

#button
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #0f5132;   /* dark green */
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #198754;   /* lighter green on hover */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

analyze_clicked = st.button("Analyze Skill Gap", type="primary", use_container_width=True)

if analyze_clicked:
    if not cv_file:
        st.error("Please upload your CV.")
        st.stop()
    if not target_role.strip():
        st.error("Please select or enter the target job position.")
        st.stop()
    if use_jd and len(job_description.strip()) < 60:
        st.error("Job description is too short. Please paste the full job post.")
        st.stop()

    with st.spinner("Processing your CV and analyzing skill gaps..."):
        #Parse CV
        cv_text = extract_text_from_upload(cv_file)

        #Build/Load vector DB (roles + playbooks)
        vectordb = get_or_build_vectordb()

        #Skill extraction from CV (evidence scoring)
        cv_profile = extract_skills_with_evidence(cv_text)

        #Retrieve requirements via RAG
        role_key = normalize_role_name(target_role)
        rag_query = f"Role: {target_role}\nFind required skills, tools, and responsibilities.\n"
        if use_jd:
            rag_query += f"\nJob description:\n{job_description}\n"
        docs = rag_retrieve(vectordb, rag_query, filters={"type": "role"})  # role docs
        playbooks = rag_retrieve(vectordb, "Provide learning playbooks for missing skills.", filters={"type": "playbook"})

        #Extract required skills from retrieved role docs
        required_skills = set()
        for d in docs:
            required_skills |= set(d.metadata.get("skills", "").split("|")) if d.metadata.get("skills") else set()

        required_skills = {s.strip() for s in required_skills if s.strip()}

        #If no embedded metadata found (custom role), fallback:
        if not required_skills:
            #retrieve more generally
            docs = rag_retrieve(vectordb, f"{target_role} required skills tools stack", filters={"type": "role"})
            for d in docs:
                required_skills |= set(d.metadata.get("skills", "").split("|")) if d.metadata.get("skills") else set()
            required_skills = {s.strip() for s in required_skills if s.strip()}

        #Gap analysis
        cv_skills = set(cv_profile["skills"].keys())
        missing = sorted(list(required_skills - cv_skills))
        matched = sorted(list(required_skills & cv_skills))

        #Roadmap generation (rule-based + playbook snippets)
        roadmap = build_roadmap(missing, playbooks)

    st.success("Analysis complete!")

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Skills detected in CV", len(cv_skills))
    col2.metric("Matched target skills", len(matched))
    col3.metric("Missing target skills", len(missing))

    st.subheader("Missing Skills (Gap)")
    if missing:
        st.write(", ".join(missing))
    else:
        st.write("No obvious missing skills found from the target role docs. Consider adding a job description for higher accuracy.")

    st.subheader("Skill Evidence Table (from CV)")
    rows = []
    for skill, info in sorted(cv_profile["skills"].items(), key=lambda x: (-x[1]["score"], x[0].lower())):
        rows.append({
            "skill": skill,
            "evidence_score": info["score"],
            "mentions": info["mentions"],
            "evidence": "; ".join(info["evidence"][:3])
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Personalized Roadmap")
    for week in roadmap:
        st.markdown(f"### Week {week['week']}: {week['title']}")
        st.write("**Focus:**", ", ".join(week["focus"]))
        st.write("**Tasks:**")
        for t in week["tasks"]:
            st.write(f"- {t}")
        if week.get("resources"):
            st.write("**Resources (from playbooks):**")
            for r in week["resources"]:
                st.write(f"- {r}")

    st.caption(f"Generated at {now_ts()}")

#about me
st.divider()
st.markdown(
    """
    <div style="text-align: left; font-size: 0.50rem; color: #6c757d;">
        Developed by 
        <a href="https://www.linkedin.com/in/kmrashedulalam/" target="_blank" style="color: #0f5132; text-decoration: none; font-weight: 600;">
            Rashedul Alam
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
