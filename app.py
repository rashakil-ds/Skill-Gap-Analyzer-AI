import os
import shutil
from pathlib import Path

import streamlit as st
import pandas as pd

from src.parsing import extract_text_from_upload
from src.skills import extract_skills_with_evidence, normalize_role_name
from src.rag import get_or_build_vectordb, rag_retrieve
from src.roadmap import build_roadmap
from src.utils import now_ts

from dotenv import load_dotenv
from src.llm_groq import generate_gap_report

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ROLES_DIR = DATA_DIR / "roles"
CHROMA_DIR = BASE_DIR / ".chroma"

ROLE_FILE_MAP = {
    "Data/BI Analyst": "data_analyst.md",
    "Data Scientist": "data_scientist.md",
    "Data Engineer": "data_engineer.md",
    "ML Engineer": "ml_engineer.md",
    "AI Engineer": "ai_engineer.md",
    "GenAI/NLP Engineer": "genai_nlp_engineer.md",
    "Software Engineer ML AI": "software_engineer_ml_ai.md",
}

def _parse_list(text: str, key: str) -> set:
    for line in text.splitlines():
        if line.strip().lower().startswith(key.lower()):
            return {x.strip() for x in line.split(":", 1)[1].split(",") if x.strip()}
    return set()

def load_role_scope(role_name: str) -> dict:
    file_name = ROLE_FILE_MAP.get(role_name)
    if not file_name:
        return {"core": set(), "optional": set(), "exclude": set(), "source": None}

    role_path = ROLES_DIR / file_name
    if not role_path.exists():
        return {"core": set(), "optional": set(), "exclude": set(), "source": None}

    text = role_path.read_text(encoding="utf-8", errors="ignore")

    core = _parse_list(text, "CORE_SKILLS")
    optional = _parse_list(text, "OPTIONAL_SKILLS")
    exclude = _parse_list(text, "EXCLUDE_SKILLS")

    return {
        "core": core,
        "optional": optional,
        "exclude": exclude,
        "source": role_path.name,
    }

def filter_by_role_scope(required: set, scope: dict) -> set:
    allowed = scope["core"] | scope["optional"] if scope["core"] or scope["optional"] else required
    return (required & allowed) - scope["exclude"]

st.set_page_config(page_title="CV Skill Gap Analyzer", layout="centered")
st.title("Skill Gap Analyzer")
st.caption(
    "Upload your CV and target role to get an intelligent skill gap assessment and a customized learning roadmap."
)

COMMON_ROLES = [
    "Select a role",
    "Data/BI Analyst",
    "Data Scientist",
    "Data Engineer",
    "ML Engineer",
    "AI Engineer",
    "GenAI/NLP Engineer",
    "Software Engineer ML AI",
]

cv_file = st.file_uploader("Upload your CV/Resume *", type=["pdf", "docx"])
selected_role = st.selectbox("Choose a common role (optional)", COMMON_ROLES)
custom_role = st.text_input("Or enter a custom job position", placeholder="e.g., Analytics Engineer")

use_jd = st.checkbox("I have a job description (recommended)")
job_description = ""
if use_jd:
    job_description = st.text_area("Job description from job post", height=220)

use_llm = st.checkbox("Generate LLM explanation", value=True)

def resolve_target_role(selected, custom):
    if custom.strip():
        return custom.strip()
    if selected != "Select a role":
        return selected
    return ""

target_role = resolve_target_role(selected_role, custom_role)
st.caption(f"Analyzing for role: **{target_role if target_role else '—'}**")

force_rebuild = st.checkbox("Force rebuild knowledge index", value=False)

st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #0f5132;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #198754;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

analyze_clicked = st.button("Analyze Skill Gap", type="primary", use_container_width=True)

if analyze_clicked:
    if not cv_file:
        st.error("Please upload your CV.")
        st.stop()

    if not target_role:
        st.error("Please select or enter a target role.")
        st.stop()

    if use_jd and len(job_description.strip()) < 60:
        st.error("Please paste the full job description.")
        st.stop()

    role_key = normalize_role_name(target_role)
    role_scope = load_role_scope(role_key)

    with st.spinner("Processing your CV and analyzing skill gaps..."):
        if force_rebuild and CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

        cv_text = extract_text_from_upload(cv_file)
        vectordb = get_or_build_vectordb()
        cv_profile = extract_skills_with_evidence(cv_text)

        cv_skills = set(cv_profile["skills"].keys())

        if role_scope["core"] or role_scope["optional"]:
            required_skills = role_scope["core"] | role_scope["optional"]
        else:
            docs = rag_retrieve(
                vectordb,
                f"{role_key} required skills tools stack",
                filters={"type": "role"},
            )
            required_skills = {
                s.strip()
                for d in docs
                for s in d.metadata.get("skills", "").split("|")
                if s.strip()
            }

        required_skills = filter_by_role_scope(required_skills, role_scope)

        matched = sorted(required_skills & cv_skills)
        missing = sorted(required_skills - cv_skills)

        playbooks = []
        if missing:
            playbooks = rag_retrieve(
                vectordb,
                "Learning guidance for: " + ", ".join(missing),
                filters={"type": "playbook"},
            )

        roadmap = build_roadmap(missing, playbooks)

        llm_report = ""
        if use_llm:
            try:
                playbook_snippets = "\n\n".join([p.page_content[:700] for p in playbooks[:4]])
                llm_report = generate_gap_report(
                    target_role=target_role,
                    matched=matched,
                    missing=missing,
                    cv_skill_evidence=cv_profile["skills"],
                    role_scope=role_scope,
                    playbook_snippets=playbook_snippets,
                )
            except Exception as e:
                llm_report = f"LLM report unavailable: {e}"


    st.success("Analysis complete")

    st.subheader("LLM Insights")
    st.write(llm_report)


    with st.expander("Role scope used"):
        st.write("Role file:", role_scope["source"])
        st.write("Core:", sorted(role_scope["core"]))
        st.write("Optional:", sorted(role_scope["optional"]))
        st.write("Excluded:", sorted(role_scope["exclude"]))

    col1, col2, col3 = st.columns(3)
    col1.metric("Skills detected", len(cv_skills))
    col2.metric("Matched skills", len(matched))
    col3.metric("Missing skills", len(missing))

    st.subheader("Missing Skills")
    st.write(", ".join(missing) if missing else "No missing core skills detected.")

    st.subheader("Skill Evidence")
    rows = [
        {
            "Skill": skill,
            "Evidence Score": info["score"],
            "Mentions": info["mentions"],
            "Evidence": "; ".join(info["evidence"][:3]),
        }
        for skill, info in sorted(
            cv_profile["skills"].items(),
            key=lambda x: (-x[1]["score"], x[0].lower()),
        )
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Personalized Roadmap")
    for week in roadmap:
        st.markdown(f"### Week {week['week']}: {week['title']}")
        st.write("Focus:", ", ".join(week["focus"]))
        for task in week["tasks"]:
            st.write("-", task)

    st.caption(f"Generated at {now_ts()}")

st.divider()
st.markdown(
    """
    <div style="font-size:0.75rem;color:#6c757d">
        Developed by <a href="https://www.linkedin.com/in/kmrashedulalam/" target="_blank"
        style="color:#0f5132;font-weight:600;text-decoration:none">
        Rashedul Alam</a>
    </div>
    """,
    unsafe_allow_html=True,
)
