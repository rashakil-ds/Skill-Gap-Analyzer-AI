import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import re

from src.parsing import extract_text_from_upload
from src.skills import extract_skills_with_evidence, normalize_role_name
from src.rag import get_or_build_vectordb, rag_retrieve
from src.roadmap import build_roadmap
from src.utils import now_ts
from src.llm_groq import generate_gap_report

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ROLES_DIR = DATA_DIR / "roles"
CHROMA_DIR = BASE_DIR / "chroma"  # ok; real path used depends on src/rag.py

ROLE_FILE_MAP = {
    "Data/BI Analyst": "data_bi_analyst.md",
    "Data Scientist": "data_scientist.md",
    "Data Engineer": "data_engineer.md",
    "ML Engineer": "ml_engineer.md",
    "AI Engineer": "ai_engineer.md",
    "GenAI/NLP Engineer": "genai_nlp_engineer.md",
    "Software Engineer ML AI": "software_engineer_ml_ai.md",
}

def _parse_list(text: str, key: str) -> set:
    lines = text.splitlines()
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*)\s*$", re.IGNORECASE)

    items = []
    capture = False

    for line in lines:
        m = key_re.match(line)
        if m:
            capture = True
            inline = m.group(1).strip()
            if inline:
                items.append(inline)
            continue

        if capture:
            s = line.strip()
            if not s:
                break
            if s.endswith(":"):  # next section header
                break
            items.append(s)

    joined = ", ".join(items)
    return {x.strip() for x in joined.split(",") if x.strip()}


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

    return {"core": core, "optional": optional, "exclude": exclude, "source": role_path.name}

def filter_by_role_scope(required: set, scope: dict) -> set:
    allowed = (scope["core"] | scope["optional"]) if (scope["core"] or scope["optional"]) else required
    return (required & allowed) - scope["exclude"]

def build_llm_instructions(output_style: str, use_sources_only: bool, custom_instructions: str) -> str:
    base = (
        "You are a career skill-gap advisor. "
        "Use the provided CV skill evidence and retrieved documents to generate role-specific recommendations."
    )

    style_map = {
        "Professional (default)": "Write in a professional, structured tone.",
        "Concise": "Be concise. Use short paragraphs and bullet points only.",
        "Detailed": "Be detailed but avoid repeating information.",
        "Recruiter-friendly": "Write in recruiter-friendly language with clear impact statements.",
    }

    grounding = ""
    if use_sources_only:
        grounding = (
            "Important: Use only the retrieved sources as facts. "
            "If a claim is not supported by the sources, say 'Not found in provided sources'."
        )

    extra = custom_instructions.strip()
    return "\n".join([base, style_map.get(output_style, ""), grounding, extra]).strip()

def resolve_target_role(selected: str, custom: str) -> str:
    if custom.strip():
        return custom.strip()
    if selected != "Select a role":
        return selected
    return ""

st.set_page_config(page_title="CV Skill Gap Analyzer", layout="centered")
st.title("Skill Gap Analyzer")
st.caption("Upload your CV and target role to get an intelligent skill gap assessment and a customized learning roadmap.")

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

output_style = "Professional (default)"
use_sources_only = True
custom_instructions = ""

with st.expander("LLM settings (optional)"):
    output_style = st.selectbox(
        "Output style",
        ["Professional (default)", "Concise", "Detailed", "Recruiter-friendly"],
    )
    use_sources_only = st.checkbox("Force grounded output (use only retrieved sources)", value=True)
    custom_instructions = st.text_area(
        "Additional instructions to the LLM",
        placeholder="e.g., Focus only on core skills for the selected role. Avoid DevOps topics for analysts.",
        height=120,
    )

target_role = resolve_target_role(selected_role, custom_role)
st.caption(f"Analyzing for role: **{target_role if target_role else '—'}**")

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

    instructions = build_llm_instructions(output_style, use_sources_only, custom_instructions)

    with st.spinner("Processing your CV and analyzing skill gaps..."):
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

        roadmaps = rag_retrieve(
            vectordb,
            f"{role_key} roadmap responsibilities skills learning path",
            filters={"type": "roadmap"},
        )

        build_roadmap(missing, playbooks)

        playbook_snippets = "\n\n".join([p.page_content[:700] for p in playbooks[:4]]) if playbooks else ""
        roadmap_snippets = "\n\n".join([r.page_content[:700] for r in roadmaps[:4]]) if roadmaps else ""

        llm_report = ""
        if use_llm:
            try:
                llm_report = generate_gap_report(
                    target_role=target_role,
                    matched=matched,
                    missing=missing,
                    cv_skill_evidence=cv_profile["skills"],
                    role_scope=role_scope,
                    playbook_snippets=playbook_snippets,
                    roadmap_snippets=roadmap_snippets,
                    instructions=instructions,
                )
            except Exception as e:
                llm_report = "LLM insights are temporarily unavailable.\n\nReason: " + str(e)

    st.success("Analysis complete 😊")

    st.subheader("LLM Insights:")
    st.write(llm_report if llm_report else "LLM insights were skipped.")

    #with st.expander("Role scope used"):
        #st.write("Role file:", role_scope["source"])
        #st.write("Core:", sorted(role_scope["core"]))
        #st.write("Optional:", sorted(role_scope["optional"]))
        #st.write("Excluded:", sorted(role_scope["exclude"]))

    st.subheader("Missing Skills")
    st.write(", ".join(missing) if missing else "No missing core skills detected.")

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
