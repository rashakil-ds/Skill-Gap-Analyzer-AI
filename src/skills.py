import re
from pathlib import Path
from typing import Dict, List, Set
from langchain.schema import Document 

SKILL_PATTERNS: Dict[str, List[str]] = {
    "Python": [r"\bpython\b"],
    "SQL": [r"\bsql\b", r"\bpostgres\b", r"\bpostgresql\b", r"\bmysql\b", r"\btsql\b", r"\bt-sql\b"],
    "Excel": [r"\bexcel\b"],
    "Power BI": [r"\bpower\s?bi\b"],
    "Tableau": [r"\btableau\b"],
    "Data Visualization": [r"\bdata visualization\b", r"\bvisualization\b", r"\bdashboard\b", r"\bdashboards\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical\b", r"\bhypothesis\b", r"\bregression\b"],
    "Business Analysis": [r"\bbusiness analysis\b", r"\brequirements\b", r"\bstakeholder\b"],

    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Spark": [r"\bspark\b", r"\bpyspark\b"],
    "Airflow": [r"\bairflow\b"],
    "dbt": [r"\bdbt\b"],
    "Kafka": [r"\bkafka\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b", r"\bbitbucket\b"],
    "Docker": [r"\bdocker\b"],
    "Linux": [r"\blinux\b", r"\bbash\b", r"\bshell\b"],

    "Scikit-learn": [r"scikit[- ]learn", r"\bsklearn\b"],
    "PyTorch": [r"\bpytorch\b", r"\btorch\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "Transformers": [r"\btransformers\b", r"\bhugging\s?face\b"],

    "RAG": [r"\brag\b", r"retrieval[- ]augmented"],
    "Embeddings": [r"\bembedding\b", r"\bembeddings\b"],
    "Vector Database": [r"\bvector\s?db\b", r"\bfaiss\b", r"\bchroma\b", r"\bqdrant\b", r"\bpinecone\b", r"\bweaviate\b"],
    "LLM": [r"\bllm\b", r"large language model"],
    "Prompt Engineering": [r"\bprompt engineering\b", r"\bprompting\b"],
    "LangChain": [r"\blangchain\b"],
    "LlamaIndex": [r"\bllamaindex\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Streamlit": [r"\bstreamlit\b"],
    "CI/CD": [r"\bci\/cd\b", r"\bgithub actions\b", r"\bjenkins\b", r"\bgitlab ci\b"],
    "MLOps": [r"\bmlops\b", r"\bmlflow\b", r"\bkubeflow\b"],
    "Cloud": [r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bgoogle cloud\b"],
    "BigQuery": [r"\bbigquery\b", r"\bbig query\b"],
    "Looker Studio": [r"\blooker studio\b", r"\bgoogle data studio\b"],
    "GA4": [r"\bga4\b", r"\bgoogle analytics 4\b"],
}

ROLE_FILE_MAP: Dict[str, str] = {
    "Data/BI Analyst": "data_bi_analyst.md",
    "Data Scientist": "data_scientist.md",
    "Data Engineer": "data_engineer.md",
    "ML Engineer": "ml_engineer.md",
    "AI Engineer": "ai_engineer.md",
    "GenAI/NLP Engineer": "genai_nlp_engineer.md",
    "Software Engineer ML AI": "software_engineer_ml_ai.md",
}

ROLE_ALIASES: Dict[str, List[str]] = {
    "Data/BI Analyst": ["data analyst", "bi analyst", "business intelligence", "reporting analyst"],
    "Data Scientist": ["data scientist", "applied scientist"],
    "Data Engineer": ["data engineer", "analytics engineer", "etl developer"],
    "ML Engineer": ["ml engineer", "machine learning engineer"],
    "AI Engineer": ["ai engineer"],
    "GenAI/NLP Engineer": ["genai engineer", "generative ai engineer", "llm engineer", "rag engineer", "nlp engineer"],
    "Software Engineer ML AI": ["software engineer ml", "software engineer ai", "ml software engineer"],
}

ROLES_DIR = Path("data/roles")


def normalize_role_name(role: str) -> str:
    r = (role or "").strip().lower()
    if not r:
        return ""
    for canonical, aliases in ROLE_ALIASES.items():
        if r == canonical.lower():
            return canonical
        if any(a in r for a in aliases):
            return canonical
    return role.strip()


def _parse_skill_block(text: str, header: str) -> Set[str]:
    if header not in text:
        return set()
    after = text.split(header + ":")[1]
    block = after.split("\n\n")[0].strip()
    items = [s.strip() for s in block.split(",")]
    return {s for s in items if s}


def load_role_scope(role_label: str) -> Dict[str, Set[str]]:
    role_label = normalize_role_name(role_label)
    filename = ROLE_FILE_MAP.get(role_label)
    if not filename:
        return {"core": set(), "optional": set(), "exclude": set(), "role_label": role_label}

    path = ROLES_DIR / filename
    if not path.exists():
        return {"core": set(), "optional": set(), "exclude": set(), "role_label": role_label}

    text = path.read_text(encoding="utf-8", errors="ignore")

    core = _parse_skill_block(text, "CORE_SKILLS")
    optional = _parse_skill_block(text, "OPTIONAL_SKILLS")
    exclude = _parse_skill_block(text, "EXCLUDE_SKILLS")

    return {"core": core, "optional": optional, "exclude": exclude, "role_label": role_label}


def extract_skills_with_evidence(text: str) -> Dict:
    if not text:
        return {"skills": {}}

    t = text.lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    skills: Dict[str, Dict] = {}
    for skill, patterns in SKILL_PATTERNS.items():
        mentions = 0
        for p in patterns:
            hits = re.findall(p, t, flags=re.IGNORECASE)
            if hits:
                mentions += len(hits)

        if mentions <= 0:
            continue

        matched_lines = []
        for ln in lines:
            if any(re.search(p, ln, flags=re.IGNORECASE) for p in patterns):
                matched_lines.append(ln)

        score = 1
        if mentions >= 2:
            score = 2
        if mentions >= 4:
            score = 3

        skills[skill] = {"mentions": mentions, "score": score, "evidence": matched_lines[:5]}

    return {"skills": skills}


def apply_role_exclusions(cv_profile: Dict, role_scope: Dict[str, Set[str]]) -> Dict:
    excluded = {s.strip().lower() for s in role_scope.get("exclude", set()) if s.strip()}
    if not excluded:
        return cv_profile

    filtered = {}
    for skill, info in cv_profile.get("skills", {}).items():
        if skill.strip().lower() in excluded:
            continue
        filtered[skill] = info

    return {"skills": filtered}
