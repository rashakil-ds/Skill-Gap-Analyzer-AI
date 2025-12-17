import re

# Small but strong baseline dictionary (extend anytime)
SKILL_PATTERNS = {
    # core
    "Python": [r"\bpython\b"],
    "SQL": [r"\bsql\b", r"\bpostgres\b", r"\bmysql\b", r"\btsql\b"],
    "Git": [r"\bgit\b", r"github", r"gitlab"],
    "Docker": [r"\bdocker\b"],
    "Linux": [r"\blinux\b", r"\bbash\b", r"\bshell\b"],
    # data
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Power BI": [r"\bpower\s?bi\b"],
    "Tableau": [r"\btableau\b"],
    "Spark": [r"\bspark\b", r"\bpyspark\b"],
    "Airflow": [r"\bairflow\b"],
    "dbt": [r"\bdbt\b"],
    "Kafka": [r"\bkafka\b"],
    # ml / dl
    "Scikit-learn": [r"scikit[- ]learn", r"\bsklearn\b"],
    "PyTorch": [r"\bpytorch\b", r"\btorch\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "Transformers": [r"\btransformers\b", r"hugging\s?face"],
    # genai
    "RAG": [r"\brag\b", r"retrieval[- ]augmented"],
    "Embeddings": [r"\bembedding", r"\bembeddings\b"],
    "Vector Database": [r"\bvector\s?db\b", r"\bfaiss\b", r"\bchroma\b", r"\bqdrant\b", r"\bpinecone\b", r"\bweaviate\b"],
    "LLM": [r"\bllm\b", r"large language model"],
    "LangChain": [r"\blangchain\b"],
    "LlamaIndex": [r"\bllamaindex\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Streamlit": [r"\bstreamlit\b"],
    "CI/CD": [r"\bci\/cd\b", r"\bgithub actions\b", r"\bjenkins\b"],
    "MLOps": [r"\bmlops\b", r"\bmlflow\b", r"\bkubeflow\b"],
    "Cloud": [r"\baws\b", r"\bazure\b", r"\bgcp\b"],
}

ROLE_ALIASES = {
    "Data Analyst": ["data analyst", "bi analyst", "business intelligence"],
    "Data Scientist": ["data scientist", "applied scientist"],
    "Data Engineer": ["data engineer", "analytics engineer"],
    "ML Engineer": ["ml engineer", "machine learning engineer"],
    "GenAI Engineer": ["genai engineer", "generative ai engineer", "llm engineer", "rag engineer"],
}

def normalize_role_name(role: str) -> str:
    r = role.strip().lower()
    for canonical, aliases in ROLE_ALIASES.items():
        if r == canonical.lower():
            return canonical
        if any(a in r for a in aliases):
            return canonical
    return role.strip()

def extract_skills_with_evidence(text: str) -> dict:
    t = text.lower()
    skills = {}
    for skill, patterns in SKILL_PATTERNS.items():
        mentions = 0
        evidence = []
        for p in patterns:
            hits = re.findall(p, t, flags=re.IGNORECASE)
            if hits:
                mentions += len(hits)

        if mentions > 0:
            # Very simple evidence: keep some matching lines
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            matched_lines = []
            for ln in lines:
                if any(re.search(p, ln, flags=re.IGNORECASE) for p in patterns):
                    matched_lines.append(ln)
            evidence = matched_lines[:5]

            # Evidence score: listed + repeated mention boosts
            # (You can later improve with section detection: Projects > Skills > Experience)
            score = 1
            if mentions >= 2:
                score = 2
            if mentions >= 4:
                score = 3

            skills[skill] = {"mentions": mentions, "score": score, "evidence": evidence}

    return {"skills": skills}
