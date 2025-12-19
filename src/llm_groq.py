import os
import json
from typing import Dict, List, Any
from groq import Groq

def _safe_join(items) -> str:
    if not items:
        return ""
    if isinstance(items, (set, tuple)):
        items = list(items)
    return ", ".join([str(x) for x in items if str(x).strip()])

def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[:max_chars]

def generate_gap_report(
    target_role: str,
    matched: List[str],
    missing: List[str],
    cv_skill_evidence: Dict[str, Dict[str, Any]],
    role_scope: Dict[str, Any],
    playbook_snippets: str = "",
    roadmap_snippets: str = "",
    instructions: str = "",
) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to .env (local) or Streamlit Secrets (cloud).")

    #model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile").strip()
    #model = os.getenv("GROQ_MODEL", "llama3-70b-8192").strip()
    #model = os.getenv("GROQ_MODEL", "qwen-qwq-32b").strip() #good
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() #finally supported in cloud

    client = Groq(api_key=api_key)

    core = sorted(list(role_scope.get("core", [])))
    optional = sorted(list(role_scope.get("optional", [])))
    exclude = sorted(list(role_scope.get("exclude", [])))
    role_file = role_scope.get("source")

    evidence_lines = []
    top_skills = sorted(
        cv_skill_evidence.items(),
        key=lambda x: (-int(x[1].get("score", 0)), -int(x[1].get("mentions", 0)), x[0].lower()),
    )[:18]

    for skill, info in top_skills:
        ev = info.get("evidence", [])
        if isinstance(ev, list):
            ev = " | ".join(ev[:2])
        evidence_lines.append(f"- {skill} (score={info.get('score')}, mentions={info.get('mentions')}): {ev}")

    cv_evidence_block = "\n".join(evidence_lines)

    playbook_snippets = _truncate(playbook_snippets, 3000)
    roadmap_snippets = _truncate(roadmap_snippets, 3000)
    instructions = _truncate(instructions, 1200)

    system_msg = (
        "You are an expert career coach and hiring-aligned skill gap analyst. "
        "You must be role-specific and avoid recommending irrelevant skills."
    )

    user_msg = f"""

ROLE
- Target role: {target_role}
- Role file used: {role_file if role_file else "None"}

ROLE SCOPE (authoritative)
- Core skills: {_safe_join(core)}
- Optional skills: {_safe_join(optional)}
- Excluded skills (do not recommend): {_safe_join(exclude)}

GAP SIGNALS
- Matched skills: {_safe_join(matched)}
- Missing skills: {_safe_join(missing)}

CV EVIDENCE (top extracted)
{cv_evidence_block}

RETRIEVED PLAYBOOK CONTEXT
{playbook_snippets}

RETRIEVED ROADMAP CONTEXT
{roadmap_snippets}

ADDITIONAL INSTRUCTIONS
{instructions}

TASK
1) Give a short role-specific gap explanation (do not mention excluded skills as gaps).
2) Provide a prioritized 4-week learning plan aligned to the role scope.
3) Suggest 2 portfolio projects to prove the missing core skills (role-relevant).
4) Keep output professional and structured with headings and bullet points.
""".strip()

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=900,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    return resp.choices[0].message.content.strip()
