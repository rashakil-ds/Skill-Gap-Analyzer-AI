import os
from groq import Groq

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def _client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY")
    return Groq(api_key=key)

def generate_gap_report(
    target_role: str,
    matched: list[str],
    missing: list[str],
    cv_skill_evidence: dict,
    role_scope: dict,
    playbook_snippets: str = "",
) -> str:
    client = _client()

    top_skills = sorted(
        cv_skill_evidence.items(),
        key=lambda x: (-x[1].get("score", 0), x[0].lower()),
    )[:10]

    top_skills_text = "\n".join(
        [f"- {k} (score={v.get('score')}, mentions={v.get('mentions')})" for k, v in top_skills]
    )

    core = sorted(list(role_scope.get("core", set())))[:40]
    optional = sorted(list(role_scope.get("optional", set())))[:40]

    prompt = f"""
You are a career coach and hiring-aligned advisor.

Target role: {target_role}

Candidate CV (extracted skills summary):
Top skills:
{top_skills_text}

Matched target skills:
{', '.join(matched) if matched else 'None'}

Missing target skills:
{', '.join(missing) if missing else 'None'}

Role scope:
Core skills: {', '.join(core) if core else 'Not provided'}
Optional skills: {', '.join(optional) if optional else 'Not provided'}

Reference playbooks (grounding, do not invent resources):
{playbook_snippets if playbook_snippets else 'No playbooks provided'}

Write:
1) Gap explanation (6-10 lines, role-specific)
2) Prioritized learning plan (4 weeks, week-wise)
3) 2 portfolio projects aligned with missing skills
Keep it practical and concise.
"""

    r = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=900,
    )
    return r.choices[0].message.content
