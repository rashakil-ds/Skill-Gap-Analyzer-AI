from typing import List

def build_roadmap(missing_skills: List[str], playbook_docs) -> List[dict]:
    # Collect a few helpful lines from playbooks (lightweight "RAG grounded" resources)
    playbook_snippets = []
    for d in playbook_docs:
        # take first lines as quick resources (avoid dumping whole doc)
        lines = [ln.strip() for ln in d.page_content.splitlines() if ln.strip()]
        playbook_snippets.append(f"{d.metadata.get('source','playbook')}: " + " | ".join(lines[:3]))

    missing = missing_skills[:12]  # keep roadmap manageable
    weeks = []

    # 4-week default plan
    buckets = [
        missing[0:3],
        missing[3:6],
        missing[6:9],
        missing[9:12],
    ]

    titles = [
        "Foundation & Target Gaps",
        "Build Proof via Mini-Projects",
        "End-to-End Project & Deployment",
        "Polish, Interview Prep, Portfolio",
    ]

    for i, focus in enumerate(buckets, start=1):
        if not focus:
            continue
        weeks.append({
            "week": i,
            "title": titles[i-1],
            "focus": focus,
            "tasks": [
                f"Learn core concepts for: {', '.join(focus)}",
                "Create 1 small demo (notebook or mini app) showing these skills",
                "Add 2 strong CV bullets with measurable outcomes",
            ],
            "resources": playbook_snippets[:3] if i <= 2 else playbook_snippets[:2],
        })

    return weeks
