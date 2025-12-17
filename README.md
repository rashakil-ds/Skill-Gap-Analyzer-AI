# LLM-Powered CV Skill Gap Analyzer: End-to-End RAG-Based GenAI System
---
### [Try APP NOW!](https://skill-gap-analyzer-ai.streamlit.app/)

This project is an **end-to-end Generative AI application** that analyzes a candidate’s CV against a target job role, identifies **skill gaps**, and generates **personalized learning insights** using:

- Rule-based skill extraction with evidence
- Role-scoped knowledge (core / optional / excluded skills)
- Retrieval-Augmented Generation (RAG)
- Vector databases (ChromaDB)
- Large Language Models (Groq)
- Streamlit UI

The system is **role-aware**, **grounded**, and **production-ready**, with graceful fallback when LLMs are unavailable.

---

## Key Features

- Upload CV in **PDF or DOCX**
- Regex-based **skill extraction with evidence**
- **Role-scoped filtering** of required skills
- **RAG pipeline** using ChromaDB
- Learning guidance via **Playbooks**
- Career grounding via **Roadmap documents**
- LLM-generated professional insights (optional)
- Fault-tolerant LLM layer (UI never crashes)
- Deployable on **Streamlit Cloud**
  
---

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py


