#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import os
import re
from typing import List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:  
    OpenAI = None  
    OPENAI_AVAILABLE = False

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:  
    OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:  
    genai = None  
    GEMINI_AVAILABLE = False

from ranking_engine import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    get_candidate_skills,
    get_job_skills,
    get_driver,
    init_driver,
    list_candidates,
    list_jobs,
    rank_candidates_for_job,
    rank_jobs_for_candidate,
    search_candidates_by_name,
    search_jobs_by_title,
    suggest_skills_for_candidate,
)


load_dotenv()


def icon_img_tag(path: str, alt: str, size: int = 40) -> str:
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{encoded}" alt="{alt}" width="{size}" height="{size}">'
    except FileNotFoundError:
        return ""


st.set_page_config(page_title="HireMatch AI", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(140deg, #fafafa 0%, #e4e5f1 55%, #d2d3db 100%);
        color: #484b6a;
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    body, p, label, span, div, section {
        color: #484b6a !important;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #484b6a;
        animation: slideIn 1.2s ease forwards;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #9394a5;
        margin-bottom: 1.5rem;
        animation: fadeIn 1.4s ease forwards;
    }
    .glass-panel {
        background: rgba(250,250,250,0.96);
        border-radius: 18px;
        padding: 1.2rem 1.6rem;
        box-shadow: 0px 20px 45px rgba(72,75,106,0.12);
        border: 1px solid rgba(147,148,165,0.25);
        margin-bottom: 1.2rem;
    }
    .primary-accent {
        color: #484b6a;
        font-weight: 600;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.5rem;
    }
    .section-header img {
        width: 34px;
        height: 34px;
    }
    .section-header h3 {
        margin: 0;
        font-size: 1.25rem;
        color: #484b6a;
    }
    button[kind="primary"], button[kind="secondary"] {
        background-color: #484b6a !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(72,75,106,0.25);
    }
    button[kind="primary"]:hover, button[kind="secondary"]:hover {
        background-color: #9394a5 !important;
        color: #ffffff !important;
    }
    .stSelectbox, .stTextInput, .stNumberInput, .stSlider, .stFileUploader {
        color: #484b6a !important;
    }

    /* ==== INPUT STYLING (SCOPED, so it doesn't break password widget) ==== */
    .stTextInput div[data-baseweb="input"],
    .stNumberInput div[data-baseweb="input"],
    .stSelectbox div[data-baseweb="select"] {
        width: 100% !important;
        background-color: rgba(255,255,255,0.98) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(72,75,106,0.35) !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
        min-height: 48px;
    }

    .stTextInput div[data-baseweb="input"] > div,
    .stNumberInput div[data-baseweb="input"] > div,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stTextInput div[data-baseweb="input"],
    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"],
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        min-height: 42px !important;
    }

    div[data-baseweb="select"] span, div[data-baseweb="select"] label {
        color: #484b6a !important;
    }

    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background-color: rgba(255,255,255,0.98) !important;
        color: #484b6a !important;
        border-radius: 14px !important;
        border: 1px solid rgba(72,75,106,0.35) !important;
        min-height: 48px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] .stTextInput input {
        min-height: 42px !important;
    }
    .stTextInput input::placeholder, .stNumberInput input::placeholder {
        color: #9394a5 !important;
    }

    /* ==== PASSWORD FIELD: keep layout, just restyle ==== */
    [data-testid="stPassword"] > div {
        background-color: rgba(255,255,255,0.98) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(72,75,106,0.35) !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] [data-testid="stPassword"] > div {
        min-height: 42px !important;
    }
    [data-testid="stPassword"] input {
        background-color: transparent !important;
        color: #484b6a !important;
        border: none !important;
    }
    [data-testid="stPassword"] button,
    button[aria-label="Show password input"],
    button[aria-label="Hide password input"] {
        background-color: transparent !important;
        border-left: 1px solid rgba(72,75,106,0.25) !important;
        height: 100% !important;
        width: 40px !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stPassword"] button:hover {
        background-color: rgba(210,211,219,0.3) !important;
    }
    [data-testid="stPassword"] button svg,
    button[aria-label="Show password input"] svg,
    button[aria-label="Hide password input"] svg {
        display: block !important;
        width: 18px !important;
        height: 18px !important;
        color: #484b6a !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Extra spacing between stacked inputs in sidebar */
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] [data-testid="stPassword"] {
        margin-bottom: 0.75rem !important;
    }

    @keyframes slideIn {
        from {opacity: 0; transform: translateY(-10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

hero_icon_html = icon_img_tag("job-search.png", "Job search icon", 40)

st.markdown(
    f"""
    <div class="glass-panel" style="margin-top: -30px;">
        <div style="display:flex; align-items:center; gap:0.75rem;">
            {hero_icon_html}
            <div class="hero-title">HireMatch AI : Powering two-way matching between candidates and employers</div>
        </div>
        <div class="hero-subtitle">
            Upload a resume, explore graph-driven insights, and discover the top matches for both sides of the talent marketplace.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session defaults
# -----------------------------
if "neo4j_connected" not in st.session_state:
    st.session_state["neo4j_connected"] = False
if "neo4j_error" not in st.session_state:
    st.session_state["neo4j_error"] = ""
if "candidate_search_input" not in st.session_state:
    st.session_state["candidate_search_input"] = ""
if "candidate_search_origin" not in st.session_state:
    st.session_state["candidate_search_origin"] = None
if "selected_candidate_id" not in st.session_state:
    st.session_state["selected_candidate_id"] = None
if "selected_job_id" not in st.session_state:
    st.session_state["selected_job_id"] = None
if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = os.getenv("OLLAMA_MODEL", "llama3")
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = os.getenv("GEMINI_MODEL", "gemini-pro")
if "llm_backend" not in st.session_state:
    if OPENAI_AVAILABLE:
        st.session_state["llm_backend"] = "OpenAI"
    elif OLLAMA_AVAILABLE:
        st.session_state["llm_backend"] = "Ollama"
    elif GEMINI_AVAILABLE:
        st.session_state["llm_backend"] = "Gemini"
    else:
        st.session_state["llm_backend"] = "Disabled"


# -----------------------------
# Sidebar: connection + weights
# -----------------------------
st.sidebar.header("Neo4j Connection")

if "neo4j_uri_input" not in st.session_state:
    st.session_state["neo4j_uri_input"] = NEO4J_URI
if "neo4j_user_input" not in st.session_state:
    st.session_state["neo4j_user_input"] = NEO4J_USER
if "neo4j_password_input" not in st.session_state:
    st.session_state["neo4j_password_input"] = NEO4J_PASSWORD

uri = st.sidebar.text_input("URI", key="neo4j_uri_input")
user = st.sidebar.text_input("Username", key="neo4j_user_input")
password = st.sidebar.text_input("Password", type="password", key="neo4j_password_input")
connect_clicked = st.sidebar.button("Connect", use_container_width=True)

if connect_clicked:
    try:
        drv = init_driver(uri, user, password)
        with drv.session() as session:
            session.run("RETURN 1 AS ok").single()
        st.session_state["neo4j_connected"] = True
        st.session_state["neo4j_error"] = ""
    except Exception as exc:
        st.session_state["neo4j_connected"] = False
        st.session_state["neo4j_error"] = str(exc)

if st.session_state["neo4j_connected"]:
    st.sidebar.success("Connected to Neo4j")
else:
    if st.session_state["neo4j_error"]:
        st.sidebar.error(st.session_state["neo4j_error"])
    else:
        st.sidebar.info("Enter your Neo4j credentials and click Connect.")

st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid rgba(147,148,165,0.3);
        color: #484b6a;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p {
        color: #484b6a !important;
    }
    [data-testid="stSidebar"] input {
        color: #484b6a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.header("Scoring Weights")
alpha = st.sidebar.slider("Skill coverage (α)", 0.0, 1.0, 1.0, 0.05)
beta = st.sidebar.slider("Rasch (β)", 0.0, 1.0, 0.2, 0.05)
gamma = st.sidebar.slider("TransR (γ)", 0.0, 1.0, 0.1, 0.05)
top_k = st.sidebar.slider("Top K", 1, 50, 10)

st.sidebar.markdown("---")
st.sidebar.header("AI Assistant")
llm_choices = []
if OPENAI_AVAILABLE:
    llm_choices.append("OpenAI")
if OLLAMA_AVAILABLE:
    llm_choices.append("Ollama")
if GEMINI_AVAILABLE:
    llm_choices.append("Gemini")
llm_choices.append("Disabled")

default_backend = st.session_state.get("llm_backend", "Disabled")
if default_backend not in llm_choices:
    default_backend = "Disabled"
selected_backend = st.sidebar.selectbox(
    "LLM backend",
    llm_choices,
    index=llm_choices.index(default_backend),
)
st.session_state["llm_backend"] = selected_backend
if selected_backend != "OpenAI":
    st.session_state.pop("_openai_client", None)

if selected_backend == "OpenAI":
    st.sidebar.caption("Requires OPENAI_API_KEY in environment/.env.")
elif selected_backend == "Ollama":
    if not OLLAMA_AVAILABLE:
        st.sidebar.error("Install the `ollama` Python package to enable this backend.")
    else:
        st.session_state["ollama_model"] = st.sidebar.text_input(
            "Ollama model",
            value=st.session_state.get("ollama_model", "llama3"),
            help="Model name exposed by your local Ollama server.",
        )
        st.sidebar.caption("Ensure `ollama serve` is running locally.")
elif selected_backend == "Gemini":
    if not GEMINI_AVAILABLE:
        st.sidebar.error("Install `google-generativeai` to enable this backend.")
    else:
        st.sidebar.caption("Requires GEMINI_API_KEY in environment/.env.")
        st.session_state["gemini_model"] = st.sidebar.text_input(
            "Gemini model",
            value=st.session_state.get("gemini_model", "gemini-pro"),
            help="Public Gemini model name, e.g., gemini-pro or gemini-1.5-flash-latest.",
        )
else:
    st.sidebar.info("AI summaries disabled.")


def extract_text_from_resume(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = (uploaded_file.name or "").lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    data = uploaded_file.read()
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def guess_candidate_name(resume_text: str) -> Optional[str]:
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    for line in lines[:10]:
        cleaned = re.sub(r"[^A-Za-z\s\-\.'`]", " ", line).strip()
        if not cleaned:
            continue
        tokens = cleaned.split()
        if len(tokens) < 2 or len(tokens) > 4:
            continue
        if all(token[0].isupper() for token in tokens if token):
            return cleaned
    return None


def safe_excerpt(text: str, limit: int = 1200) -> str:
    return text[:limit].strip()


def get_openai_client() -> Optional["OpenAI"]:
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def llm_backend_ready() -> bool:
    backend = st.session_state.get("llm_backend")
    if backend == "OpenAI":
        client = st.session_state.get("_openai_client")
        if client is None:
            client = get_openai_client()
            st.session_state["_openai_client"] = client
        return client is not None
    if backend == "Ollama":
        return OLLAMA_AVAILABLE
    if backend == "Gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        return GEMINI_AVAILABLE and bool(api_key)
    return False


def call_llm(messages, temperature: float = 0.2) -> Optional[str]:
    backend = st.session_state.get("llm_backend")
    if backend == "OpenAI":
        client = st.session_state.get("_openai_client")
        if client is None:
            client = get_openai_client()
            st.session_state["_openai_client"] = client
        if client is None:
            return None
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()
    if backend == "Ollama":
        if not OLLAMA_AVAILABLE:
            return None
        model_name = st.session_state.get("ollama_model") or os.getenv("OLLAMA_MODEL", "llama3")
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"].strip()
    if backend == "Gemini":
        if not GEMINI_AVAILABLE:
            return None
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model_name = st.session_state.get("gemini_model") or os.getenv("GEMINI_MODEL", "gemini-pro")
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        model = genai.GenerativeModel(model_name)
        formatted = [
            {
                "role": msg.get("role", "user"),
                "parts": [msg.get("content", "")],
            }
            for msg in messages
        ]
        response = model.generate_content(
            formatted,
            generation_config={"temperature": temperature},
        )
        return response.text.strip()
    return None


def llm_job_match_summary(
    resume_text: str,
    cand_name: str,
    cand_skills: List[str],
    job_row: dict,
    job_skills: List[str],
    missing_skills: List[dict],
) -> str:
    missing_list = ", ".join(f"{s['skill']} ({s['jobCount']} jobs)" for s in missing_skills) if missing_skills else "None"
    prompt = f"""
Candidate: {cand_name}
Candidate skills: {', '.join(cand_skills[:50]) or 'Unknown'}
Job: {job_row.get('jobTitle', 'Unknown')} at {job_row.get('company', '')}
Score metrics: coverage={job_row.get('coverage')}, rasch={job_row.get('rasch_match') or job_row.get('raschMatch')}, transR={job_row.get('transr_sim')}
Job skills: {', '.join(job_skills[:50]) or 'Unknown'}
Missing/upskill suggestions: {missing_list}

Resume excerpt:
{safe_excerpt(resume_text)}

Write two short paragraphs explaining why this job is or is not a strong fit, referencing the metrics and skills above. Call out concrete strengths and the most important gaps.
"""
    response = call_llm(
        [
            {"role": "system", "content": "You are an assistant that explains candidate-job matches based on structured data."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    if response is None:
        raise RuntimeError("LLM backend not configured.")
    return response


def llm_skill_coaching(
    resume_text: str,
    cand_name: str,
    target_job: str,
    missing_skills: List[dict],
) -> str:
    prompt = f"""
Candidate: {cand_name}
Target job: {target_job}
Missing skills to prioritize: {', '.join(f"{s['skill']} (needed in {s['jobCount']} matches)" for s in missing_skills)}

Resume excerpt:
{safe_excerpt(resume_text)}

Provide a concise bulleted plan (<=5 bullets) suggesting how the candidate could build or highlight those skills in their resume or learning plan. Reference concrete actions or phrasing tweaks.
"""
    response = call_llm(
        [
            {"role": "system", "content": "You coach candidates on improving resumes and filling skill gaps."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    if response is None:
        raise RuntimeError("LLM backend not configured.")
    return response


def llm_candidate_for_job_summary(
    job_meta: dict,
    job_skills: List[str],
    candidate_row: dict,
    candidate_skills: List[str],
) -> str:
    prompt = f"""
Job: {job_meta.get('job_title')} at {job_meta.get('company')}
Job skills: {', '.join(job_skills[:50])}

Candidate: {candidate_row.get('candidateName')} (ID {candidate_row.get('cand_id')})
Candidate skills: {', '.join(candidate_skills[:50])}
Score metrics: coverage={candidate_row.get('coverage')}, rasch={candidate_row.get('rasch_match')}, transR={candidate_row.get('transr_sim')}
Shared skill count: {candidate_row.get('sharedCount')}

Explain why this candidate ranks where they do for the job above. Mention strengths, gaps, and hiring manager considerations in 2 paragraphs.
"""
    response = call_llm(
        [
            {"role": "system", "content": "You assist recruiters by summarizing why a candidate suits a job based on graph metrics."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    if response is None:
        raise RuntimeError("LLM backend not configured.")
    return response


if not st.session_state["neo4j_connected"]:
    st.info("Enter valid Neo4j credentials in the sidebar to start.")
    st.stop()

get_driver()  # ensureing we keep the active connection alive


tab_candidate, tab_employer = st.tabs(["Candidate View", "Employer View"])


# -----------------------------
# Candidate view
# -----------------------------
with tab_candidate:
    candidate_icon_html = icon_img_tag("cv.png", "Candidate icon", 34)
    st.markdown(
        f"""
        <div class="section-header">
            {candidate_icon_html}
            <h3>Candidate Journey</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("#### 1. Upload a resume")
    uploaded_resume = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    resume_text = st.session_state.get("resume_text", "")
    resume_token = st.session_state.get("resume_token")

    if uploaded_resume is not None:
        resume_text = extract_text_from_resume(uploaded_resume)
        st.session_state["resume_text"] = resume_text
        resume_token = f"{uploaded_resume.name}-{getattr(uploaded_resume, 'size', len(resume_text))}"
        st.session_state["resume_token"] = resume_token

        if not resume_text.strip():
            st.error("No readable text was extracted from that file.")
        else:
            st.success(f"Loaded resume: {uploaded_resume.name}")
            with st.expander("Preview extracted text"):
                preview = resume_text[:4000]
                st.text(preview + ("\n...\n" if len(resume_text) > len(preview) else ""))

            detected_name = guess_candidate_name(resume_text)
            st.session_state["detected_candidate_name"] = detected_name
            if detected_name:
                st.info(f"Detected the name **{detected_name}** in this resume. Searching the knowledge graph for matches.")
                st.session_state["candidate_search_input"] = detected_name
                st.session_state["candidate_search_origin"] = resume_token
            else:
                st.warning("Could not detect a candidate name automatically. Use the search box below.")

    if not resume_text:
        st.info("Upload a resume to match it against the knowledge graph.")
    else:
        st.markdown("### 2. Match the resume to a candidate node")
        cand_query = st.text_input(
            "Search by name",
            key="candidate_search_input",
            help="Edit if the automatic guess is incorrect.",
        )

        candidate_rows: List[dict] = []
        if cand_query:
            candidate_rows = search_candidates_by_name(cand_query, limit=25)
        if not candidate_rows:
            candidate_rows = list_candidates(limit=25)

        selected_candidate = None
        if candidate_rows:
            options = [f"{row['name']} ({row['cand_id']})" for row in candidate_rows]
            cand_map = {label: row for label, row in zip(options, candidate_rows)}
            default_idx = 0
            stored_id = st.session_state.get("selected_candidate_id")
            if stored_id:
                for idx, row in enumerate(candidate_rows):
                    if row["cand_id"] == stored_id:
                        default_idx = idx
                        break

            selected_label = st.selectbox(
                "Select the matching candidate",
                options,
                index=min(default_idx, len(options) - 1),
            )
            selected_candidate = cand_map[selected_label]
            st.session_state["selected_candidate_id"] = selected_candidate["cand_id"]
            st.session_state["selected_candidate_name"] = selected_candidate["name"]
            prev_meta = st.session_state.get("candidate_job_results_meta", {})
            if prev_meta.get("cand_id") != selected_candidate["cand_id"]:
                st.session_state["candidate_job_results"] = []
                st.session_state["candidate_job_results_meta"] = {}
            st.success(f"Resume linked to candidate **{selected_candidate['name']}** in the graph.")
        else:
            st.error("No candidates were found. Adjust the search query.")

        if selected_candidate:
            cand_skills = get_candidate_skills(selected_candidate["cand_id"])
            st.caption(f"{selected_candidate['name']} has {len(cand_skills)} recorded skills.")
            st.session_state["selected_candidate_skills"] = cand_skills
            if cand_skills:
                cols = st.columns(2)
                cols[0].metric("Skill count", len(cand_skills))
                cols[1].metric("Candidate ID", selected_candidate["cand_id"])
                with st.expander("Candidate skills"):
                    st.write(", ".join(sorted(cand_skills)))

            st.markdown("### 3. Recommend jobs")
            find_jobs = st.button("Find top jobs", key="find_jobs_btn")
            if find_jobs:
                with st.spinner("Ranking jobs..."):
                    job_rows = rank_jobs_for_candidate(
                        selected_candidate["cand_id"],
                        k=top_k,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                st.session_state["candidate_job_results"] = job_rows
                st.session_state["candidate_job_results_meta"] = {
                    "cand_id": selected_candidate["cand_id"],
                    "cand_name": selected_candidate["name"],
                }

            job_results = st.session_state.get("candidate_job_results", [])
            meta = st.session_state.get("candidate_job_results_meta", {})
            if job_results:
                st.success(f"Top {min(len(job_results), top_k)} jobs for {meta.get('cand_name', 'candidate')}.")
                job_df = pd.DataFrame(job_results)
                st.dataframe(
                    job_df,
                    use_container_width=True,
                    hide_index=True,
                )

                suggested = suggest_skills_for_candidate(
                    meta.get("cand_id", selected_candidate["cand_id"]),
                    [row.get("job_id") for row in job_results],
                    limit=15,
                )
                if suggested:
                    st.markdown("#### Skills to add for better coverage")
                    sug_df = pd.DataFrame(suggested)
                    sug_df = sug_df.rename(columns={"skill": "Skill"})
                    if "jobCount" in sug_df.columns:
                        sug_df = sug_df.drop(columns=["jobCount"])
                    st.dataframe(sug_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No missing skills detected for the recommended jobs.")

                if not llm_backend_ready():
                    st.info("Select an AI backend in the sidebar to enable explanations.", icon="🤖")
                else:
                    job_labels = [f"{row['jobTitle']} ({row['job_id']})" for row in job_results]
                    job_map = {label: row for label, row in zip(job_labels, job_results)}
                    explain_job_label = st.selectbox(
                        "Select a job to explain",
                        job_labels,
                        key="job_explain_select",
                    )
                    target_job = job_map[explain_job_label]

                    if st.button("Generate AI match summary", key="btn_job_ai_summary"):
                        try:
                            job_skill_list = get_job_skills(target_job["job_id"])
                            summary = llm_job_match_summary(
                                st.session_state.get("resume_text", resume_text),
                                selected_candidate["name"],
                                st.session_state.get("selected_candidate_skills", cand_skills),
                                target_job,
                                job_skill_list,
                                suggested if suggested else [],
                            )
                            st.markdown(summary)
                        except Exception as exc:  # pragma: no cover - API failure path
                            st.error(f"LLM summary failed: {exc}")

                    if suggested:
                        if st.button("AI skill coaching plan", key="btn_skill_coach"):
                            try:
                                plan = llm_skill_coaching(
                                    st.session_state.get("resume_text", resume_text),
                                    selected_candidate["name"],
                                    f"{target_job.get('jobTitle')} @ {target_job.get('company')}",
                                    suggested,
                                )
                                st.markdown(plan)
                            except Exception as exc:
                                st.error(f"LLM coaching failed: {exc}")
            elif find_jobs:
                st.warning("No jobs were returned for this candidate. Try different weights or another resume.")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Recruiter view
# -----------------------------
with tab_employer:
    employer_icon_html = icon_img_tag("coding.png", "Employer icon", 34)
    st.markdown(
        f"""
        <div class="section-header">
            {employer_icon_html}
            <h3>Employer Insight</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("#### 1. Search a job")
    job_query = st.text_input("Search job title", key="job_search_input")

    job_rows: List[dict] = []
    if job_query:
        job_rows = search_jobs_by_title(job_query, limit=25)
    if not job_rows:
        job_rows = list_jobs(limit=25)

    selected_job = None
    if job_rows:
        job_options = [f"{row['title']} @ {row['company']} ({row['job_id']})" for row in job_rows]
        job_map = {label: row for label, row in zip(job_options, job_rows)}
        default_job_idx = 0
        stored_job_id = st.session_state.get("selected_job_id")
        if stored_job_id:
            for idx, row in enumerate(job_rows):
                if row["job_id"] == stored_job_id:
                    default_job_idx = idx
                    break

        selected_job_label = st.selectbox(
            "Select a job",
            job_options,
            index=min(default_job_idx, len(job_options) - 1),
        )
        selected_job = job_map[selected_job_label]
        st.session_state["selected_job_id"] = selected_job["job_id"]
        prev_job_meta = st.session_state.get("job_candidate_results_meta", {})
        if prev_job_meta.get("job_id") != selected_job["job_id"]:
            st.session_state["job_candidate_results"] = []
            st.session_state["job_candidate_results_meta"] = {}
        st.success(f"Selected **{selected_job['title']}** at **{selected_job['company']}**.")
    else:
        st.error("No jobs found. Adjust the search query.")

    if selected_job:
        job_skills = get_job_skills(selected_job["job_id"])
        st.caption(f"{selected_job['title']} has {len(job_skills)} required skills in the graph.")
        st.session_state["selected_job_skills"] = job_skills
        if job_skills:
            with st.expander("Job skill requirements"):
                st.write(", ".join(job_skills))

        find_candidates = st.button("Find top candidates", key="find_candidates_btn")
        if find_candidates:
            with st.spinner("Ranking candidates..."):
                cand_rows = rank_candidates_for_job(
                    selected_job["job_id"],
                    k=top_k,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
            st.session_state["job_candidate_results"] = cand_rows
            st.session_state["job_candidate_results_meta"] = {
                "job_id": selected_job["job_id"],
                "job_title": selected_job["title"],
                "company": selected_job["company"],
            }

        cand_results = st.session_state.get("job_candidate_results", [])
        cand_meta = st.session_state.get("job_candidate_results_meta", {})
        if cand_results:
            st.success(f"Top {min(len(cand_results), top_k)} candidates for {cand_meta.get('job_title', 'job')}.")
            cand_df = pd.DataFrame(cand_results)
            st.dataframe(
                cand_df,
                use_container_width=True,
                hide_index=True,
            )
            if not llm_backend_ready():
                st.info("Select an AI backend in the sidebar to enable recruiter summaries.", icon="🤖")
            else:
                cand_labels = [f"{row['candidateName']} ({row['cand_id']})" for row in cand_results]
                cand_map = {label: row for label, row in zip(cand_labels, cand_results)}
                explain_cand_label = st.selectbox(
                    "Select a candidate to explain",
                    cand_labels,
                    key="cand_explain_select",
                )
                explain_row = cand_map[explain_cand_label]
                if st.button("Generate AI candidate summary", key="btn_employer_ai_summary"):
                    try:
                        cand_skill_list = get_candidate_skills(explain_row["cand_id"])
                        job_skill_list = st.session_state.get("selected_job_skills", job_skills)
                        summary = llm_candidate_for_job_summary(
                            cand_meta,
                            job_skill_list,
                            explain_row,
                            cand_skill_list,
                        )
                        st.markdown(summary)
                    except Exception as exc:
                        st.error(f"LLM candidate summary failed: {exc}")
        elif find_candidates:
            st.warning("No candidates were returned for this job. Try relaxing Top K or weights.")

    st.markdown("</div>", unsafe_allow_html=True)