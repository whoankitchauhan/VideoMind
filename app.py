import html
import time

import streamlit as st
from dotenv import load_dotenv

from core.rag_Engine import ask_question
from main import run_pipeline


load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="Video",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg: #0b0d12;
    --surface: #141821;
    --surface-2: #1d2430;
    --border: #2f3a4a;
    --accent: #4f8cff;
    --accent-2: #12b981;
    --text: #eef2f7;
    --muted: #9aa8ba;
}

.stApp { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
h1, h2, h3 { color: var(--text); }
.small-muted { color: var(--muted); font-size: 0.85rem; }
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.result-label {
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-body {
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.65;
    white-space: pre-wrap;
}
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    color: var(--text);
}
.chat-msg { margin-bottom: 0.9rem; }
.chat-user { color: var(--accent); font-weight: 700; }
.chat-assistant { color: var(--accent-2); font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "result": None,
        "chat_history": [],
        "pipeline_cache": {},
        "chat_cache": {},
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def cache_key(source: str, language: str) -> str:
    return f"{source.strip()}::{language.strip().lower()}"


def render_card(label: str, value: str):
    st.markdown(
        f"""
<div class="result-card">
    <div class="result-label">{html.escape(label)}</div>
    <div class="result-body">{html.escape(value)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_results(result: dict):
    render_card("Session Title", result["title"])

    summary_col, transcript_col = st.columns([3, 2], gap="medium")
    with summary_col:
        render_card("Summary", result["summary"])
    with transcript_col:
        with st.expander("Full Transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{html.escape(result["transcript"])}</div>',
                unsafe_allow_html=True,
            )

    item_col, decision_col, question_col = st.columns(3, gap="medium")
    with item_col:
        render_card("Action Items", result["action_items"])
    with decision_col:
        render_card("Key Decisions", result["key_decisions"])
    with question_col:
        render_card("Open Questions", result["open_questions"])


def render_chat(result: dict):
    st.subheader("Chat With Your Meeting")

    for message in st.session_state.chat_history:
        label = "You" if message["role"] == "user" else "Assistant"
        css_class = "chat-user" if message["role"] == "user" else "chat-assistant"
        st.markdown(
            f"""
<div class="chat-msg">
    <div class="{css_class}">{label}</div>
    <div>{html.escape(message["content"])}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="What were the main decisions made?",
        )
        submitted = st.form_submit_button("Send")

    if submitted and question.strip():
        normalized_question = question.strip()
        answer_cache_key = normalized_question.casefold()

        if answer_cache_key in st.session_state.chat_cache:
            answer = st.session_state.chat_cache[answer_cache_key]
        else:
            with st.spinner("Thinking..."):
                answer = ask_question(result["rag_chain"], normalized_question)
            st.session_state.chat_cache[answer_cache_key] = answer

        st.session_state.chat_history.append(
            {"role": "user", "content": normalized_question}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    if st.session_state.chat_history and st.button("Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


init_state()

with st.sidebar:
    st.title("AI Video Assistant")
    st.caption("Transcribe, summarize, extract decisions, and chat.")

    source = st.text_input(
        "YouTube URL or File Path",
        placeholder="https://youtube.com/watch?v=... or D:\\path\\file.mp4",
    )
    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    analyze_clicked = st.button(
        "Analyse",
        use_container_width=True,
        disabled=st.session_state.processing,
    )

    if st.session_state.result:
        st.success("Analysis ready")

st.title("AI Video Assistant")
st.markdown(
    '<div class="small-muted">A server-side Streamlit app for meeting intelligence.</div>',
    unsafe_allow_html=True,
)
st.divider()

if analyze_clicked:
    cleaned_source = source.strip()
    if not cleaned_source:
        st.error("Please enter a YouTube URL or local file path.")
    else:
        key = cache_key(cleaned_source, language)
        cached_result = st.session_state.pipeline_cache.get(key)

        if cached_result:
            st.session_state.result = cached_result
            st.session_state.chat_history = []
            st.session_state.chat_cache = {}
            st.info("Loaded cached analysis for this input.")
        else:
            st.session_state.processing = True
            st.session_state.chat_history = []
            st.session_state.chat_cache = {}

            try:
                with st.status("Running pipeline...", expanded=True) as status:
                    st.write("Processing audio and creating transcript.")
                    result = run_pipeline(cleaned_source, language)
                    st.write("Caching result for this session.")
                    st.session_state.pipeline_cache[key] = result
                    st.session_state.result = result
                    status.update(label="Analysis complete", state="complete")
                    time.sleep(0.4)
            except Exception as exc:
                st.error(f"Error: {exc}")
            finally:
                st.session_state.processing = False

if st.session_state.result:
    render_results(st.session_state.result)
    st.divider()
    render_chat(st.session_state.result)
else:
    st.info("Paste a YouTube URL or local file path in the sidebar, then run Analyse.")
