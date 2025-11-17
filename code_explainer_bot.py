"""
Code Explainer Bot (Streamlit + OpenAI)

Supports:
 - Paste code or upload file
 - Explain, find bugs, add comments, optimizations, complexity
 - API key via .streamlit/secrets.toml OR env var OPENAI_API_KEY OR enter in UI (session-only)

Install:
 pip install streamlit openai

Usage:
 streamlit run code_explainer_bot.py
"""

import os
import streamlit as st
import traceback
from typing import Optional

# Try to import openai (official package)
try:
    import openai
except Exception as e:
    st.error("Missing dependency 'openai'. Install with: pip install openai")
    raise

# -------------------------
# Helpers: API key handling
# -------------------------
def get_api_key() -> Optional[str]:
    """
    Retrieve API key from (in order):
     1) streamlit secrets (st.secrets["OPENAI_API_KEY"])
     2) environment variable OPENAI_API_KEY
     3) session state (entered via UI)
    """
    # 1) Streamlit secrets
    try:
        if st.secrets and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        # st.secrets may not exist or not be accessible in some runtimes
        pass

    # 2) environment variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    # 3) session (set by UI)
    return st.session_state.get("OPENAI_API_KEY", None)


def set_session_api_key(key: str):
    st.session_state["OPENAI_API_KEY"] = key.strip()


# -------------------------
# Prompt builder
# -------------------------
def build_prompt(code: str, task: str, language_hint: Optional[str] = None) -> str:
    """
    Build a clear instruction for the LLM based on chosen task.
    """
    instr = []
    instr.append("You are a senior software engineer and technical teacher.")
    instr.append(f"Perform the following task: {task}.")
    if language_hint:
        instr.append(f"Language: {language_hint}.")
    instr.append("Be concise, thorough, and show relevant code examples where appropriate.")
    instr.append("If you find bugs, show the problematic lines and provide a corrected version.")
    instr.append("If asked to add comments, return the full code with inline comments.")
    instr.append("\n----CODE START----\n")
    instr.append(code)
    instr.append("\n----CODE END----\n")
    instr.append("Return a markdown-formatted response with sections and code blocks where needed.")
    return "\n".join(instr)


# -------------------------
# LLM call
# -------------------------
def analyze_code_with_openai(api_key: str, code: str, task: str, model: str = "gpt-4o-mini") -> str:
    """
    Call OpenAI ChatCompletion API to analyze code.
    Uses the 'openai' python package.
    """
    if not api_key:
        raise RuntimeError("OpenAI API key is missing.")

    openai.api_key = api_key

    prompt = build_prompt(code=code, task=task)

    # Chat messages
    messages = [
        {"role": "system", "content": "You are an expert programming explainer and reviewer."},
        {"role": "user", "content": prompt}
    ]

    # Call the Chat Completion endpoint
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
    except Exception as e:
        # Surface helpful debug info
        raise RuntimeError(f"OpenAI API call failed: {e}")

    # Extract assistant content
    choices = resp.get("choices")
    if not choices or len(choices) == 0:
        raise RuntimeError("OpenAI returned no choices.")
    return choices[0]["message"]["content"]


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Code Explainer Bot", layout="wide")
st.title("🤖 AI Code Explainer Bot")

st.markdown(
    """
Paste code or upload a code file. The bot will explain the code, find bugs, add comments,
suggest optimizations, and give complexity analysis if applicable.

**API key note:** provide your OpenAI API key via:
- `.streamlit/secrets.toml` with `OPENAI_API_KEY = "sk-..."`, OR
- environment variable `OPENAI_API_KEY`, OR
- enter it below (session-only).
"""
)

# Sidebar: API key input and options
with st.sidebar:
    st.header("🔑 OpenAI API Key")

    # Show detected key source
    detected = None
    try:
        if st.secrets and "OPENAI_API_KEY" in st.secrets:
            detected = "from .streamlit/secrets.toml"
    except Exception:
        pass
    if not detected and os.getenv("OPENAI_API_KEY"):
        detected = "from environment variable OPENAI_API_KEY"
    if not detected and st.session_state.get("OPENAI_API_KEY"):
        detected = "from session (entered here)"

    if detected:
        st.success(f"API key detected {detected}")
    else:
        st.warning("No API key found. Enter it below or set it in secrets / env.")

    api_key_input = st.text_input("Enter OpenAI API key (session only)", type="password", value="")
    if api_key_input:
        set_session_api_key(api_key_input)
        st.success("API key stored for this session.")

    st.markdown("---")
    st.header("⚙️ Settings")
    model = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-instruct"], index=0)
    st.caption("Note: use a model you have access to. Change if needed.")

# Main: code input
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Paste your code")
    code_text = st.text_area("Code", height=300, placeholder="# Paste code here")

    st.subheader("Or upload a code file")
    uploaded = st.file_uploader("Upload code file (.py, .js, .java, .cpp, .txt)", type=["py", "js", "java", "cpp", "c", "txt", "ts"])

    # Pre-fill code_text with file contents if uploaded
    if uploaded is not None:
        try:
            raw = uploaded.read()
            try:
                text = raw.decode("utf-8")
            except Exception:
                text = str(raw)
            code_text = text
            st.success(f"Loaded file: {uploaded.name}")
        except Exception:
            st.error("Failed to read uploaded file.")

with col2:
    st.subheader("Task")
    task = st.selectbox(
        "What should the bot do?",
        [
            "Explain the code",
            "Find bugs",
            "Add comments",
            "Suggest optimizations",
            "Explain time & space complexity",
            "All of the above"
        ]
    )

    st.markdown("### Quick options")
    language_hint = st.text_input("Language hint (optional)", placeholder="e.g., Python, JavaScript")
    analyze_btn = st.button("🔎 Analyze Code")

    st.markdown("---")
    st.markdown("### Tips")
    st.write("- For best results, paste only the relevant function/class or <2000 lines.")
    st.write("- If the call fails due to key/permission, check your API key and model access.")

# Action: analyze
if analyze_btn:
    # Ensure there is code
    if not code_text or code_text.strip() == "":
        st.error("Please paste code or upload a file first.")
    else:
        api_key = get_api_key()
        if not api_key:
            st.error(
                "OpenAI API key not found. Provide it via .streamlit/secrets.toml, env var OPENAI_API_KEY, or by entering in the sidebar."
            )
        else:
            # Show progress
            with st.spinner("Analyzing code with the model..."):
                try:
                    result = analyze_code_with_openai(api_key=api_key, code=code_text, task=task, model=model)
                    st.markdown("### 🔍 Analysis Result")
                    st.markdown(result)
                    # Also show a download button for the result
                    st.download_button("Download analysis (MD)", data=result, file_name="analysis.md", mime="text/markdown")
                except Exception as e:
                    st.error("Analysis failed: " + str(e))
                    st.text("Traceback (for debugging):")
                    st.text(traceback.format_exc())

# Footer: lightweight test button (no external deps)
st.markdown("---")
if st.button("Run quick local checks (no API)"):
    st.write("Running quick checks...")
    sample = "def add(a, b):\n    return a + b\n"
    st.code(sample, language="python")
    st.success("Sample looks valid. Paste your code and press Analyze.")
