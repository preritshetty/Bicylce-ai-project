import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import plotly.express as px   # ✅ added
from analyzer.modules.llm_agent import create_agent
from analyzer.modules.query_handler import run_query, pick_axes, pick_chart_type



# -------------------------------
# Environment setup
# -------------------------------
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="📊 LLM Data Analyst",
    page_icon="🤖",
    layout="wide"
)


# ==============================================================
# ✅ New reusable function
# ==============================================================
def render_analysis(df: pd.DataFrame | None):
    """
    Reuse the full analysis UI. If df is None, fall back to the original
    upload/default-loading logic from this file (so app.py still runs standalone).
    """
    # Always load the cleaner’s output
    DATA_PATH = "data/final_cleaned.csv"
    st.sidebar.header("📂 Data Source (Cleaner Output)")
    st.sidebar.info("This analyser always reads the cleaner’s final output.")

    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Expected cleaned file not found: {DATA_PATH}")
        st.stop()

    try:
        df = pd.read_csv(DATA_PATH)
        st.sidebar.success(f"✅ Loaded cleaned dataset: {DATA_PATH}")
    except Exception as e:
        st.error(f"❌ Failed to read {DATA_PATH}: {e}")
        st.stop()

    # ensure downstream code sees the same frame
    st.session_state["df"] = df

    # -------------------------------
    # Session State Init
    # -------------------------------
    st.session_state.setdefault("query_cache", {})     
    st.session_state.setdefault("query_history", [])   
    st.session_state.setdefault("current_cache_key", None)
    st.session_state.setdefault("answer_text", None)
    st.session_state.setdefault("proof_df", None)
    st.session_state.setdefault("current_query", "")
    st.session_state.setdefault("is_cached_result", False)
    st.session_state.setdefault("agent", None)

    # -------------------------------
    # Agent Creation
    # -------------------------------
    if st.session_state.agent is None:
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0,
            model=os.environ.get("MODEL_NAME", "gpt-4o")
        )
        st.session_state.agent = create_agent(llm, df)

    # -------------------------------
    # Embeddings setup for semantic cache
    # -------------------------------
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    def get_embedding(text: str) -> list:
        return st.session_state.embeddings.embed_query(text)

    def cosine_similarity(v1: list, v2: list) -> float:
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def find_similar_cached_query(query: str, min_similarity: float = 0.92) -> str | None:
        if not query or not st.session_state.query_cache:
            return None
        query_embedding = get_embedding(query.lower().strip())
        max_similarity, most_similar_key = 0, None
        for entry in st.session_state.query_history:
            cached_query = entry['query']
            cache_key = entry['cache_key']
            if 'embedding' not in entry:
                entry['embedding'] = get_embedding(cached_query.lower().strip())
            similarity = cosine_similarity(query_embedding, entry['embedding'])
            if similarity > max_similarity:
                max_similarity, most_similar_key = similarity, cache_key
        return most_similar_key if max_similarity >= min_similarity else None

    def make_cache_key(query: str) -> str:
        return f"q::{query.lower().strip()}"

    # -------------------------------
    # Sidebar: Query History + Cache Control
    # -------------------------------
    if st.session_state.query_history:
        st.sidebar.header("📝 Query History")
        for entry in st.session_state.query_history:
            is_cached = entry["cache_key"] in st.session_state.query_cache
            cache_status = "🔄 (cached)" if is_cached else "⚡ (new)"
            if st.sidebar.button(f"{entry['query']} {cache_status}", key=f"hist_{entry['cache_key']}"):
                q = entry['query']
                cache_key = entry['cache_key']
                st.session_state.current_cache_key = cache_key
                if cache_key in st.session_state.query_cache:
                    cached = st.session_state.query_cache[cache_key]
                    st.session_state.answer_text = cached["answer"]
                    st.session_state.proof_df = cached["proof_df"]
                    st.session_state.is_cached_result = True
                else:
                    with st.spinner("🤖 Agent is thinking..."):
                        response, proof_df = run_query(st.session_state.agent, q, df)
                    st.session_state.answer_text = response
                    st.session_state.proof_df = proof_df
                    st.session_state.is_cached_result = False
                    st.session_state.query_cache[cache_key] = {"answer": response, "proof_df": proof_df}
                st.session_state.current_query = q
                st.rerun()

    cache_size = len(st.session_state.query_cache)
    if cache_size > 0:
        st.sidebar.header("🗑️ Cache Control")
        st.sidebar.text(f"Cached queries: {cache_size}")
        if st.sidebar.button("🧹 Clear Cache"):
            st.session_state.query_cache.clear()
            st.session_state.query_history.clear()
            st.sidebar.success("Cache cleared!")
            st.rerun()

    # -------------------------------
    # Main Query Input
    # -------------------------------
    query = st.text_input("🔍 Ask a Question", 
                        placeholder="e.g., Which category has the most records?")

    if st.button("Run Analysis"):
        q = query.strip()
        if not q:
            st.warning("Please enter a question to analyze.")
        else:
            similar_cache_key = find_similar_cached_query(q)
            cache_key = similar_cache_key if similar_cache_key else make_cache_key(q)
            st.session_state.current_cache_key = cache_key

            if similar_cache_key:
                entry = st.session_state.query_cache[cache_key]
                st.session_state.answer_text = entry["answer"]
                st.session_state.proof_df = entry["proof_df"]
                st.session_state.is_cached_result = True
                st.session_state.current_query = q
                original_query = next(h['query'] for h in st.session_state.query_history if h['cache_key'] == cache_key)
                st.info(f"ℹ️ Using cached result from similar question: '{original_query}'")
            else:
                with st.spinner("🤖 Agent is thinking..."):
                    response, proof_df = run_query(st.session_state.agent, q, df)
                st.session_state.answer_text = response
                st.session_state.proof_df = proof_df
                st.session_state.is_cached_result = False
                st.session_state.query_cache[cache_key] = {"answer": response, "proof_df": proof_df}
                st.session_state.query_history.insert(0, {
                    "cache_key": cache_key, "query": q, "embedding": get_embedding(q.lower().strip())
                })
                st.session_state.current_query = q
            st.rerun()

    st.divider()

    # -------------------------------
    # Results Display
    # -------------------------------
    if st.session_state.answer_text:
        st.subheader("Answer")
        if st.session_state.is_cached_result:
            st.info("🔄 This answer was retrieved from cache", icon="ℹ️")
        st.success(st.session_state.answer_text)

    if st.session_state.proof_df is not None and not st.session_state.proof_df.empty:
        st.subheader("📑 Proof DataFrame")
        st.dataframe(st.session_state.proof_df.head(15), use_container_width=True)

        df_proof = st.session_state.proof_df
        x, y = pick_axes(df_proof)
        default_chart = pick_chart_type(x, st.session_state.current_query)

        st.subheader("📊 Visualization")
        chart_type = st.selectbox(
            "Choose chart type",
            ["Bar", "Line", "Scatter", "Pie", "Box"],
            index=["Bar", "Line", "Scatter", "Pie", "Box"].index(default_chart),
            key="chart_type_select"
        )

        try:
            if chart_type == "Bar":
                fig = px.bar(df_proof, x=x, y=y, title=f"{y} by {x}")
            elif chart_type == "Line":
                fig = px.line(df_proof, x=x, y=y, title=f"{y} over {x}")
            elif chart_type == "Scatter":
                fig = px.scatter(df_proof, x=x, y=y, title=f"{y} vs {x}")
            elif chart_type == "Pie":
                fig = px.pie(df_proof, names=x, values=y, title=f"{y} by {x}")
            elif chart_type == "Box":
                fig = px.box(df_proof, x=x, y=y, title=f"{y} by {x}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not render chart: {str(e)}")
    else:
        st.info("ℹ️ No proof DataFrame or visualization available for this query.")


# ==============================================================
# ✅ Keep standalone app functionality
# ==============================================================
def main():
    st.title("📊 LLM-Powered Data Analysis")
    st.markdown("Upload a CSV or use the default dataset. Ask natural language questions...")
    render_analysis(df=None)   # when run directly, fall back to upload/default


if __name__ == "__main__":
    main()
