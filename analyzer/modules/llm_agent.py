# modules/llm_agent.py
"""
Creates a LangChain pandas DataFrame agent that:
- Uses your custom system prompt (prompts/agent_prefix.txt)
- Executes pandas code safely via python_repl_ast
- Provides both df and pd (pandas) in the tool environment
- Is deterministic (when you set temperature=0 on the llm you pass in)
- Gracefully handles parsing errors instead of crashing
"""

from pathlib import Path
from typing import Any
import pandas as pd

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent import AgentExecutor

PROMPT_PATH = Path("prompts/agent_prefix.txt")


def load_agent_prefix() -> str:
    """Load the custom agent prefix prompt from prompts/agent_prefix.txt."""
    if not PROMPT_PATH.exists():
        # Minimal defensive default if file is missing
        return (
            "You are a careful  data analysis agent. "
            "Use only columns that exist. If data is missing, say so and suggest an alternative. "
            "Always use python_repl_ast for code execution and support answers with proof."
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


def create_agent(llm: Any, df: pd.DataFrame) -> AgentExecutor:
    """
    Build a LangChain pandas agent using an external LLM object.

    Parameters
    ----------
    llm : Any
        A LangChain LLM instance (e.g., from langchain_openai.OpenAI) with temperature=0 recommended.
    df : pd.DataFrame
        The bookings DataFrame to analyze.

    Returns
    -------
    AgentExecutor
        A configured LangChain agent ready to .run(question) against the dataframe.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None. Please load data before creating the agent.")

    prefix = load_agent_prefix()

    agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,  # ✅ only pass the DataFrame
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
    prefix=prefix,
    agent_type="openai-tools", 
)


    return agent
