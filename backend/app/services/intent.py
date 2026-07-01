"""
Intent extraction service.

Keeps the local LLM as a module-level singleton (mirrors how CLIP is held
in rag.py) so the model is loaded once per process.

Uses Ollama via LangChain for fully offline inference.
Model is configured via LLM_MODEL_NAME in config (e.g. "qwen2.5:1.5b").

Public API
----------
extract_intent(query: str) -> IntentFilter
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_ollama import ChatOllama

from backend.app.core.config import get_settings
from backend.app.core.exceptions import IntentExtractionError
from backend.app.core.logging import get_logger
from backend.app.models.intent import IntentFilter

logger = get_logger(__name__)

# ── Module-level LLM singleton ────────────────────────────────────────────────

_llm: Optional[ChatOllama] = None


def _load_llm() -> ChatOllama:
    """
    Load and cache the Ollama LLM.

    ChatOllama is used here for fully offline inference via a locally
    running Ollama server.  The model name is set in config as
    LLM_MODEL_NAME (e.g. "qwen2.5:1.5b").

    To swap back to LlamaCpp, replace ChatOllama(...) with:
        from langchain_community.llms import LlamaCpp
        return LlamaCpp(model_path=str(cfg.LLM_MODEL_PATH), ...)
    No other code needs to change.
    """
    global _llm
    if _llm is None:
        cfg = get_settings()
        logger.info("Loading Ollama LLM: model=%s", cfg.LLM_MODEL_NAME)
        _llm = ChatOllama(
            model=cfg.LLM_MODEL_NAME,
            base_url="http://localhost:11434",
            temperature=cfg.LLM_TEMPERATURE,
            num_predict=cfg.LLM_MAX_TOKENS,
            format="json",          # Ollama native JSON mode — forces valid JSON output
        )
        logger.info("Ollama LLM ready.")
    return _llm


# ── Prompt loader ─────────────────────────────────────────────────────────────

def _load_system_prompt() -> str:
    """
    Load the system prompt from QUERY_PROMPT_PATH.
    Falls back to a minimal inline prompt if the file is missing so the
    service degrades gracefully rather than crashing on startup.
    """
    cfg = get_settings()
    prompt_path = Path(cfg.QUERY_PROMPT_PATH)

    if prompt_path.exists():
        logger.debug("Loading prompt from: %s", prompt_path)
        return prompt_path.read_text(encoding="utf-8")

    logger.warning(
        "Prompt file not found at %s — using inline fallback prompt.", prompt_path
    )
    return _FALLBACK_PROMPT


# Minimal fallback used only if the prompt file is missing
_FALLBACK_PROMPT = """\
You are a structured-output parser for a surveillance camera alert system.
Extract search filters from the user query and return only valid JSON.
{format_instructions}
Today is {today}.
"""

_HUMAN_PROMPT = "User query: {query}"


# ── Chain builder ─────────────────────────────────────────────────────────────

def _build_chain():
    """
    Build the LangChain extraction chain.

    Chain:  ChatPromptTemplate → ChatOllama (JSON mode) → OutputFixingParser

    The chain is rebuilt on each call so {today}/{yesterday} stay current.
    The LLM singleton is reused, so this is cheap.
    """
    from datetime import date, timedelta

    today_str     = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()

    base_parser = PydanticOutputParser(pydantic_object=IntentFilter)

    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=_load_llm(),
        max_retries=2,
    )

    system_template = _load_system_prompt()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            system_template,
            partial_variables={
                "format_instructions": base_parser.get_format_instructions(),
                "today":     today_str,
                "yesterday": yesterday_str,
            },
        ),
        HumanMessagePromptTemplate.from_template(_HUMAN_PROMPT),
    ])

    return prompt | _load_llm() | fixing_parser


# ── Public API ────────────────────────────────────────────────────────────────

def extract_intent(query: str) -> IntentFilter:
    """
    Parse a natural-language surveillance query into structured filters.

    Returns IntentFilter on success.
    Falls back to a pure semantic search (no metadata filters) on any
    failure so the system never hard-crashes due to LLM issues.
    """
    if not query.strip():
        raise IntentExtractionError("Query must not be empty.")

    logger.info("=" * 60)
    logger.info("Extracting intent | query: %s", query)

    try:
        chain = _build_chain()
        result: IntentFilter = chain.invoke({"query": query})

        if not result.semantic_query:
            if result.label:
                result.semantic_query = result.label
            else:
                result.semantic_query = query
        logger.info("Intent extracted: %s", result)
        logger.info("=" * 60)
        return result

    except Exception as exc:
        import traceback
        logger.error("Intent extraction FAILED — falling back to semantic search")
        logger.error("Exception: %s", exc)
        traceback.print_exc()
        logger.info("=" * 60)

        _maybe_raise(exc)
        return IntentFilter(semantic_query=query)


def _maybe_raise(exc: Exception) -> None:
    """Re-raise only hard failures that make the service unusable."""
    if isinstance(exc, (MemoryError,)):
        raise IntentExtractionError(f"LLM out of memory: {exc}") from exc