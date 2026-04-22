"""
LLM Factory — supports Groq, Gemini, or OpenAI.
Set LLM_PROVIDER=groq | gemini | openai in .env
"""
import os
from langchain_core.language_models.chat_models import BaseChatModel

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

def get_llm(temperature: float = 0, max_tokens: int = 512) -> BaseChatModel:
    """Return the configured LLM instance."""

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=1,
            timeout=15.0,
        )

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use groq | gemini | openai")

def supports_vision() -> bool:
    """Returns True if the current provider+model supports image input."""
    if LLM_PROVIDER == "openai":
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return "gpt-4o" in model
    if LLM_PROVIDER == "gemini":
        return True   # all Gemini 1.5+ support vision
    return False      # Groq text-only
