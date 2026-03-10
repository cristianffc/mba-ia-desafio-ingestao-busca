from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_NO_CONTEXT_ANSWER = (
    "Não tenho informações necessárias para responder sua pergunta."
)


def get_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def require_env(*names: str) -> str:
    value = get_env(*names)
    if value is None:
        joined_names = " or ".join(names)
        raise RuntimeError(f"Environment variable {joined_names} is not set")
    return value


def get_provider() -> str:
    provider = get_env("AI_PROVIDER")
    if provider:
        normalized_provider = provider.strip().lower()
        if normalized_provider not in {"openai", "google"}:
            raise RuntimeError("AI_PROVIDER must be 'openai' or 'google'")
        return normalized_provider

    if get_env("OPENAI_API_KEY"):
        return "openai"
    if get_env("GOOGLE_API_KEY"):
        return "google"

    raise RuntimeError(
        "Set AI_PROVIDER and its matching API key, or provide OPENAI_API_KEY/GOOGLE_API_KEY."
    )


def get_pdf_path() -> Path:
    raw_path = get_env("PDF_PATH", default="document.pdf")
    pdf_path = Path(raw_path)

    if not pdf_path.is_absolute():
        pdf_path = BASE_DIR / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    return pdf_path


def get_database_url() -> str:
    return require_env("DATABASE_URL", "PGVECTOR_URL")


def get_collection_name() -> str:
    return get_env(
        "PG_VECTOR_COLLECTION_NAME",
        "PGVECTOR_COLLECTION_NAME",
        "PGVECTOR_COLLECTION",
        default="pdf_chunks",
    )


def get_embeddings() -> OpenAIEmbeddings | GoogleGenerativeAIEmbeddings:
    provider = get_provider()

    if provider == "google":
        require_env("GOOGLE_API_KEY")
        model = get_env("GOOGLE_EMBEDDING_MODEL", default="models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    require_env("OPENAI_API_KEY")
    model = get_env(
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_MODEL",
        default="text-embedding-3-small",
    )
    return OpenAIEmbeddings(model=model)


def get_llm() -> ChatOpenAI | ChatGoogleGenerativeAI:
    provider = get_provider()

    if provider == "google":
        require_env("GOOGLE_API_KEY")
        model = get_env("GOOGLE_CHAT_MODEL", default="gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    require_env("OPENAI_API_KEY")
    model = get_env("OPENAI_CHAT_MODEL", default="gpt-5-nano")
    return ChatOpenAI(model=model)


def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=get_collection_name(),
        connection=get_database_url(),
        use_jsonb=True,
    )


def message_to_text(message: Any) -> str:
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts).strip()

    return str(content).strip()
