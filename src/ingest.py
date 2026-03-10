from __future__ import annotations

import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import get_collection_name, get_pdf_path, get_vector_store


def build_document_id(document: Document, chunk_index: int) -> str:
    source = document.metadata.get("source", "document")
    page = document.metadata.get("page", "unknown")
    raw_id = f"{source}:{page}:{chunk_index}:{document.page_content}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def ingest_pdf() -> None:
    pdf_path = get_pdf_path()
    documents = PyPDFLoader(str(pdf_path)).load()

    if not documents:
        raise RuntimeError(f"Nenhum conteúdo foi carregado do PDF: {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    )
    splits = splitter.split_documents(documents)

    if not splits:
        raise RuntimeError("Nenhum chunk foi gerado a partir do PDF informado.")

    enriched_documents: list[Document] = []
    ids: list[str] = []

    for chunk_index, split in enumerate(splits):
        content = split.page_content.strip()
        if not content:
            continue

        metadata = {
            key: value for key, value in split.metadata.items() if value not in ("", None)
        }
        metadata["chunk_index"] = chunk_index
        metadata.setdefault("source", pdf_path.name)

        document = Document(page_content=content, metadata=metadata)
        enriched_documents.append(document)
        ids.append(build_document_id(document, chunk_index))

    if not enriched_documents:
        raise RuntimeError("Nenhum chunk válido foi encontrado para ingestão.")

    store = get_vector_store()
    store.add_documents(documents=enriched_documents, ids=ids)

    print(f"PDF carregado: {pdf_path.name}")
    print(f"Páginas lidas: {len(documents)}")
    print(f"Chunks salvos: {len(enriched_documents)}")
    print(f"Coleção: {get_collection_name()}")


if __name__ == "__main__":
    ingest_pdf()