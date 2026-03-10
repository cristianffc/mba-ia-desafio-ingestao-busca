from __future__ import annotations

import argparse

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

from common import DEFAULT_NO_CONTEXT_ANSWER, get_llm, get_vector_store, message_to_text


def _format_page(metadata: dict) -> str:
    page = metadata.get("page")
    if page is None:
        return "desconhecida"

    try:
        return str(int(page) + 1)
    except (TypeError, ValueError):
        return str(page)


def _format_context(results: list[tuple]) -> str:
    context_blocks: list[str] = []

    for index, (document, score) in enumerate(results, start=1):
        page = _format_page(document.metadata)
        header = f"Trecho {index} | página {page} | score {score:.4f}"
        context_blocks.append(f"{header}\n{document.page_content.strip()}")

    return "\n\n---\n\n".join(context_blocks)


def search_prompt(question: str | None = None) -> str:
    if question is None or not question.strip():
        raise ValueError("Informe uma pergunta válida.")

    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(question.strip(), k=10)

    if not results:
        return DEFAULT_NO_CONTEXT_ANSWER

    context = _format_context(results)
    if not context:
        return DEFAULT_NO_CONTEXT_ANSWER

    prompt = PROMPT_TEMPLATE.format(contexto=context, pergunta=question.strip())
    response = get_llm().invoke(prompt)
    answer = message_to_text(response)

    return answer or DEFAULT_NO_CONTEXT_ANSWER


def main() -> None:
    parser = argparse.ArgumentParser(description="Realiza busca semântica no PDF ingerido.")
    parser.add_argument("question", nargs="*", help="Pergunta a ser respondida.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("PERGUNTA: ").strip()

    if not question:
        raise SystemExit("Nenhuma pergunta foi informada.")

    print(f"RESPOSTA: {search_prompt(question)}")


if __name__ == "__main__":
    main()