from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main() -> None:
    print('Faça sua pergunta. Digite "sair" para encerrar.')

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat encerrado.")
            break

        if not question:
            print('Digite uma pergunta válida ou "sair" para encerrar.')
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Chat encerrado.")
            break

        try:
            answer = search_prompt(question)
        except Exception as exc:
            print(f"Erro ao processar a pergunta: {exc}")
            continue

        print(f"RESPOSTA: {answer}")

if __name__ == "__main__":
    main()