# Desafio MBA Engenharia de Software com IA - Full Cycle

## Como executar

1. Crie o arquivo `.env` a partir do exemplo:
```bash
cp .env.example .env
```

2. Escolha o provedor em `AI_PROVIDER`:
- `openai` para usar `text-embedding-3-small` e `gpt-5-nano`
- `google` para usar `models/embedding-001` e `gemini-2.5-flash-lite`

3. Preencha a API key correspondente no `.env`.

4. Suba o PostgreSQL com `pgvector`:
```bash
docker compose up -d
```

5. Crie e ative o ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

6. Instale as dependências:
```bash
pip install -r requirements.txt
```

7. Execute a ingestão do PDF:
```bash
python src/ingest.py
```

8. Inicie o chat no terminal:
```bash
python src/chat.py
```

## Comandos úteis

- Fazer uma pergunta única:
```bash
python src/search.py "Qual o faturamento da Empresa SuperTechIABrazil?"
```

- Encerrar o chat: digite `sair`