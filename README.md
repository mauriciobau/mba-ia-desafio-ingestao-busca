# Ingestão e Busca Semântica com LangChain e Postgres

Sistema de busca semântica que ingere um PDF, armazena embeddings no PostgreSQL (pgVector) e permite consultas via CLI.

## Pré-requisitos

- Python 3.11+
- Docker & Docker Compose
- API Key do Google Gemini (ou OpenAI)

## Configuração

### 1. Clonar e configurar ambiente

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

Copie o arquivo de exemplo e configure suas API keys:

```bash
cp .env.example .env
```

Edite o `.env` com suas credenciais:
```env
GOOGLE_API_KEY=sua_api_key_aqui
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=document_vectors
PDF_PATH=document.pdf
```

### 3. Iniciar banco de dados

```bash
docker compose up -d
```

## Execução

### 1. Ingestão do PDF

```bash
python src/ingest.py
```

Este script:
- Carrega o PDF (`document.pdf`)
- Divide em chunks de 1000 caracteres (overlap: 150)
- Gera embeddings com Google Gemini
- Armazena no PostgreSQL com pgVector

### 2. Chat Interativo

```bash
python src/chat.py
```

Digite suas perguntas e receba respostas baseadas no conteúdo do PDF.
Para sair: `sair`, `exit`, `quit` ou Ctrl+C.

## Estrutura do Projeto

```
.
├── docker-compose.yml    # PostgreSQL + pgVector
├── requirements.txt      # Dependências Python
├── .env                  # Variáveis de ambiente
├── document.pdf          # PDF para ingestão
├── src/
│   ├── ingest.py         # Script de ingestão
│   ├── search.py         # Lógica de busca semântica
│   └── chat.py           # CLI para interação
└── README.md
```

## Tecnologias

- **LangChain** - Framework de IA
- **PostgreSQL + pgVector** - Banco vetorial
- **Google Gemini** - Embeddings e LLM
- **Python** - Linguagem principal