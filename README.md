# MathSolverAgent

Agente para resolução de problemas de matemática avançada com pipeline **inteiramente generativo**.
A LLM analisa o problema, converte para chamadas de ferramentas e explica a solução; as bibliotecas matemáticas executam somente os cálculos.

## Visão geral

- Pipeline: `analysis -> converter -> solving -> verification`.
- Modo estrito de IA: sem LLM disponível, o agente falha em `failed_precondition`.
- Suporte multimodal: texto, imagem, ou texto + imagem.
- API FastAPI + WebSocket + UI Streamlit.
- Checkpointing de sessão para retomada.

## Arquitetura

```text
math_solver_agent/
├── configs/
│   ├── graph_config.yml
│   └── prompts.yml
├── src/
│   ├── agents/
│   │   ├── graph.py
│   │   └── state.py
│   ├── api/
│   │   └── server.py
│   ├── llm/
│   │   └── client.py
│   ├── nodes/
│   │   ├── analyzer.py
│   │   ├── converter.py
│   │   ├── solver.py
│   │   └── verifier.py
│   ├── tools/
│   │   ├── calculator.py
│   │   ├── plotter.py
│   │   └── utils.py
│   ├── ui/
│   │   ├── api_client.py
│   │   └── streamlit_app.py
│   └── main.py
├── tests/
├── docs/
├── .env.example
└── pyproject.toml
```

## Requisitos

- Python `>=3.13`
- Chave NVIDIA (`NVIDIA_API_KEY`)
- Dependências do projeto

Instalação:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Configuração de ambiente:

```bash
cp .env.example .env
# editar .env e definir NVIDIA_API_KEY=...
```

## Execução

### CLI

Somente texto:

```bash
python3 -m src.main --mode cli --problem "Calcule a derivada de x^3"
```

Somente imagem:

```bash
python3 -m src.main --mode cli --image-path ./problema.png
```

Texto + imagem:

```bash
python3 -m src.main --mode cli --problem "Resolva" --image-path ./problema.png
```

Retomada de sessão:

```bash
python3 -m src.main --mode cli --problem "2+2" --session-id demo
python3 -m src.main --mode cli --resume --session-id demo
```

### API FastAPI

```bash
python3 -m src.main --mode api --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /v1/solve`
- `WS /v1/solve/stream`
- `POST /v1/export`

Exemplo `POST /v1/solve`:

```bash
curl -X POST "http://localhost:8000/v1/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Calcule a derivada de x^3",
    "session_id": "sessao-001",
    "resume": false
  }'
```

Exemplo multimodal (`image_url`):

```bash
curl -X POST "http://localhost:8000/v1/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "",
    "image_url": "https://exemplo.com/problema.png",
    "image_media_type": "image/png"
  }'
```

### UI Streamlit

Terminal 1 (API):

```bash
python3 -m src.main --mode api --host 0.0.0.0 --port 8000
```

Terminal 2 (UI):

```bash
streamlit run src/ui/streamlit_app.py
```

A UI permite:

- enviar somente texto;
- enviar somente imagem;
- enviar texto + imagem;
- escrever o enunciado em Markdown/LaTeX.

## Modo estrito generativo

No `configs/graph_config.yml`, seção `llm`:

- `enabled: true`
- `require_available: true`

Com isso:

- sem LLM disponível, o agente retorna `failed_precondition`;
- nos endpoints REST (`/v1/solve` e `/v1/export`), a API retorna HTTP `503`.

## Testes

Principal (unittest):

```bash
python3 -m unittest discover -s tests -v
```

Opcional (pytest):

```bash
pytest
```

## Documentação complementar

- API: `docs/API.md`
- Operação e troubleshooting: `docs/OPERATIONS.md`

## 📬 Contato

Projeto mantido por Thiago Bibiano. Para dúvidas, sugestões ou colaboração, entre em contato:

🔗 LinkedIn: https://www.linkedin.com/in/thiago-bibiano-da-silva-510b3b15b/
