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
│   │   ├── chainlit_app.py
│   │   └── streamlit_app.py
│   └── main.py
├── tests/
├── docs/
├── .env.example
└── pyproject.toml
```

## Requisitos

- Python `>=3.13`
- Chave NVIDIA (`NVIDIA_API_KEY`) ou MARITACA (`MARITACA_API_KEY`)
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

## Seleção de modelo (flexível)

A seleção do modelo fica na seção `llm` do `configs/graph_config.yml`.

Campos principais:

- `model_profile`: alias do perfil (recomendado)
- `model`: id explícito do modelo (opcional; sobrescreve o id do perfil)
- `temperature`, `top_p`, `max_tokens`: overrides de amostragem
- `chat_template_kwargs`: kwargs específicos por modelo
- `multimodal_enabled`: habilita imagem quando o modelo suporta

Perfis prontos incluídos:

- `kimi_k2_5` -> `moonshotai/kimi-k2.5` (multimodal)
- `deepseek_v3_2` -> `deepseek-ai/deepseek-v3.2` (não multimodal)
- `glm4_7` -> `z-ai/glm4.7` (multimodal)
- `glm5` -> `z-ai/glm5` (multimodal)
- `minimax_m2_1` -> `minimaxai/minimax-m2.1` (multimodal)
- `sabia_4` -> `sabia-4` (Maritaca, não multimodal)
- `sabiazinho_4` -> `sabiazinho-4` (Maritaca, não multimodal)

Exemplo rápido (DeepSeek):

```yaml
llm:
  model_profile: deepseek_v3_2
  model: deepseek-ai/deepseek-v3.2
  temperature: 1.0
  top_p: 0.95
  max_tokens: 8192
  multimodal_enabled: true # será automaticamente efetivo como false para esse modelo
```

Exemplo rápido (Maritaca):

```yaml
llm:
  provider: maritaca
  model_profile: sabiazinho_4
  api_key_env: MARITACA_API_KEY
  temperature: 0.7
  max_tokens: 8192
  multimodal_enabled: true # será automaticamente efetivo como false (modelo text-to-text)
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
- `GET /v1/runtime/status`
- `POST /v1/jobs/solve`
- `GET /v1/jobs/{job_id}`
- `DELETE /v1/jobs/{job_id}`
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

Exemplo com override por requisição (provider/model):

```bash
curl -X POST "http://localhost:8000/v1/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Calcule a derivada de x^3",
    "provider": "maritaca",
    "model_profile": "sabiazinho_4",
    "temperature": 0.7,
    "max_tokens": 8192
  }'
```

Observação: a API resolve automaticamente `api_key_env` com base no `provider` quando esse campo não é enviado.

Exemplo async (jobs):

```bash
curl -X POST "http://localhost:8000/v1/jobs/solve" \
  -H "Content-Type: application/json" \
  -d '{"problem":"Calcule a derivada de x^3"}'
```

```bash
curl "http://localhost:8000/v1/jobs/<job_id>"
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

### UI Chainlit (chat de agentes)

Terminal 1 (API):

```bash
python3 -m src.main --mode api --host 0.0.0.0 --port 8001
```

Terminal 2 (UI):

```bash
chainlit run src/ui/chainlit_app.py -w
```

No Chainlit, mantenha `API Base URL` apontando para a API FastAPI (padrao: `http://localhost:8001`).

`timeout_seconds` na UI Chainlit vai de `60` a `1000` segundos (padrao `600`).
A lista de `model_profile` exibe todos os perfis do `graph_config.yml`; a combinacao efetiva `provider/profile` e validada no backend.

Para debug do frontend Chainlit, ajuste o nivel de log:

```bash
MATH_SOLVER_UI_LOG_LEVEL=INFO chainlit run src/ui/chainlit_app.py -w
```

Os logs exibem `request_id`, URL da API, tempo de chamada e status, ajudando a identificar onde ocorre timeout.

Se o Chainlit reclamar que `.chainlit/config.toml` esta desatualizado, remova a pasta local e rode novamente:

```bash
rm -rf .chainlit
chainlit run src/ui/chainlit_app.py -w
```

A UI Chainlit permite:

- chat estilo assistente com render de Markdown/LaTeX;
- upload de imagem (PNG/JPG) pelo clipe na caixa de mensagem;
- seleção de `provider`, `model_profile`, `temperature`, `max_tokens` e `session_id`;
- exibição de `decision_trace` antes da resposta final;
- consulta de status operacional (`runtime status`) e feedback de fila/ocupação;
- execução padrão via jobs assíncronos com polling de progresso (`queued/running/succeeded`);
- comando `/resume` para retomar checkpoint da sessão ativa.
- envio de overrides para `/v1/solve` sem precisar informar `api_key_env` (resolvido pelo backend).

## Modo estrito generativo

No `configs/graph_config.yml`, seção `llm`:

- `enabled: true`
- `require_available: true`

Com isso:

- sem LLM disponível, o agente retorna `failed_precondition`;
- nos endpoints REST (`/v1/solve` e `/v1/export`), a API retorna HTTP `503`.

Resiliência operacional adicional:

- `429` quando o runtime está ocupado (limites de concorrência/fila);
- `504` quando o request excede `solve_hard_timeout_seconds`;
- monitoramento em `GET /v1/runtime/status`.

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
