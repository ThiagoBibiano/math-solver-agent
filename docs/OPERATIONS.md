# Operação e Troubleshooting

## Diagnóstico rápido de LLM

Verifique no payload de resposta do agente:

- `llm.available`
- `llm.reason`
- `llm.api_key_present`
- `llm.api_key_env`

## Causas comuns

### `missing_api_key`

- `NVIDIA_API_KEY` não está definido no processo da API.
- confirme no mesmo shell que inicia o servidor.

### `missing_dependency_langchain_nvidia_ai_endpoints`

- dependência ausente no ambiente ativo.

Ação:

```bash
pip install -e .[dev]
```

### HTTP `503` em `/v1/solve`

- LLM indisponível no modo estrito (`require_available: true`).
- ajuste credenciais/dependências ou altere configuração conscientemente.

## Boas práticas

- rode API e Streamlit no mesmo virtualenv.
- mantenha `configs/graph_config.yml` versionado por ambiente.
- prefira trocar `llm.model_profile` em vez de alterar código para mudar de modelo.
- para modelos não multimodais (ex.: `deepseek_v3_2`), entrada de imagem é ignorada no payload de inferência.
- não armazene chaves em código; use `.env` ou secrets do runtime.
- monitore diretório de checkpoints (`.checkpoints/`) e limpe periodicamente.
