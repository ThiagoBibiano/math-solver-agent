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
- ou `MARITACA_API_KEY` não está definido quando `provider=maritaca`.
- confirme no mesmo shell que inicia o servidor.

### `missing_dependency_langchain_nvidia_ai_endpoints`

- dependência ausente no ambiente ativo.

Ação:

```bash
pip install -e .[dev]
```

### `missing_dependency_langchain_community`

- dependência `langchain-community` ausente no ambiente ativo (necessária para integração com Maritaca via `ChatMaritalk`).

Ação:

```bash
pip install -e .[dev]
```

### HTTP `503` em `/v1/solve`

- LLM indisponível no modo estrito (`require_available: true`).
- ajuste credenciais/dependências ou altere configuração conscientemente.

### HTTP `502` em `/v1/solve`

- falha na chamada do provider upstream durante execução de nó (ex.: `MaritalkHTTPError`).
- para Maritaca, pode ocorrer `HTTP 500 Internal Server Error` retornado pelo próprio endpoint do provider.

Ações sugeridas:

- tentar novamente (erro pode ser transitório do provider);
- validar se `provider/model_profile` estão compatíveis;
- reduzir `max_tokens` e manter `temperature` em faixa moderada;
- conferir no log o `request_id`, `provider`, `model` e `api_key_env` efetivo.

## Boas práticas

- rode API e Streamlit no mesmo virtualenv.
- mantenha `configs/graph_config.yml` versionado por ambiente.
- prefira trocar `llm.model_profile` em vez de alterar código para mudar de modelo.
- para modelos não multimodais (ex.: `deepseek_v3_2`), entrada de imagem é ignorada no payload de inferência.
- modelos Maritaca (`sabia_4`, `sabiazinho_4`) são text-to-text e não usam entrada multimodal.
- integração Maritaca no projeto usa `ChatMaritalk` (`langchain-community`).
- não armazene chaves em código; use `.env` ou secrets do runtime.
- monitore diretório de checkpoints (`.checkpoints/`) e limpe periodicamente.
