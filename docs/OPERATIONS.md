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

### HTTP `429` (`busy` / fila cheia)

- concorrência do runtime atingiu limite;
- fila de espera do runtime/jobs atingiu capacidade.

Ações sugeridas:

- respeitar `Retry-After`;
- reduzir picos de paralelismo do cliente;
- trocar para perfil/modelo menos congestionado;
- monitorar `GET /v1/runtime/status`.

### HTTP `504` (`solve_hard_timeout`)

- request ultrapassou o timeout hard do servidor;
- comum em tráfego alto no provider upstream.

Ações sugeridas:

- reenviar como job assíncrono (`/v1/jobs/solve`) e acompanhar status;
- diminuir complexidade do enunciado por requisição;
- ajustar `runtime_control.solve_hard_timeout_seconds` com cautela.

### HTTP `502` em `/v1/solve`

- falha na chamada do provider upstream durante execução de nó (ex.: `MaritalkHTTPError`).
- para Maritaca, pode ocorrer `HTTP 500 Internal Server Error` retornado pelo próprio endpoint do provider.

Ações sugeridas:

- tentar novamente (erro pode ser transitório do provider);
- validar se `provider/model_profile` estão compatíveis;
- reduzir `max_tokens` e manter `temperature` em faixa moderada;
- conferir no log o `request_id`, `provider`, `model` e `api_key_env` efetivo.

## OCR dedicado (RapidOCR)

### HTTP `503` em `/v1/ocr/extract`

- dependência `rapidocr_onnxruntime` ausente no ambiente da API.

Ação:

```bash
pip install -e .[dev]
```

### HTTP `422` em `/v1/solve` com erro de OCR obrigatório

- cenário típico: `ocr_mode=auto` com modelo não multimodal + imagem;
- ou `ocr_mode=on` sem enviar `ocr_text`.

Ações sugeridas:

- chamar `POST /v1/ocr/extract`, validar texto e reenviar em `ocr_text`;
- no Chainlit, manter `OCR Mode` em `auto`/`on` e confirmar OCR;
- se o modelo for multimodal e você não quiser OCR, use `ocr_mode=off`.

### OCR de baixa qualidade

- imagem desfocada, inclinada ou com contraste baixo;
- fórmula com ruído visual ou escrita muito pequena.

Ações sugeridas:

- reenviar imagem com maior resolução/corte mais fechado;
- ajustar `min_confidence` no `/v1/ocr/extract` para aceitar mais linhas quando necessário;
- revisar manualmente o texto OCR antes de confirmar cálculo.

## Playbook de congestionamento NVIDIA

1. Verifique `GET /v1/runtime/status`:
   - `in_flight_by_provider.nvidia`
   - `queue_depth`
   - `timeouts_last_5m_by_provider.nvidia`
2. Se `busy=true` com fila crescente:
   - use jobs assíncronos para absorver pico;
   - direcione parte do tráfego para outro profile/provider.
3. Se timeouts aumentarem:
   - reduza payloads longos;
   - ajuste limites de timeout e concorrência no `graph_config.yml`.

## Boas práticas

- rode API e Streamlit no mesmo virtualenv.
- mantenha `configs/graph_config.yml` versionado por ambiente.
- prefira trocar `llm.model_profile` em vez de alterar código para mudar de modelo.
- para modelos não multimodais (ex.: `deepseek_v3_2`), entrada de imagem é ignorada no payload de inferência.
- use `ocr_mode=auto` como padrão: OCR só é exigido quando o modelo não suporta multimodal.
- modelos Maritaca (`sabia_4`, `sabiazinho_4`) são text-to-text e não usam entrada multimodal.
- integração Maritaca no projeto usa `ChatMaritalk` (`langchain-community`).
- não armazene chaves em código; use `.env` ou secrets do runtime.
- monitore diretório de checkpoints (`.checkpoints/`) e limpe periodicamente.
