# API Contract

## Health

`GET /health`

Resposta:

```json
{"status":"ok"}
```

## Solve

`POST /v1/solve`

Payload:

- `problem` (`string`, opcional se imagem enviada)
- `session_id` (`string`, opcional)
- `resume` (`bool`, opcional)
- `analysis_only` (`bool`, opcional)
- `ocr_mode` (`string`, opcional: `auto|on|off`, default `auto`)
- `ocr_text` (`string`, opcional, texto OCR previamente confirmado)
- `image_path` (`string`, opcional)
- `image_url` (`string`, opcional)
- `image_base64` (`string`, opcional)
- `image_media_type` (`string`, opcional, default `image/png`)
- `images` (`array`, opcional): lista de objetos `{image_path?, image_url?, image_base64?, image_media_type?}`
- `provider` (`string`, opcional)
- `model_profile` (`string`, opcional)
- `model` (`string`, opcional)
- `api_key_env` (`string`, opcional)
- `temperature` (`number`, opcional)
- `top_p` (`number`, opcional)
- `max_tokens` (`integer`, opcional)

Regras:

- precisa enviar `problem` ou imagem (`image_*`/`images`) quando `resume=false`;
- `analysis_only=true` executa apenas o nó `analyzer` e retorna `normalized_problem`;
- `ocr_mode=auto`: OCR só é exigido quando o modelo efetivo não suporta multimodal;
- `ocr_mode=on`: OCR obrigatório sempre que houver imagem (via `ocr_text` ou `/v1/ocr/extract`);
- `ocr_mode=off`: OCR nunca roda; se modelo não multimodal receber apenas imagem, a API retorna `422`;
- no modo estrito, se LLM indisponível, retorna `503`.
- em falha de chamada ao provider, retorna `502`.

Resposta (resumo):

- `session_id`
- `status`
- `normalized_problem`
- `domain`
- `strategy`
- `llm`
- `has_visual_input`
- `result`
- `numeric_result`
- `verification`
- `explanation`
- `metrics`
- `decision_trace` (lista de eventos intermediários)
- `artifacts`
- `tool_call`

## Solve Streaming

`WS /v1/solve/stream`

Mensagens:

- `{"type":"trace","data":...}` para eventos do fluxo
- `{"type":"result","data":...}` no fim
- `{"type":"error","message":...}` em falha

Payload aceita os mesmos campos de entrada do `POST /v1/solve`, incluindo `images`, `analysis_only`, `ocr_mode` e `ocr_text`.

Eventos adicionais de runtime:

- `{"type":"queue_status","data":{"queue_position": number, "queue_wait_ms": number, "runtime": {...}}}` quando aplicável.

## Runtime Status

`GET /v1/runtime/status`

Resposta:

```json
{
  "busy": false,
  "in_flight_total": 1,
  "queue_depth": 0,
  "queue_capacity": 64,
  "in_flight_by_provider": {"nvidia": 1},
  "timeouts_last_5m_by_provider": {},
  "avg_latency_ms_last_5m": 812.4,
  "suggested_retry_after_seconds": 2
}
```

## Jobs (async)

### Submit

`POST /v1/jobs/solve`

Payload: mesmo contrato do `POST /v1/solve`.

Resposta:

```json
{
  "job_id": "a1b2c3...",
  "status": "queued",
  "queue_position": 3,
  "submitted_at": "2026-02-17T16:00:00Z"
}
```

### Status

`GET /v1/jobs/{job_id}`

Resposta:

- `status`: `queued|running|succeeded|failed|timeout|canceled`
- `queue_position`, `started_at`, `finished_at`, `error`, `result`

### Cancel

`DELETE /v1/jobs/{job_id}`

Cancela job em fila (ou marca cancelamento quando já em execução).

## OCR Extract

`POST /v1/ocr/extract`

Payload:

- `images` (`array`, obrigatório): lista `{image_path?, image_url?, image_base64?, image_media_type?}`
- `language_hint` (`string`, opcional)
- `min_confidence` (`number`, opcional)
- `merge_strategy` (`string`, opcional)

Resposta:

```json
{
  "text": "enunciado extraido",
  "pages": [
    {
      "index": 1,
      "text": "x^2 + 2x + 1 = 0",
      "confidence": 0.92,
      "lines": [{"text": "x^2 + 2x + 1 = 0", "confidence": 0.92}]
    }
  ],
  "engine": "rapidocr",
  "warnings": []
}
```

## Sessions

`GET /v1/sessions?limit=10`

Resposta:

```json
{
  "sessions": [
    {
      "session_id": "...",
      "updated_at": "2026-01-01T00:00:00Z",
      "status": "verified",
      "domain": "calculo_i",
      "problem_preview": "Calcule 2+2"
    }
  ]
}
```

## Export

`POST /v1/export`

Payload:

- mesmos campos de entrada de `solve`
- `resume` (`bool`, opcional; default automático para `true` quando `session_id` existe e `problem` está vazio)
- `format`: `latex|notebook|json`
- `output_path`

Resposta:

```json
{
  "session_id": "...",
  "format": "latex",
  "file_path": "artifacts/result.tex"
}
```

## Semântica de Erro Operacional

- `429 Too Many Requests`: servidor ocupado (limite de concorrência/fila).
- `504 Gateway Timeout`: execução excedeu `solve_hard_timeout_seconds`.
