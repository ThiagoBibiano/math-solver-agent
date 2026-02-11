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
- `image_path` (`string`, opcional)
- `image_url` (`string`, opcional)
- `image_base64` (`string`, opcional)
- `image_media_type` (`string`, opcional, default `image/png`)

Regras:

- precisa enviar `problem` ou imagem (`image_*`) quando `resume=false`;
- no modo estrito, se LLM indisponível, retorna `503`.

Resposta (resumo):

- `session_id`
- `status`
- `domain`
- `strategy`
- `llm`
- `has_visual_input`
- `result`
- `numeric_result`
- `verification`
- `explanation`
- `metrics`

## Solve Streaming

`WS /v1/solve/stream`

Mensagens:

- `{"type":"trace","data":...}` para eventos do fluxo
- `{"type":"result","data":...}` no fim
- `{"type":"error","message":...}` em falha

## Export

`POST /v1/export`

Payload:

- mesmos campos de entrada de `solve`
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
