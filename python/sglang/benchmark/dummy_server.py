"""A lightweight server for inspecting bench_serving request payloads.

Usage:
  python3 python/sglang/benchmark/dummy_server.py --port 30000 \
    --log-file /tmp/bench_serving_requests.jsonl

Then point bench_serving at it, for example:
  python3 -m sglang.bench_serving --backend sglang \
    --base-url http://127.0.0.1:30000 --dataset-name sharegpt \
    --tokenizer gpt2 --num-prompts 4 --warmup-requests 0
"""

import argparse
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit


DEFAULT_MODEL_ID = "gpt2"
SENSITIVE_HEADER_NAMES = {"authorization", "cookie", "x-api-key"}


@dataclass(frozen=True)
class ServerConfig:
    model_id: str
    log_file: Optional[Path]
    stdout_body_chars: int
    response_text: str
    stream_delay_ms: float
    redact_headers: bool


class RequestLogger:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._next_id = 0
        self._lock = threading.Lock()
        self._log_file_handle = None
        if config.log_file:
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file_handle = config.log_file.open("a", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            if self._log_file_handle:
                self._log_file_handle.close()
                self._log_file_handle = None

    def record(self, entry: Dict[str, Any]) -> int:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            entry["request_id"] = request_id

            if self._log_file_handle:
                self._log_file_handle.write(
                    json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                self._log_file_handle.flush()

            self._print_summary(entry)
            return request_id

    def _print_summary(self, entry: Dict[str, Any]) -> None:
        request_id = entry["request_id"]
        method = entry["method"]
        path = entry["path"]
        body = entry.get("json")

        print(f"\n=== request #{request_id}: {method} {path} ===", flush=True)
        if body is None:
            return

        payload_summary = summarize_payload(body)
        if payload_summary:
            print(payload_summary, flush=True)

        limit = self.config.stdout_body_chars
        if limit > 0:
            text = json.dumps(body, ensure_ascii=False, indent=2)
            if len(text) > limit:
                text = text[:limit] + f"\n... <truncated; full body is in log file>"
            print(text, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_headers(headers: Dict[str, str], redact: bool) -> Dict[str, str]:
    if not redact:
        return headers
    return {
        name: ("<redacted>" if name.lower() in SENSITIVE_HEADER_NAMES else value)
        for name, value in headers.items()
    }


def summarize_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    parts = []
    for key in ("model", "stream", "max_tokens", "max_completion_tokens"):
        if key in payload:
            parts.append(f"{key}={payload[key]!r}")
    if "sampling_params" in payload and isinstance(payload["sampling_params"], dict):
        sampling_params = payload["sampling_params"]
        if "max_new_tokens" in sampling_params:
            parts.append(f"sampling_params.max_new_tokens={sampling_params['max_new_tokens']!r}")
        if "temperature" in sampling_params:
            parts.append(f"sampling_params.temperature={sampling_params['temperature']!r}")

    prompt_preview = get_prompt_preview(payload)
    if prompt_preview:
        parts.append(f"prompt_preview={prompt_preview!r}")

    return " | ".join(parts)


def get_prompt_preview(payload: Dict[str, Any], limit: int = 180) -> str:
    prompt: Any = payload.get("prompt")
    if prompt is None:
        prompt = payload.get("text")
    if prompt is None and "messages" in payload:
        prompt = payload["messages"]

    if prompt is None:
        return ""

    if isinstance(prompt, str):
        text = prompt
    else:
        text = json.dumps(prompt, ensure_ascii=False)

    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def response_token_count(payload: Dict[str, Any]) -> int:
    candidates = [
        payload.get("max_completion_tokens"),
        payload.get("max_tokens"),
        (payload.get("sampling_params") or {}).get("max_new_tokens")
        if isinstance(payload.get("sampling_params"), dict)
        else None,
    ]
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        return max(parsed, 1)
    return 1


def make_handler(config: ServerConfig, logger: RequestLogger):
    class BenchDummyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            self._handle_request()

        def do_POST(self) -> None:  # noqa: N802
            self._handle_request()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _handle_request(self) -> None:
            try:
                body_bytes = self._read_body()
                payload, body_text = self._parse_body(body_bytes)
                self._log_request(payload, body_text, len(body_bytes))

                path = urlsplit(self.path).path
                if self.command == "GET":
                    self._handle_get(path)
                    return

                if path in {"/v1/chat/completions", "/chat/completions"}:
                    self._handle_chat_completion(payload)
                elif path in {"/v1/completions", "/completions"}:
                    self._handle_completion(payload)
                elif path == "/generate":
                    self._handle_generate(payload)
                elif path in {"/flush_cache", "/start_profile", "/stop_profile"}:
                    self._send_json({"success": True})
                elif path in {"/v1/embeddings", "/embeddings"}:
                    self._handle_embeddings(payload)
                else:
                    self._send_json({"ok": True, "path": path})
            except BrokenPipeError:
                return
            except Exception as exc:  # pragma: no cover - defensive server path
                traceback.print_exc()
                self._send_json(
                    {"error": str(exc), "traceback": traceback.format_exc()},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return b""
            return self.rfile.read(length)

        def _parse_body(self, body_bytes: bytes) -> Tuple[Optional[Any], str]:
            if not body_bytes:
                return None, ""
            body_text = body_bytes.decode("utf-8", errors="replace")
            try:
                return json.loads(body_text), body_text
            except json.JSONDecodeError:
                return None, body_text

        def _log_request(
            self, payload: Optional[Any], body_text: str, body_size: int
        ) -> None:
            split = urlsplit(self.path)
            entry: Dict[str, Any] = {
                "timestamp": now_iso(),
                "client": self.client_address[0],
                "method": self.command,
                "path": split.path,
                "query": parse_qs(split.query),
                "headers": redact_headers(dict(self.headers), config.redact_headers),
                "body_size": body_size,
            }
            if payload is not None:
                entry["json"] = payload
            elif body_text:
                entry["body"] = body_text
            logger.record(entry)

        def _handle_get(self, path: str) -> None:
            if path in {"/", "/health", "/health_generate"}:
                self._send_json({"status": "ok"})
            elif path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": config.model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "sglang-dummy",
                            }
                        ],
                    }
                )
            elif path == "/server_info":
                self._send_json(
                    {
                        "model_id": config.model_id,
                        "dummy_server": True,
                        "internal_states": [],
                    }
                )
            else:
                self._send_json({"ok": True, "path": path})

        def _handle_chat_completion(self, payload: Optional[Any]) -> None:
            request = payload if isinstance(payload, dict) else {}
            completion_tokens = response_token_count(request)
            if request.get("stream", False):
                chunks = [
                    {
                        "id": "chatcmpl-dummy",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.get("model", config.model_id),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": config.response_text},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-dummy",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.get("model", config.model_id),
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": completion_tokens,
                            "total_tokens": completion_tokens,
                        },
                    },
                ]
                self._send_sse(chunks)
                return

            self._send_json(
                {
                    "id": "chatcmpl-dummy",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model", config.model_id),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": config.response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": completion_tokens,
                        "total_tokens": completion_tokens,
                    },
                }
            )

        def _handle_completion(self, payload: Optional[Any]) -> None:
            request = payload if isinstance(payload, dict) else {}
            completion_tokens = response_token_count(request)
            response = {
                "id": "cmpl-dummy",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.get("model", config.model_id),
                "choices": [
                    {
                        "index": 0,
                        "text": config.response_text,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens,
                },
            }
            if request.get("stream", False):
                self._send_sse([response])
            else:
                self._send_json(response)

        def _handle_generate(self, payload: Optional[Any]) -> None:
            request = payload if isinstance(payload, dict) else {}
            completion_tokens = response_token_count(request)
            response = {
                "text": config.response_text,
                "meta_info": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                },
            }
            if request.get("stream", False):
                self._send_sse([response])
            else:
                self._send_json(response)

        def _handle_embeddings(self, payload: Optional[Any]) -> None:
            request = payload if isinstance(payload, dict) else {}
            input_value = request.get("input", [])
            count = len(input_value) if isinstance(input_value, list) else 1
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [0.0],
                            "index": idx,
                        }
                        for idx in range(count)
                    ],
                    "model": request.get("model", config.model_id),
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                }
            )

        def _send_sse(self, chunks: Any) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in chunks:
                data = json.dumps(chunk, ensure_ascii=False).encode("utf-8")
                self.wfile.write(b"data: " + data + b"\n\n")
                self.wfile.flush()
                if config.stream_delay_ms > 0:
                    time.sleep(config.stream_delay_ms / 1000.0)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    return BenchDummyHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dummy SGLang/OpenAI-compatible server that logs "
        "bench_serving requests."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Model id returned by /v1/models and echoed in responses.",
    )
    parser.add_argument(
        "--log-file",
        default="bench_serving_dummy_requests.jsonl",
        help="JSONL file that receives full request records. Use '' to disable.",
    )
    parser.add_argument(
        "--stdout-body-chars",
        type=int,
        default=2000,
        help="Maximum request body characters printed to stdout. Use 0 for summary only.",
    )
    parser.add_argument(
        "--response-text",
        default="OK",
        help="Text returned in dummy completion responses.",
    )
    parser.add_argument(
        "--stream-delay-ms",
        type=float,
        default=0.0,
        help="Optional delay between streamed SSE chunks.",
    )
    parser.add_argument(
        "--no-redact-headers",
        action="store_true",
        help="Log sensitive headers such as Authorization without redaction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file) if args.log_file else None
    config = ServerConfig(
        model_id=args.model_id,
        log_file=log_file,
        stdout_body_chars=args.stdout_body_chars,
        response_text=args.response_text,
        stream_delay_ms=args.stream_delay_ms,
        redact_headers=not args.no_redact_headers,
    )
    logger = RequestLogger(config)
    handler = make_handler(config, logger)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(
        f"Dummy bench server listening on http://{args.host}:{args.port} "
        f"(model_id={args.model_id!r})",
        flush=True,
    )
    if log_file:
        print(f"Writing full request records to {log_file}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dummy bench server.", flush=True)
    finally:
        server.server_close()
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to start dummy bench server: {exc}", file=sys.stderr)
        raise
