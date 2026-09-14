"""SSE 流式接口测试客户端。

运行：
    python test_stream_client.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests


BASE_URL = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
THREAD_ID = os.getenv("AGENT_THREAD_ID", "stream-demo")
MESSAGE = os.getenv("AGENT_STREAM_MESSAGE", "现在温度多少？")


def main() -> int:
    started = time.perf_counter()
    first_event_at: float | None = None
    first_token_at: float | None = None
    event_type = ""
    event_types: list[str] = []
    answer_parts: list[str] = []

    with requests.post(
        f"{BASE_URL}/chat/stream",
        json={"message": MESSAGE, "thread_id": THREAD_ID},
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if first_event_at is None:
                first_event_at = time.perf_counter()

            line = raw_line or ""
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ").strip()
                event_types.append(event_type)
                print(f"event={event_type}")
                continue
            if not line.startswith("data: "):
                continue

            payload = json.loads(line.removeprefix("data: "))
            print(json.dumps(payload, ensure_ascii=False))

            if event_type == "token":
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                answer_parts.append(str(payload.get("content", "")))
            if event_type == "error":
                return 1

    total_seconds = time.perf_counter() - started
    print("=" * 72)
    print(f"事件类型: {event_types}")
    print(f"回答: {''.join(answer_parts)}")
    print(f"首事件延迟: {(first_event_at - started) * 1000:.1f}ms" if first_event_at else "首事件延迟: N/A")
    print(f"首 Token 延迟: {(first_token_at - started) * 1000:.1f}ms" if first_token_at else "首 Token 延迟: N/A")
    print(f"总耗时: {total_seconds * 1000:.1f}ms")

    return 0 if "done" in event_types else 1


if __name__ == "__main__":
    sys.exit(main())
