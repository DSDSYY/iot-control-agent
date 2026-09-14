"""并发联调测试。

运行：
    python test_concurrency.py --requests 50 --concurrency 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio) - 1))
    return ordered[index]


def one_request(base_url: str, index: int, timeout: float) -> dict[str, float]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat",
        json={
            "message": "现在温度多少？",
            "thread_id": f"concurrency-{index}",
        },
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    reply = str(response.json().get("reply", "")).strip()
    if not reply:
        raise ValueError("空 reply")
    return {"latency_ms": latency_ms}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    started = time.perf_counter()
    latencies: list[float] = []
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(one_request, base_url, index, args.timeout): index
            for index in range(1, args.requests + 1)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
                latencies.append(result["latency_ms"])
            except Exception as exc:
                failures.append(f"#{index}: {exc}")

    elapsed = time.perf_counter() - started
    total = args.requests
    success = len(latencies)
    qps = total / elapsed if elapsed else 0.0

    print("并发联调结果")
    print(f"总请求: {total}")
    print(f"并发度: {args.concurrency}")
    print(f"成功: {success}/{total} ({success / total * 100:.1f}%)")
    print(f"吞吐: {qps:.1f} req/s")
    print(f"平均延迟: {statistics.mean(latencies):.1f}ms" if latencies else "平均延迟: N/A")
    print(f"P95 延迟: {percentile(latencies, 0.95):.1f}ms")
    if failures:
        print("失败样例:")
        for failure in failures[:10]:
            print(f"  {failure}")

    return 0 if success / total >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
