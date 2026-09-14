"""IoT Control Agent 评测脚本。

先启动 Broker、device_simulator.py 和 app.py，然后执行：

    python evaluate_agent.py

默认读取 eval_cases.json，逐条调用 /chat?debug=true，输出：
- 意图准确率
- 工具调用成功率
- 任务完成率
- 平均延迟与 P95 延迟

可选参数：
    --base-url http://127.0.0.1:8000
    --cases eval_cases.json
    --report evaluation_report.json
    --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("评测集必须是非空 JSON 数组")
    return data


def get_metrics(base_url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}
    except requests.RequestException:
        return {}


def call_agent(
    base_url: str,
    message: str,
    thread_id: str,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat",
        json={
            "message": message,
            "thread_id": thread_id,
            "debug": True,
        },
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("响应必须是 JSON 对象")
    return payload, latency_ms


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio) - 1))
    return ordered[index]


def intent_matches(case: dict[str, Any], trace: dict[str, Any]) -> bool:
    intent = trace.get("intent", {})
    return (
        str(intent.get("action", "")) == str(case.get("action", ""))
        and str(intent.get("target_device", "")) == str(case.get("target", ""))
        and str(intent.get("command", "")) == str(case.get("command", ""))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default="eval_cases.json")
    parser.add_argument("--report", default="evaluation_report.json")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    cases_path = Path(args.cases)
    report_path = Path(args.report)
    cases = load_cases(cases_path)

    metrics_before = get_metrics(base_url)
    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    print(f"评测服务: {base_url}")
    print(f"评测集: {cases_path.resolve()} ({len(cases)} cases)")
    print("=" * 72)

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", index))
        message = str(case.get("message", ""))
        thread_id = f"eval-{case_id}"

        try:
            payload, latency_ms = call_agent(
                base_url,
                message,
                thread_id,
                args.timeout,
            )
            latencies.append(latency_ms)
            reply = str(payload.get("reply", ""))
            trace = payload.get("trace", {})
            trace = trace if isinstance(trace, dict) else {}

            intent_ok = intent_matches(case, trace)
            tool_success = bool(trace.get("tool_success", False))
            expected_keywords = [
                str(keyword) for keyword in case.get("expected_keywords", [])
            ]
            keyword_hit = any(keyword in reply for keyword in expected_keywords)
            task_success = intent_ok and tool_success and keyword_hit

            result = {
                "id": case_id,
                "category": case.get("category", ""),
                "message": message,
                "reply": reply,
                "latency_ms": round(latency_ms, 1),
                "expected": {
                    "action": case.get("action", ""),
                    "target": case.get("target", ""),
                    "command": case.get("command", ""),
                    "keywords": expected_keywords,
                },
                "trace": trace,
                "intent_ok": intent_ok,
                "tool_success": tool_success,
                "keyword_hit": keyword_hit,
                "task_success": task_success,
            }
            results.append(result)

            print(
                f"[{index:02d}/{len(cases):02d}] {case_id} "
                f"intent={'OK' if intent_ok else 'FAIL'} "
                f"tool={'OK' if tool_success else 'FAIL'} "
                f"task={'OK' if task_success else 'FAIL'} "
                f"latency={latency_ms:.1f}ms"
            )
        except Exception as exc:
            latencies.append(0.0)
            result = {
                "id": case_id,
                "category": case.get("category", ""),
                "message": message,
                "error": str(exc),
                "intent_ok": False,
                "tool_success": False,
                "keyword_hit": False,
                "task_success": False,
            }
            results.append(result)
            print(f"[{index:02d}/{len(cases):02d}] {case_id} ERROR: {exc}")

    metrics_after = get_metrics(base_url)
    total = len(results)
    intent_success = sum(1 for item in results if item.get("intent_ok"))
    tool_success_count = sum(1 for item in results if item.get("tool_success"))
    task_success_count = sum(1 for item in results if item.get("task_success"))
    effective_latencies = [value for value in latencies if value > 0]

    summary = {
        "total_cases": total,
        "intent_success": intent_success,
        "intent_accuracy": round(intent_success / total * 100, 1) if total else 0.0,
        "tool_success": tool_success_count,
        "tool_success_rate": round(tool_success_count / total * 100, 1) if total else 0.0,
        "task_success": task_success_count,
        "task_success_rate": round(task_success_count / total * 100, 1) if total else 0.0,
        "average_latency_ms": round(statistics.mean(effective_latencies), 1) if effective_latencies else 0.0,
        "p95_latency_ms": round(percentile(effective_latencies, 0.95), 1),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }

    report = {"summary": summary, "results": results}
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入: {report_path.resolve()}")

    return 0 if summary["task_success_rate"] >= args.threshold * 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
