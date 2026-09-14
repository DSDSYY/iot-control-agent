"""IoT Control Agent 联调测试客户端。

使用前先启动 Mosquitto、device_simulator.py 和 app.py，然后执行：

    python test_client.py

脚本会依次发送 4 条自然语言指令，统计 HTTP 成功率、关键词命中率
和平均响应延迟。可通过环境变量覆盖服务地址与会话 ID：

    AGENT_BASE_URL=http://127.0.0.1:8000
    AGENT_THREAD_ID=integration-demo
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Sequence

import requests


BASE_URL = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
THREAD_ID = os.getenv("AGENT_THREAD_ID", "integration-demo")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("AGENT_REQUEST_TIMEOUT", "60"))


@dataclass(frozen=True)
class TestCase:
    name: str
    message: str
    expected_keywords: Sequence[str]


TEST_CASES = [
    TestCase(
        name="查询温度",
        message="现在温度多少？",
        expected_keywords=("温度", "°C", "度"),
    ),
    TestCase(
        name="打开风扇",
        message="打开风扇",
        expected_keywords=("打开", "ON", "已开启", "已执行"),
    ),
    TestCase(
        name="温度阈值自动化",
        message="如果温度超过28度就打开风扇",
        expected_keywords=("温度", "超过", "未超过", "打开", "ON"),
    ),
    TestCase(
        name="查询风扇状态",
        message="风扇现在什么状态？",
        expected_keywords=("ON", "OFF", "开启", "关闭"),
    ),
]


def call_agent(message: str) -> tuple[str, float]:
    """调用 /chat，返回回复文本和耗时（秒）。"""

    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": message, "thread_id": THREAD_ID},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    elapsed = response.elapsed.total_seconds()

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text[:500]}"
        )

    payload = response.json()
    reply = str(payload.get("reply", "")).strip()
    if not reply:
        raise RuntimeError(f"/chat 返回了空 reply: {payload}")

    return reply, elapsed


def main() -> int:
    print(f"Agent 服务: {BASE_URL}")
    print(f"会话 ID: {THREAD_ID}")
    print("=" * 72)

    success_count = 0
    latency_values: list[float] = []

    for index, case in enumerate(TEST_CASES, start=1):
        print(f"[{index}/{len(TEST_CASES)}] {case.name}")
        print(f"用户: {case.message}")

        try:
            reply, elapsed = call_agent(case.message)
            latency_values.append(elapsed)

            keyword_hit = any(
                keyword in reply for keyword in case.expected_keywords
            )
            if keyword_hit:
                success_count += 1

            print(f"Agent: {reply}")
            print(
                f"结果: HTTP=200, 关键词={'命中' if keyword_hit else '未命中'}, "
                f"耗时={elapsed:.3f}s"
            )
        except requests.RequestException as exc:
            print(f"Agent: 请求失败: {exc}")
            print("结果: HTTP=FAIL")
        except Exception as exc:
            print(f"Agent: 处理失败: {exc}")
            print("结果: FAIL")

        print("-" * 72)

    total = len(TEST_CASES)
    average_latency = (
        sum(latency_values) / len(latency_values) if latency_values else 0.0
    )
    keyword_rate = success_count / total * 100

    print("联调汇总")
    print(f"用例总数: {total}")
    print(f"关键词命中: {success_count}/{total} ({keyword_rate:.1f}%)")
    print(f"平均响应延迟: {average_latency:.3f}s")
    print("说明: 关键词命中用于快速联调，不代表最终 Agent 评测准确率。")

    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())


