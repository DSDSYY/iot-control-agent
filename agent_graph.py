"""LangGraph Agent 核心。

Agent 负责：
1. 解析用户自然语言意图；
2. 查询 MCP 工具中的设备状态/传感器读数；
3. 生成受限 Python 代码并通过 MCP 工具执行设备控制；
4. 把工具结果转换为自然语言，并保留 thread_id 级别的多轮记忆。

安全说明：
- 模型生成的代码只允许一条 result = <allowed_tool>(...) 调用；
- 使用 AST 白名单，禁止 import、open、eval、exec、属性访问和双下划线名称；
- exec 命名空间不暴露 __builtins__，只暴露三个同步工具包装器；
- 该方案用于项目演示。生产环境建议改成结构化工具调用，不执行模型生成代码。
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("agent-graph")

BASE_DIR = Path(__file__).resolve().parent
MCP_SERVER_SCRIPT = BASE_DIR / "mcp_server.py"
MCP_PROTOCOL_VERSION = "2025-06-18"
ALLOWED_TOOLS = {"send_command", "read_sensor", "get_device_status"}

INTENT_PROMPT = """
你是一个 IoT 设备控制意图解析器。请只输出一个合法 JSON 对象，不要输出 Markdown。

输出格式：
{
  "action": "query" | "control",
  "target_device": "temp_01" | "fan_01" | "",
  "command": "ON" | "OFF" | "",
  "condition": "必要的补充说明，没有则为空字符串"
}

设备映射规则：
- 温度、温度传感器、传感器、temp、temp_01 -> temp_01
- 风扇、电扇、fan、fan_01 -> fan_01
- 如果用户要查询但没有明确设备，target_device 可以为空；
- 如果用户要控制但没有明确 ON/OFF，command 必须为空；
- action=query 时 command 固定为空；
- 不要把“打开查询界面”等模糊表达擅自解释成控制命令。

示例：
"帮我看看温度" -> {"action":"query","target_device":"temp_01","command":"","condition":""}
"打开风扇" -> {"action":"control","target_device":"fan_01","command":"ON","condition":""}
"把风扇处理一下" -> {"action":"control","target_device":"fan_01","command":"","condition":"需要明确打开还是关闭风扇"}
""".strip()

RESPOND_PROMPT = """
你是智能家居 Agent。请根据工具结果，用简洁、自然的中文回答用户。
要求：
1. 不得编造工具结果中不存在的信息；
2. 如果 success=false，明确说明失败原因和下一步建议；
3. 回答中保留关键数值、设备状态和单位；
4. 不要输出 JSON，除非用户明确要求。
""".strip()


class AgentState(TypedDict):
    """LangGraph 在节点之间传递的状态。"""

    messages: Annotated[list[BaseMessage], add_messages]
    intent: dict[str, Any]
    tool_result: str
    final_answer: str


_llm: ChatOpenAI | None = None
_mcp_session: "MCPStdioSession | None" = None
_main_loop: asyncio.AbstractEventLoop | None = None
_streaming_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agent_streaming_mode",
    default=False,
)


def _get_llm() -> ChatOpenAI:
    """延迟创建 LLM，避免仅导入模块时就要求 API Key。"""

    global _llm

    if _llm is not None:
        return _llm

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_key":
        raise RuntimeError("请先在 .env 中配置 OPENAI_API_KEY")

    model_name = os.getenv("MODEL_NAME", "deepseek-chat").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url

    _llm = ChatOpenAI(**kwargs)
    return _llm


def _content_to_text(content: Any) -> str:
    """兼容 OpenAI 兼容接口返回的字符串或多段 content 结构。"""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


def _strip_code_fence(text: str) -> str:
    """去掉模型偶尔附加的 Markdown 代码围栏。"""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取第一个 JSON 对象。"""

    cleaned = _strip_code_fence(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"模型没有返回 JSON 对象: {text!r}")

    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("意图解析结果必须是 JSON 对象")
    return value


class MCPToolExecutionError(RuntimeError):
    """MCP 工具已返回业务错误，不应自动重试。"""


class MCPStdioSession:
    """管理一个 FastMCP stdio 子进程，并提供 JSON-RPC 工具调用客户端。"""

    def __init__(self, server_script: Path, request_timeout: float = 20.0) -> None:
        self.server_script = server_script
        self.request_timeout = request_timeout
        self.process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._metrics: dict[str, Any] = {
            "tool_calls_total": 0,
            "tool_calls_success": 0,
            "tool_calls_failed": 0,
            "tool_call_retries": 0,
            "tool_latency_ms": [],
        }

    async def start(self) -> None:
        """启动 mcp_server.py 并完成 MCP initialize 握手。"""

        if not self.server_script.exists():
            raise FileNotFoundError(f"MCP Server 不存在: {self.server_script}")

        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                str(self.server_script),
                cwd=str(self.server_script.parent),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._reader_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            initialize_result = await self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "iot-control-agent",
                        "version": "1.0.0",
                    },
                },
            )
            if "error" in initialize_result:
                raise RuntimeError(
                    f"MCP initialize 失败: {initialize_result['error']}"
                )

            await self._notify("notifications/initialized", {})
            LOGGER.info("MCP Server 已启动并完成初始化")
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """停止 MCP 子进程并清理异步任务。"""

        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

        tasks = [
            task
            for task in (self._reader_task, self._stderr_task)
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._reader_task = None
        self._stderr_task = None
        self.process = None
        LOGGER.info("MCP Server 已停止")

    def metrics_snapshot(self) -> dict[str, Any]:
        """返回 MCP 工具调用统计，供 /metrics 与评测脚本使用。"""

        total = int(self._metrics["tool_calls_total"])
        success = int(self._metrics["tool_calls_success"])
        failed = int(self._metrics["tool_calls_failed"])
        latencies = list(self._metrics["tool_latency_ms"])
        success_rate = success / total * 100 if total else 0.0
        average_latency = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_latencies = sorted(latencies)
        if sorted_latencies:
            p95_index = min(
                len(sorted_latencies) - 1,
                int(len(sorted_latencies) * 0.95),
            )
            p95_latency = sorted_latencies[p95_index]
        else:
            p95_latency = 0.0

        return {
            "tool_calls_total": total,
            "tool_calls_success": success,
            "tool_calls_failed": failed,
            "tool_call_retries": int(self._metrics["tool_call_retries"]),
            "tool_call_success_rate": round(success_rate, 1),
            "tool_call_avg_latency_ms": round(average_latency, 1),
            "tool_call_p95_latency_ms": round(p95_latency, 1),
        }

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> str:
        """调用 MCP 工具；仅对超时/连接类错误做指数退避重试。"""

        effective_timeout = timeout or self.request_timeout
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            try:
                result = await self._call_tool_once(
                    name,
                    arguments,
                    timeout=effective_timeout,
                )
                self._record_tool_result(result, started)
                return result
            except MCPToolExecutionError as exc:
                self._record_tool_failure(started)
                raise
            except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt < max_retries:
                    self._metrics["tool_call_retries"] += 1
                    backoff_seconds = 0.2 * (2**attempt)
                    LOGGER.warning(
                        "MCP 工具 %s 调用失败，%.1fs 后重试 (%s/%s): %s",
                        name,
                        backoff_seconds,
                        attempt + 1,
                        max_retries,
                        exc,
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue

                self._record_tool_failure(started)
                raise RuntimeError(
                    f"MCP 工具 {name} 在 {max_retries + 1} 次尝试后失败: {exc}"
                ) from exc

        raise RuntimeError(f"MCP 工具 {name} 调用失败: {last_error}")

    async def _call_tool_once(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float,
    ) -> str:
        response = await self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
            timeout=timeout,
        )
        if "error" in response:
            raise MCPToolExecutionError(
                f"MCP 工具调用失败: {response['error']}"
            )

        result = response.get("result", {})
        if result.get("isError"):
            raise MCPToolExecutionError(f"MCP 工具返回错误: {result}")

        content = result.get("content", [])
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))

        if not texts:
            return json.dumps(
                {"success": False, "message": f"MCP 工具 {name} 没有返回文本结果"},
                ensure_ascii=False,
            )
        return "\n".join(texts)

    def _record_tool_result(self, result: str, started: float) -> None:
        latency_ms = (time.perf_counter() - started) * 1000
        self._metrics["tool_calls_total"] += 1
        self._metrics["tool_latency_ms"].append(latency_ms)
        if len(self._metrics["tool_latency_ms"]) > 1000:
            self._metrics["tool_latency_ms"].pop(0)

        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = {}

        if payload.get("success") is True:
            self._metrics["tool_calls_success"] += 1
        else:
            self._metrics["tool_calls_failed"] += 1

    def _record_tool_failure(self, started: float) -> None:
        latency_ms = (time.perf_counter() - started) * 1000
        self._metrics["tool_calls_total"] += 1
        self._metrics["tool_calls_failed"] += 1
        self._metrics["tool_latency_ms"].append(latency_ms)
        if len(self._metrics["tool_latency_ms"]) > 1000:
            self._metrics["tool_latency_ms"].pop(0)

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """发送带 id 的 JSON-RPC 请求，并等待对应响应。"""

        if self.process is None or self.process.stdin is None:
            raise RuntimeError("MCP Server 尚未启动")

        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        await self._write_message(payload)

        try:
            return await asyncio.wait_for(
                future,
                timeout=timeout or self.request_timeout,
            )
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """发送 MCP 通知，不需要等待响应。"""

        await self._write_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    async def _write_message(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("MCP Server stdin 不可用")

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        async with self._write_lock:
            self.process.stdin.write(data)
            await self.process.stdin.drain()

    async def _read_stdout(self) -> None:
        """持续读取 MCP 响应，并按 JSON-RPC id 唤醒等待中的 Future。"""

        if self.process is None or self.process.stdout is None:
            return
        process = self.process

        while True:
            line = await process.stdout.readline()
            if not line:
                break

            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                LOGGER.warning("忽略非 JSON 的 MCP stdout 内容: %r", line)
                continue

            request_id = message.get("id")
            if request_id is None:
                continue

            future = self._pending.get(request_id)
            if future is None or future.done():
                continue

            if "error" in message:
                future.set_exception(RuntimeError(str(message["error"])))
            else:
                future.set_result(message)

        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("MCP Server 已退出"))

    async def _read_stderr(self) -> None:
        """转发 MCP Server 日志；stderr 不参与协议通信。"""

        if self.process is None or self.process.stderr is None:
            return
        process = self.process

        while True:
            line = await process.stderr.readline()
            if not line:
                break
            LOGGER.info("[mcp_server] %s", line.decode("utf-8", errors="replace").rstrip())

def set_mcp_session(session: MCPStdioSession | None) -> None:
    """由 app.py 在启动/关闭时注入同一个 MCP 会话。"""

    global _mcp_session
    _mcp_session = session


def get_mcp_metrics() -> dict[str, Any]:
    """返回当前 MCP 会话的工具调用指标。"""

    if _mcp_session is None:
        return {
            "tool_calls_total": 0,
            "tool_calls_success": 0,
            "tool_calls_failed": 0,
            "tool_call_retries": 0,
            "tool_call_success_rate": 0.0,
            "tool_call_avg_latency_ms": 0.0,
            "tool_call_p95_latency_ms": 0.0,
        }
    return _mcp_session.metrics_snapshot()


def _call_mcp_sync(tool_name: str, arguments: dict[str, Any]) -> str:
    """把异步 MCP 调用包装成同步函数，供受限 exec 命名空间使用。"""

    if _mcp_session is None or _main_loop is None:
        return json.dumps(
            {"success": False, "message": "MCP Server 尚未连接"},
            ensure_ascii=False,
        )

    future = asyncio.run_coroutine_threadsafe(
        _mcp_session.call_tool(tool_name, arguments),
        _main_loop,
    )
    try:
        return future.result(timeout=20)
    except Exception as exc:
        LOGGER.exception("MCP 工具同步调用失败: %s", tool_name)
        return json.dumps(
            {"success": False, "message": f"MCP 工具调用失败: {exc}"},
            ensure_ascii=False,
        )


def _send_command_sync(device_id: str, command: str) -> str:
    return _call_mcp_sync(
        "send_command",
        {"device_id": device_id, "command": command},
    )


def _read_sensor_sync(sensor_id: str) -> str:
    return _call_mcp_sync("read_sensor", {"sensor_id": sensor_id})


def _get_device_status_sync(device_id: str) -> str:
    return _call_mcp_sync("get_device_status", {"device_id": device_id})


def _validate_control_code(code: str) -> ast.Module:
    """限制模型生成的代码只能做一次白名单工具调用。"""

    if len(code) > 500:
        raise ValueError("生成的代码过长")

    tree = ast.parse(code, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("代码必须是一条变量赋值")

    assignment = tree.body[0]
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or assignment.targets[0].id != "result"
    ):
        raise ValueError("代码必须把结果赋值给 result")

    call = assignment.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError("代码只能调用白名单工具函数")
    if call.func.id not in ALLOWED_TOOLS:
        raise ValueError(f"不允许调用函数: {call.func.id}")

    allowed_nodes = (
        ast.Module,
        ast.Assign,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Call,
        ast.Constant,
        ast.keyword,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"代码包含不允许的语法: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("代码不能访问双下划线名称")

    return tree


def _execute_control_code(code: str) -> str:
    """在受限命名空间中执行已通过 AST 校验的代码。"""

    _validate_control_code(code)

    namespace: dict[str, Any] = {
        "send_command": _send_command_sync,
        "read_sensor": _read_sensor_sync,
        "get_device_status": _get_device_status_sync,
    }
    safe_globals = {"__builtins__": {}}
    exec(compile(code, "<agent-generated-control>", "exec"), safe_globals, namespace)

    result = namespace.get("result")
    if not isinstance(result, str):
        raise ValueError("控制代码没有返回 JSON 字符串")

    return result


def _fallback_control_code(device_id: str, command: str) -> str:
    """模型代码不合法时，根据已解析意图生成安全的确定性代码。"""

    return (
        f"result = send_command({json.dumps(device_id, ensure_ascii=False)}, "
        f"{json.dumps(command, ensure_ascii=False)})"
    )


async def _generate_control_code(intent: dict[str, Any]) -> str:
    """让 LLM 生成一条调用 send_command 的 Python 代码。"""

    llm = _get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "你是一个受限代码生成器。只输出一行 Python 代码，"
                    "必须形如：result = send_command(\"fan_01\", \"ON\")。"
                    "禁止 import、open、eval、exec、属性访问、循环、打印或解释。"
                )
            ),
            HumanMessage(
                content=json.dumps(intent, ensure_ascii=False),
            ),
        ]
    )
    return _strip_code_fence(_content_to_text(response.content))


def _normalize_device_id(value: str) -> str:
    """把模型或用户给出的常见设备别名统一到设备 ID。"""

    raw = value.strip()
    normalized = raw.lower()

    if "fan" in normalized or "风扇" in raw or "电扇" in raw:
        return "fan_01"
    if (
        "temp" in normalized
        or "温度" in raw
        or "传感器" in raw
        or "sensor" in normalized
    ):
        return "temp_01"
    return raw


def _rule_based_intent(user_text: str) -> dict[str, Any] | None:
    """对高频中文控制表达做确定性兜底，降低模型波动。"""

    text = user_text.strip()
    lowered = text.lower()
    has_fan = any(token in text for token in ("风扇", "电扇")) or "fan" in lowered
    has_temp = any(token in text for token in ("温度", "多少度", "℃")) or "temp" in lowered
    if not has_fan and not has_temp:
        return None

    conditional = any(
        token in text
        for token in (
            "如果",
            "当温度",
            "温度高于",
            "温度低于",
            "温度超过",
            "温度小于",
            "温度大于",
        )
    )

    if has_fan:
        close_tokens = (
            "关闭",
            "关掉",
            "关了",
            "关一下",
            "停止",
            "停下来",
            "调到off",
            "off",
        )
        open_tokens = (
            "打开",
            "开一下",
            "开启",
            "转起来",
            "调到on",
            "on",
        )
        query_tokens = (
            "状态",
            "开着",
            "关了吗",
            "开还是关",
            "运行状态",
            "开着没有",
        )

        if any(token in text for token in query_tokens):
            return {
                "action": "query",
                "target_device": "fan_01",
                "command": "",
                "condition": "",
            }
        if any(token in lowered for token in close_tokens):
            return {
                "action": "control",
                "target_device": "fan_01",
                "command": "OFF",
                "condition": text if conditional else "",
            }
        if any(token in lowered for token in open_tokens):
            return {
                "action": "control",
                "target_device": "fan_01",
                "command": "ON",
                "condition": text if conditional else "",
            }

    if has_temp:
        return {
            "action": "query",
            "target_device": "temp_01",
            "command": "",
            "condition": "",
        }

    return None


def _extract_temperature_condition(condition: str) -> tuple[str, float] | None:
    """从“温度超过 28 度”这类条件中提取比较符和阈值。"""

    if not condition:
        return None

    greater_match = re.search(
        r"(?:超过|高于|大于|>=|>)\s*(\d+(?:\.\d+)?)",
        condition,
    )
    if greater_match:
        return ">", float(greater_match.group(1))

    less_match = re.search(
        r"(?:低于|小于|<=|<)\s*(\d+(?:\.\d+)?)",
        condition,
    )
    if less_match:
        return "<", float(less_match.group(1))

    return None


async def parse_intent(state: AgentState) -> dict[str, Any]:
    """节点 1：规则优先解析高频指令，复杂表达再交给 LLM。"""

    user_text = ""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            user_text = _content_to_text(message.content)
            break

    rule_intent = _rule_based_intent(user_text)
    if rule_intent is not None:
        LOGGER.info("规则解析意图: %s", rule_intent)
        return {"intent": rule_intent}

    llm = _get_llm()
    history = state.get("messages", [])[-8:]
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=INTENT_PROMPT), *history],
            response_format={"type": "json_object"},
        )
        parsed = _parse_json_object(_content_to_text(response.content))
    except Exception as exc:
        LOGGER.warning("LLM 意图解析失败，降级为澄清请求: %s", exc)
        parsed = {
            "action": "query",
            "target_device": "",
            "command": "",
            "condition": "我暂时无法确定设备或操作，请补充设备名称和动作。",
        }

    action = str(parsed.get("action", "query")).strip().lower()
    if action not in {"query", "control"}:
        action = "query"

    target_device = _normalize_device_id(
        str(parsed.get("target_device", ""))
    )
    command = str(parsed.get("command", "")).strip().upper()
    condition = str(parsed.get("condition", "")).strip()

    if action == "query":
        command = ""
    if action == "control" and command not in {"ON", "OFF"}:
        command = ""

    intent = {
        "action": action,
        "target_device": target_device,
        "command": command,
        "condition": condition,
    }
    LOGGER.info("解析意图: %s", intent)
    return {"intent": intent}


def route_after_intent(state: AgentState) -> str:
    """条件边：查询走 query_node，控制走 codegen_node。"""

    if state.get("intent", {}).get("action") == "control":
        return "codegen_node"
    return "query_node"


async def query_node(state: AgentState) -> dict[str, Any]:
    """节点 2A：根据设备类型调用 read_sensor 或 get_device_status。"""

    intent = state.get("intent", {})
    target_device = str(intent.get("target_device", "")).strip()

    if not target_device:
        return {
            "tool_result": json.dumps(
                {
                    "success": False,
                    "message": intent.get("condition")
                    or "请说明要查询哪一台设备，例如温度传感器或风扇",
                },
                ensure_ascii=False,
            )
        }

    try:
        if target_device.startswith("temp") or target_device.endswith("sensor"):
            result = await _mcp_session.call_tool(  # type: ignore[union-attr]
                "read_sensor",
                {"sensor_id": target_device},
            )
        else:
            result = await _mcp_session.call_tool(  # type: ignore[union-attr]
                "get_device_status",
                {"device_id": target_device},
            )
    except Exception as exc:
        LOGGER.exception("查询工具调用失败")
        result = json.dumps(
            {"success": False, "message": f"查询失败: {exc}"},
            ensure_ascii=False,
        )

    return {"tool_result": result}


async def codegen_node(state: AgentState) -> dict[str, Any]:
    """节点 2B：执行条件判断，生成控制代码并在受限命名空间中执行。"""

    intent = state.get("intent", {})
    target_device = str(intent.get("target_device", "")).strip()
    command = str(intent.get("command", "")).strip().upper()
    condition = str(intent.get("condition", "")).strip()

    if not target_device or command not in {"ON", "OFF"}:
        return {
            "tool_result": json.dumps(
                {
                    "success": False,
                    "message": condition
                    or "请明确要控制的设备和 ON/OFF 指令",
                },
                ensure_ascii=False,
            )
        }

    condition_context: dict[str, Any] | None = None
    threshold = _extract_temperature_condition(condition)

    # 支持“如果温度超过 28 度就打开风扇”这类单条件自动化。
    if threshold is not None:
        operator, threshold_value = threshold
        if _mcp_session is None:
            return {
                "tool_result": json.dumps(
                    {"success": False, "message": "MCP Server 尚未连接"},
                    ensure_ascii=False,
                )
            }

        try:
            sensor_result = await _mcp_session.call_tool(
                "read_sensor",
                {"sensor_id": "temp_01"},
            )
            sensor_payload = json.loads(sensor_result)
            if sensor_payload.get("success") is False:
                return {
                    "tool_result": json.dumps(
                        {
                            "success": False,
                            "message": sensor_payload.get(
                                "message", "读取温度失败"
                            ),
                        },
                        ensure_ascii=False,
                    )
                }

            current_value = float(sensor_payload["value"])
        except Exception as exc:
            LOGGER.exception("条件控制读取温度失败")
            return {
                "tool_result": json.dumps(
                    {"success": False, "message": f"读取温度失败: {exc}"},
                    ensure_ascii=False,
                )
            }

        condition_met = (
            current_value > threshold_value
            if operator == ">"
            else current_value < threshold_value
        )
        condition_context = {
            "type": "sensor_threshold",
            "sensor_id": "temp_01",
            "operator": operator,
            "threshold": threshold_value,
            "current_value": current_value,
            "matched": condition_met,
        }

        if not condition_met:
            comparison = "未超过" if operator == ">" else "未低于"
            return {
                "tool_result": json.dumps(
                    {
                        "success": True,
                        "executed": False,
                        "device_id": target_device,
                        "command": command,
                        "message": (
                            f"当前温度 {current_value}°C，{comparison} "
                            f"{threshold_value}°C，未执行 {command} 指令"
                        ),
                        "condition": condition_context,
                    },
                    ensure_ascii=False,
                )
            }

    try:
        generated_code = await _generate_control_code(intent)
        try:
            _validate_control_code(generated_code)
        except (SyntaxError, ValueError) as exc:
            LOGGER.warning("模型生成代码不合法，使用安全兜底: %s", exc)
            generated_code = _fallback_control_code(target_device, command)

        result = await asyncio.to_thread(_execute_control_code, generated_code)

        if condition_context is not None:
            try:
                result_payload = json.loads(result)
                if isinstance(result_payload, dict):
                    result_payload["executed"] = result_payload.get(
                        "success", False
                    )
                    result_payload["condition"] = condition_context
                    result = json.dumps(result_payload, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                LOGGER.warning("无法把条件上下文合并进工具结果")
    except Exception as exc:
        LOGGER.exception("控制代码执行失败")
        result = json.dumps(
            {"success": False, "message": f"设备控制失败: {exc}"},
            ensure_ascii=False,
        )

    return {"tool_result": result}


async def respond_node(state: AgentState) -> dict[str, Any]:
    """节点 3：把工具结果转换为自然语言，并写入对话记忆。"""

    tool_result = state.get("tool_result", "")
    messages = state.get("messages", [])
    user_message = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            user_message = _content_to_text(message.content)
            break

    try:
        parsed_result = json.loads(tool_result)
    except json.JSONDecodeError:
        parsed_result = {"success": True, "result": tool_result}

    if parsed_result.get("success") is False:
        fallback = str(parsed_result.get("message", "操作失败，请稍后重试"))
    else:
        fallback = json.dumps(parsed_result, ensure_ascii=False)

    writer = get_stream_writer()
    llm_messages = [
        SystemMessage(content=RESPOND_PROMPT),
        HumanMessage(
            content=(
                f"用户问题：{user_message}\n"
                f"工具结果：{json.dumps(parsed_result, ensure_ascii=False)}\n"
                "请生成最终回答。"
            )
        ),
    ]

    chunks: list[str] = []
    if _streaming_mode.get():
        try:
            llm = _get_llm()
            async for chunk in llm.astream(llm_messages):
                text = _content_to_text(chunk.content)
                if text:
                    chunks.append(text)
                    writer({"type": "token", "content": text})

            final_answer = "".join(chunks).strip()
            if not final_answer:
                raise RuntimeError("流式回复为空")
        except Exception:
            if chunks:
                LOGGER.warning("流式回复中断，使用已生成的片段")
                final_answer = "".join(chunks).strip()
            else:
                LOGGER.exception("流式回复失败，回退到非流式调用")
                try:
                    response = await _get_llm().ainvoke(llm_messages)
                    final_answer = _content_to_text(response.content).strip()
                except Exception:
                    LOGGER.exception("回复生成失败，使用工具结果兜底")
                    final_answer = fallback
    else:
        try:
            response = await _get_llm().ainvoke(llm_messages)
            final_answer = _content_to_text(response.content).strip()
        except Exception:
            LOGGER.exception("回复生成失败，使用工具结果兜底")
            final_answer = fallback

    if not final_answer:
        final_answer = fallback

    # 兼容不支持流式的 OpenAI 兼容服务：至少向前端发送一个完整文本事件。
    if _streaming_mode.get() and not chunks:
        writer({"type": "token", "content": final_answer})

    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
    }


def build_graph() -> StateGraph:
    """构建并返回带内存检查点的 LangGraph。"""

    builder = StateGraph(AgentState)
    builder.add_node("parse_intent", parse_intent)
    builder.add_node("query_node", query_node)
    builder.add_node("codegen_node", codegen_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "parse_intent")
    builder.add_conditional_edges(
        "parse_intent",
        route_after_intent,
        {
            "query_node": "query_node",
            "codegen_node": "codegen_node",
        },
    )
    builder.add_edge("query_node", "respond")
    builder.add_edge("codegen_node", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=MemorySaver())


agent_graph = build_graph()


def _agent_input(message: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=message)],
        "intent": {},
        "tool_result": "",
        "final_answer": "",
    }


async def _invoke_agent(message: str, thread_id: str) -> dict[str, Any]:
    """执行底层 LangGraph，并返回完整 State。"""

    global _main_loop

    _main_loop = asyncio.get_running_loop()

    if not message.strip():
        raise ValueError("message 不能为空")
    if not thread_id.strip():
        raise ValueError("thread_id 不能为空")
    if _mcp_session is None:
        raise RuntimeError("MCP Server 尚未连接，请检查 app.py 启动日志")

    return await agent_graph.ainvoke(
        _agent_input(message),
        config={"configurable": {"thread_id": thread_id}},
    )


def _build_trace(result: dict[str, Any]) -> dict[str, Any]:
    """从 Graph State 中提取评测所需的意图、工具结果与执行状态。"""

    intent = result.get("intent", {})
    tool_result_raw = result.get("tool_result", "")
    try:
        tool_result = json.loads(tool_result_raw)
    except (json.JSONDecodeError, TypeError):
        tool_result = {"success": False, "raw": tool_result_raw}

    return {
        "intent": intent,
        "tool_result": tool_result,
        "tool_success": bool(tool_result.get("success", False)),
        "final_answer": result.get("final_answer", ""),
    }


async def run_agent_with_trace(
    message: str,
    thread_id: str,
) -> tuple[str, dict[str, Any]]:
    """执行一轮 Agent，同时返回回答和结构化 trace。"""

    result = await _invoke_agent(message, thread_id)
    reply = result.get("final_answer") or "Agent 没有生成有效回答。"
    return reply, _build_trace(result)


async def run_agent(message: str, thread_id: str) -> str:
    """FastAPI 调用入口：执行一轮 Agent，并返回最终自然语言答案。"""

    reply, _ = await run_agent_with_trace(message, thread_id)
    return reply


async def run_agent_stream(
    message: str,
    thread_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """执行 Agent 并输出可映射为 SSE 的阶段、Token 和完成事件。"""

    global _main_loop

    _main_loop = asyncio.get_running_loop()

    if not message.strip():
        yield {"type": "error", "message": "请输入要查询或控制的设备指令。"}
        return
    if not thread_id.strip():
        yield {"type": "error", "message": "thread_id 不能为空"}
        return
    if _mcp_session is None:
        yield {
            "type": "error",
            "message": "MCP Server 尚未连接，请检查 app.py 启动日志。",
        }
        return

    token = _streaming_mode.set(True)
    final_reply = ""
    try:
        yield {"type": "status", "stage": "parsing", "message": "正在解析意图"}
        async for item in agent_graph.astream(
            _agent_input(message),
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["custom", "updates"],
        ):
            if isinstance(item, tuple) and len(item) == 2:
                mode, data = item
            else:
                mode, data = "updates", item

            if mode == "custom":
                if isinstance(data, dict):
                    yield data
                continue

            if not isinstance(data, dict):
                continue

            if "parse_intent" in data:
                intent = data["parse_intent"].get("intent", {})
                yield {
                    "type": "intent",
                    "intent": intent,
                    "message": "意图解析完成",
                }

            for node_name in ("query_node", "codegen_node"):
                if node_name in data:
                    raw_tool_result = data[node_name].get("tool_result", "")
                    try:
                        tool_result = json.loads(raw_tool_result)
                    except (json.JSONDecodeError, TypeError):
                        tool_result = {
                            "success": False,
                            "raw": raw_tool_result,
                        }
                    yield {
                        "type": "tool",
                        "node": node_name,
                        "success": bool(tool_result.get("success", False)),
                        "result": tool_result,
                        "message": "工具执行完成",
                    }

            if "respond" in data:
                final_reply = str(
                    data["respond"].get("final_answer", final_reply)
                )
    except Exception as exc:
        LOGGER.exception("Agent 流式执行失败")
        yield {"type": "error", "message": f"Agent 执行失败: {exc}"}
        return
    finally:
        _streaming_mode.reset(token)

    yield {"type": "done", "reply": final_reply}








