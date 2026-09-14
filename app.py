"""FastAPI HTTP 入口。

启动时创建 MCPStdioSession 子进程运行 mcp_server.py，并把同一个会话注入
agent_graph。应用提供普通对话、SSE 流式对话、健康检查和工具指标接口。
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import agent_graph


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("iot-control-agent")
BASE_DIR = Path(__file__).resolve().parent


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户自然语言指令")
    thread_id: str = Field(..., min_length=1, description="多轮会话 ID")
    debug: bool = Field(False, description="是否返回意图和工具调用 trace")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """管理 MCP 子进程的生命周期。"""

    mcp_session = agent_graph.MCPStdioSession(BASE_DIR / "mcp_server.py")
    try:
        await mcp_session.start()
        agent_graph.set_mcp_session(mcp_session)
    except Exception:
        LOGGER.exception("MCP Server 启动失败，/chat 将返回友好错误")
        agent_graph.set_mcp_session(None)
        await mcp_session.stop()
        mcp_session = None  # type: ignore[assignment]

    try:
        yield
    finally:
        agent_graph.set_mcp_session(None)
        if mcp_session is not None:
            await mcp_session.stop()


app = FastAPI(
    title="IoT Natural Language Control Agent",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict[str, object]:
    """返回进程内 MCP 工具调用指标。"""

    return agent_graph.get_mcp_metrics()


@app.post("/chat", response_model=None)
async def chat(request: ChatRequest):
    """执行一轮自然语言设备查询或控制。"""

    try:
        reply, trace = await agent_graph.run_agent_with_trace(
            message=request.message,
            thread_id=request.thread_id,
        )
        payload: dict[str, object] = {"reply": reply}
        if request.debug:
            payload["trace"] = trace
        return payload
    except Exception:
        LOGGER.exception("处理 /chat 请求失败")
        return JSONResponse(
            status_code=500,
            content={
                "reply": (
                    "抱歉，Agent 暂时无法处理这个请求。"
                    "请检查 .env 的模型配置、MQTT Broker 和 app.py 启动日志。"
                )
            },
        )


@app.post("/chat/stream", response_model=None)
async def chat_stream(request: ChatRequest):
    """以 Server-Sent Events 输出阶段事件、Token 和最终回答。"""

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in agent_graph.run_agent_stream(
                message=request.message,
                thread_id=request.thread_id,
            ):
                event_type = str(event.get("type", "message"))
                data = json.dumps(event, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"
        except Exception:
            LOGGER.exception("处理 /chat/stream 请求失败")
            error = json.dumps(
                {
                    "type": "error",
                    "message": "Agent 流式处理失败，请检查服务日志。",
                },
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {error}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
