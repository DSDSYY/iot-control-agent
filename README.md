# IoT Control Agent

一个面向智能家居场景的 IoT 自然语言控制 Agent。用户可以用自然语言查询温度、查询风扇状态、控制风扇，也可以表达“如果温度超过 28 度就打开风扇”这类条件自动化指令。

项目采用 **FastAPI + LangGraph + FastMCP + MQTT** 分层：

- LangGraph 负责意图解析、条件分支、多轮记忆和结果生成；
- FastMCP 负责把设备能力标准化为工具；
- MQTT 负责传感器上行数据与执行器控制指令；
- FastAPI 负责对外提供 HTTP 接口和托管 MCP 子进程。

## 一、架构图

```text
┌──────────────────────┐
│       用户 / 前端      │
└──────────┬───────────┘
           │ POST /chat
           v
┌──────────────────────┐
│  FastAPI app.py       │
│  - /chat              │
│  - /health            │
│  - 管理 MCP 子进程     │
└──────────┬───────────┘
           │ run_agent(message, thread_id)
           v
┌──────────────────────────────────────────────┐
│  LangGraph agent_graph.py                    │
│                                              │
│  parse_intent                                │
│      │                                       │
│      ├── action=query  ──> query_node         │
│      │                    ├─ read_sensor      │
│      │                    └─ get_device_status│
│      │                                       │
│      └── action=control ─> codegen_node       │
│                           ├─ 条件判断          │
│                           ├─ LLM 生成受限代码   │
│                           ├─ AST 白名单校验     │
│                           └─ exec -> send_command│
│                                              │
│  respond_node -> 自然语言回答 + MemorySaver    │
└──────────┬───────────────────────────────────┘
           │ MCP JSON-RPC over stdio
           v
┌──────────────────────┐
│  FastMCP mcp_server.py│
│  - get_device_status  │
│  - read_sensor        │
│  - send_command       │
└──────────┬───────────┘
           │ MQTT publish/subscribe
           v
┌──────────────────────────────────────────────┐
│              Mosquitto Broker                 │
│                                              │
│  home/sensor/temp                             │
│  home/actuator/fan                            │
│  home/actuator/fan/status                     │
└──────────┬───────────────────────────────────┘
           │
           v
┌──────────────────────┐
│ device_simulator.py  │
│ - temp_01 温度传感器  │
│ - fan_01 智能风扇     │
└──────────────────────┘
```

## 二、目录结构

```text
iot-control-agent/
├── device_simulator.py     # MQTT 虚拟设备：温度传感器 + 智能风扇
├── mcp_server.py           # FastMCP 工具层，订阅/发布 MQTT topic
├── agent_graph.py          # LangGraph 意图解析、条件控制、记忆与安全执行
├── app.py                  # FastAPI 入口，管理 MCP 子进程
├── test_client.py          # 4 条自然语言指令的联调测试脚本
├── test_stream_client.py   # SSE 流式输出测试
├── test_concurrency.py     # 并发联调测试
├── evaluate_agent.py       # 意图/工具/任务成功率和延迟评测
├── eval_cases.json         # 40 条自然语言评测集
├── Dockerfile
├── docker-compose.yml
├── mosquitto/
│   └── mosquitto.conf
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量示例
└── README.md
```

## 三、环境要求

- Python 3.10+，推荐 Python 3.11 或 3.13；
- Mosquitto 2.x；
- 支持 OpenAI 兼容接口的模型服务，例如 DeepSeek。

> Windows 下如果 `python --version` 是 3.8，请显式使用 3.11/3.13 的 Python 解释器创建虚拟环境。

## 四、安装依赖

```powershell
D:\conda\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

复制并编辑环境变量：

```powershell
Copy-Item .env.example .env
```

```dotenv
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
MQTT_BROKER=localhost
MQTT_PORT=1883
```

`OPENAI_API_KEY` 必须替换成真实值。模型只会在调用 `/chat` 时请求，不会在服务启动时请求。

## 五、启动 Mosquitto

### Windows 本地安装

```powershell
& "C:\Program Files\mosquitto\mosquitto.exe" -v
```

使用自定义配置文件：

```powershell
& "C:\Program Files\mosquitto\mosquitto.exe" -c "C:\Program Files\mosquitto\mosquitto.conf" -v
```

### Docker

```powershell
docker run --rm -it --name mosquitto -p 1883:1883 eclipse-mosquitto:2 mosquitto -v
```

验证 Broker：

```powershell
mosquitto_sub -h localhost -p 1883 -t "home/#" -v
```

## 六、启动顺序

### 1. 启动 MQTT Broker

先启动 Mosquitto，确认 `localhost:1883` 可以连接。

### 2. 启动虚拟设备

打开终端 1：

```powershell
python device_simulator.py
```

预期输出：

```text
[temp_01] 发布温度 26.5°C
[fan_01] 已连接并订阅 home/actuator/fan，当前状态=OFF
```

### 3. 启动 Agent 服务

打开终端 2：

```powershell
python app.py
```

`app.py` 会自动创建并管理 `mcp_server.py` 子进程，因此正常使用时不需要单独启动 MCP Server。

服务地址：

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/chat`

### 4. 单独调试 MCP Server（可选）

只有需要单独检查 MCP 协议时才执行：

```powershell
python mcp_server.py
```

`mcp_server.py` 使用 stdio transport，启动后不会提供 HTTP 端口，也不会持续输出普通业务日志。stdout 必须保留给 MCP JSON-RPC，日志统一写入 stderr。

## 七、测试用例

`test_client.py` 会依次发送以下 4 条消息，并统计关键词命中率和平均响应延迟：

| 序号 | 用户输入 | 预期行为 |
|---|---|---|
| 1 | 现在温度多少？ | 返回 `temp_01` 最近一次温度 |
| 2 | 打开风扇 | 调用 `send_command("fan_01", "ON")`，输出 `[fan_01] 状态变为 ON` |
| 3 | 如果温度超过28度就打开风扇 | 先读取当前温度：超过 28°C 才打开风扇，否则返回“未超过阈值，未执行” |
| 4 | 风扇现在什么状态？ | 返回 `fan_01` 当前状态 ON/OFF |

### 一键联调

```powershell
python test_client.py
```

脚本输出示例：

```text
[1/4] 查询温度
用户: 现在温度多少？
Agent: 当前温度约为 26.5°C。

[2/4] 打开风扇
用户: 打开风扇
Agent: 已打开风扇。

[3/4] 温度阈值自动化
用户: 如果温度超过28度就打开风扇
Agent: 当前温度 26.5°C，未超过 28°C，风扇未打开。

[4/4] 查询风扇状态
用户: 风扇现在什么状态？
Agent: 风扇当前状态为 OFF。

联调汇总
关键词命中: 4/4 (100.0%)
平均响应延迟: 0.210s
```

### API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/metrics` | MCP 工具成功率、平均延迟、P95 延迟 |
| POST | `/chat` | 普通对话；`debug=true` 时返回意图与工具 trace |
| POST | `/chat/stream` | SSE 流式输出：status、intent、tool、token、done |

### 流式输出测试

```powershell
python test_stream_client.py
```

### 40 条评测集

```powershell
python evaluate_agent.py --threshold 0.95
```

评测脚本会输出意图准确率、工具成功率、任务完成率、平均延迟和 P95 延迟，并写入 `evaluation_report.json`。

### 并发测试

```powershell
python test_concurrency.py --requests 50 --concurrency 20
```

### Docker 一键启动

```powershell
Copy-Item .env.example .env
# 编辑 .env，填入真实 OPENAI_API_KEY
docker compose up --build
```

### 本地实测结果（2026-09-15）

本次联调环境：Python 3.13.9、FastMCP 4.0.3、LangGraph 1.2.11、Python AMQTT Broker、Mock OpenAI 兼容模型服务。

| 指标 | 真实 DeepSeek 实测 | 说明 |
|---|---:|---|
| 自然语言评测集 | 40/40 通过 | 意图准确率 100%，任务完成率 100% |
| 基础联调平均响应延迟 | 1.393s | 4 条温度/控制/条件/状态用例 |
| 40 条评测平均响应延迟 | 1.191s | P95 为 1.736s |
| MCP 工具调用成功率 | 100% | 141/141 次调用成功，0 次失败 |
| MCP 工具平均调用延迟 | 4.8ms | P95 为 11.8ms |
| SSE 首 Token 延迟 | 702.7ms | 真实 deepseek-chat 流式输出 |
| 并发联调 | 50/50 成功 | 20 并发，18.3 req/s |
| 并发平均延迟 | 931.7ms | P95 为 1.208s |
| 条件控制分支 | 已覆盖 | 条件不成立时正确拦截，条件成立时正确控制 |
| 工具失败重试 | 已验证 | 超时场景触发指数退避重试，并正确记录失败指标 |

> 评测模型为 `deepseek-chat`，测试服务地址为 `https://api.deepseek.com/v1`。不同时间、网络与模型负载下延迟会有波动。

### 手动测试

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

发送对话：

```powershell
$body = @{
  message = "打开风扇"
  thread_id = "demo-1"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json" `
  -Body $body
```

同一 `thread_id` 会复用 `MemorySaver` 中的多轮对话状态；不同用户应使用不同 `thread_id`。

## 八、MCP Tools

| 工具 | 参数 | 说明 |
|---|---|---|
| `get_device_status` | `device_id` | 查询 `fan_01` 等设备状态 |
| `read_sensor` | `sensor_id` | 读取 `temp_01` 最近一次温度 |
| `send_command` | `device_id`, `command` | 控制 `fan_01`，支持 `ON`/`OFF` |

所有工具都返回 JSON 字符串，并且包含 `success` 字段。

## 九、条件控制说明

当前实现支持一条温度阈值条件：

- “如果温度超过 28 度就打开风扇”
- “如果温度低于 20 度就关闭风扇”

执行顺序为：

```text
解析条件 -> read_sensor("temp_01") -> 比较阈值
    ├─ 条件成立 -> send_command(...)
    └─ 条件不成立 -> 返回 executed=false
```

复杂条件编排，例如“工作日下午 6 点后且有人在家时打开风扇”，可以在后续版本中扩展为规则节点或持久化自动化任务。

## 十、代码安全边界

控制节点不会直接执行任意模型代码。`agent_graph.py` 会：

1. 只允许生成一条 `result = <allowed_tool>(...)` 赋值语句；
2. 只允许 `send_command`、`read_sensor`、`get_device_status` 三个函数；
3. 禁止 `import`、`open`、`eval`、`exec`、属性访问、循环和双下划线名称；
4. 清空 `__builtins__`；
5. 代码不合法时回退到由意图解析结果生成的确定性工具调用。

生产环境仍建议改成结构化 Function Calling / MCP 工具调用，不建议执行模型生成代码。

## 十一、常见问题

### 1. MQTT 连接失败

- 确认 Mosquitto 正在监听 `localhost:1883`；
- 确认 `.env` 中 `MQTT_BROKER`、`MQTT_PORT` 正确；
- 确认防火墙没有阻止 1883 端口；
- `mcp_server.py` 使用后台重连，Broker 恢复后无需重启 MCP Server。

### 2. `/chat` 返回模型配置错误

确认 `.env` 中 `OPENAI_API_KEY` 已替换，并且 `OPENAI_BASE_URL` 与模型服务兼容。

### 3. 查询不到温度

确认 `device_simulator.py` 已启动，并且终端中出现温度发布日志。模拟器每 3 秒才发布一次数据，启动后需要等待几秒。

### 4. 条件控制没有打开风扇

这是预期行为之一。只有当当前温度严格大于阈值时才执行控制。检查 Agent 回复中的 `current_value` 和 `threshold`。

### 5. MCP Server 启动失败

直接在终端运行：

```powershell
python mcp_server.py
```

如果没有 JSON-RPC 输入，它会保持等待，这是 stdio transport 的正常表现。重点检查 stdout 是否被日志污染。

### 6. test_client.py 请求失败

确认：

- `python app.py` 已启动；
- `GET /health` 返回 `{"status":"ok"}`；
- 模型 API Key 有效；
- Mosquitto 与 `device_simulator.py` 正在运行；
- 如有需要，通过环境变量修改服务地址：

```powershell
$env:AGENT_BASE_URL = "http://127.0.0.1:8000"
python test_client.py
```




