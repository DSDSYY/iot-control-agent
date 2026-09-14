"""IoT Control MCP Server。

该文件既是 MQTT 数据缓存服务，也是暴露给 LangGraph Agent 的 MCP 工具层。
它通过 stdio transport 与 Agent 通信，因此：

- 日志只能写 stderr，不能写 stdout；
- stdout 必须保留给 MCP JSON-RPC 协议；
- 启动时使用 connect_async + loop_start，即使 Broker 暂时未启动，
  MCP Server 也可以先启动，后续 Broker 恢复后会自动重连。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from fastmcp import FastMCP


load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

TEMP_TOPIC = "home/sensor/temp"
FAN_COMMAND_TOPIC = "home/actuator/fan"
FAN_STATUS_TOPIC = "home/actuator/fan/status"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("mcp-server")

mcp = FastMCP("IoT Control MCP")

# MCP 工具可能在 asyncio 线程中调用，MQTT 回调则在 paho 线程中执行，
# 因此共享状态必须加锁。
state_lock = threading.RLock()
latest_sensor: dict[str, dict[str, Any]] = {}
device_status: dict[str, str] = {"fan_01": "OFF"}

mqtt_client: mqtt.Client | None = None


def json_result(payload: dict[str, Any]) -> str:
    """统一输出 JSON 字符串，方便 LLM 和测试程序解析。"""

    return json.dumps(payload, ensure_ascii=False)


def is_connection_failed(reason_code: object) -> bool:
    return bool(getattr(reason_code, "is_failure", reason_code != 0))


def on_connect(
    client: mqtt.Client,
    userdata: object,
    flags: mqtt.ConnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None = None,
) -> None:
    """连接成功后订阅传感器和风扇状态 topic。"""

    if is_connection_failed(reason_code):
        LOGGER.error("MQTT 连接失败: %s", reason_code)
        return

    client.subscribe(
        [
            (TEMP_TOPIC, 0),
            (FAN_STATUS_TOPIC, 1),
        ]
    )
    LOGGER.info("MQTT 已连接，订阅: %s, %s", TEMP_TOPIC, FAN_STATUS_TOPIC)


def on_message(
    client: mqtt.Client,
    userdata: object,
    message: mqtt.MQTTMessage,
) -> None:
    """把 MQTT 消息转换为内存中的最新设备快照。"""

    try:
        payload = json.loads(message.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        LOGGER.warning("忽略非法 MQTT JSON: topic=%s error=%s", message.topic, exc)
        return

    with state_lock:
        if message.topic == TEMP_TOPIC:
            sensor_id = str(payload.get("device_id", "temp_01"))
            latest_sensor[sensor_id] = {
                "value": payload.get("value"),
                "ts": payload.get("ts", int(time.time() * 1000)),
            }
        elif message.topic == FAN_STATUS_TOPIC:
            device_id = str(payload.get("device_id", "fan_01"))
            status = str(payload.get("status", "UNKNOWN")).upper()
            device_status[device_id] = status
        else:
            return

    LOGGER.info("状态更新: topic=%s payload=%s", message.topic, payload)


def on_disconnect(
    client: mqtt.Client,
    userdata: object,
    disconnect_flags: mqtt.DisconnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None = None,
) -> None:
    LOGGER.warning("MQTT 连接断开，等待自动重连: %s", reason_code)


def start_mqtt() -> None:
    """启动 MQTT 客户端和后台网络循环。"""

    global mqtt_client

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"mcp-server-{os.getpid()}",
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    mqtt_client = client
    client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()
    LOGGER.info("MQTT 客户端已启动，Broker=%s:%s", MQTT_BROKER, MQTT_PORT)


def stop_mqtt() -> None:
    """停止 MQTT 客户端，尽量不做长时间阻塞。"""

    if mqtt_client is None:
        return

    try:
        mqtt_client.disconnect()
    except Exception:
        pass
    try:
        mqtt_client.loop_stop()
    except Exception:
        pass


@mcp.tool
def get_device_status(device_id: str) -> str:
    """查询设备当前状态，例如 get_device_status("fan_01")。"""

    normalized_id = device_id.strip()
    with state_lock:
        status = device_status.get(normalized_id)

    if status is None:
        return json_result(
            {
                "success": False,
                "device_id": normalized_id,
                "message": f"未找到设备 {normalized_id} 的状态，请确认设备是否已启动",
            }
        )

    return json_result(
        {
            "success": True,
            "device_id": normalized_id,
            "status": status,
        }
    )


@mcp.tool
def read_sensor(sensor_id: str) -> str:
    """读取最近一次传感器数据，例如 read_sensor("temp_01")。"""

    normalized_id = sensor_id.strip()
    with state_lock:
        reading = latest_sensor.get(normalized_id)

    if reading is None:
        return json_result(
            {
                "success": False,
                "sensor_id": normalized_id,
                "message": f"还没有收到 {normalized_id} 的数据，请确认模拟器正在运行",
            }
        )

    return json_result(
        {
            "success": True,
            "sensor_id": normalized_id,
            "value": reading.get("value"),
            "ts": reading.get("ts"),
        }
    )


@mcp.tool
def send_command(device_id: str, command: str) -> str:
    """向执行器发送控制指令，目前支持 device_id=fan_01, command=ON/OFF。"""

    normalized_id = device_id.strip()
    normalized_command = command.strip().upper()

    if normalized_id != "fan_01":
        return json_result(
            {
                "success": False,
                "device_id": normalized_id,
                "command": normalized_command,
                "message": f"暂不支持控制设备 {normalized_id}",
            }
        )

    if normalized_command not in {"ON", "OFF"}:
        return json_result(
            {
                "success": False,
                "device_id": normalized_id,
                "command": normalized_command,
                "message": "command 只支持 ON 或 OFF",
            }
        )

    if mqtt_client is None:
        return json_result(
            {
                "success": False,
                "device_id": normalized_id,
                "command": normalized_command,
                "message": "MQTT 客户端尚未初始化",
            }
        )

    payload = {"command": normalized_command}
    info = mqtt_client.publish(
        FAN_COMMAND_TOPIC,
        json.dumps(payload, ensure_ascii=False),
        qos=1,
        retain=False,
    )
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        return json_result(
            {
                "success": False,
                "device_id": normalized_id,
                "command": normalized_command,
                "message": f"MQTT 发布失败，rc={info.rc}",
            }
        )

    try:
        info.wait_for_publish(timeout=3)
    except (RuntimeError, ValueError) as exc:
        return json_result(
            {
                "success": False,
                "device_id": normalized_id,
                "command": normalized_command,
                "message": f"MQTT 发布确认失败: {exc}",
            }
        )

    # 乐观更新一份状态；虚拟风扇随后会发布状态 topic 进行权威确认。
    with state_lock:
        device_status[normalized_id] = normalized_command

    return json_result(
        {
            "success": True,
            "device_id": normalized_id,
            "command": normalized_command,
            "message": "控制指令已发布",
        }
    )


if __name__ == "__main__":
    start_mqtt()
    try:
        mcp.run(transport="stdio")
    finally:
        stop_mqtt()
