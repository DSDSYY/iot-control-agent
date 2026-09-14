"""MQTT 虚拟设备模拟器。

运行后会在本机 MQTT Broker 上模拟：
1. temp_01: 每 3 秒发布一次 20~35 度的温度数据；
2. fan_01: 订阅控制指令，并把最新状态发布到状态 topic。

新增设备时，可以仿照 TemperatureSensor / SmartFan：
- 继承 threading.Thread；
- 在 run() 中维护自己的 MQTT Client；
- 通过 stop() 释放连接。
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import uuid

import paho.mqtt.client as mqtt
from dotenv import load_dotenv


load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

TEMP_TOPIC = "home/sensor/temp"
FAN_COMMAND_TOPIC = "home/actuator/fan"
FAN_STATUS_TOPIC = "home/actuator/fan/status"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("device-simulator")

# 全局风扇状态，后续如果接入真实设备，可放到 Redis/设备影子中。
fan_state = "OFF"
fan_state_lock = threading.Lock()
stop_event = threading.Event()


def make_client_id(prefix: str) -> str:
    """生成随机 client_id，避免多个模拟器实例互相踢下线。"""

    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def is_connection_failed(reason_code: object) -> bool:
    """兼容 paho-mqtt 2.x 的 ReasonCode 和普通整数返回码。"""

    return bool(getattr(reason_code, "is_failure", reason_code != 0))


class TemperatureSensor(threading.Thread):
    """模拟一个周期性上报温度的温度传感器。"""

    def __init__(self, device_id: str = "temp_01") -> None:
        super().__init__(name=device_id, daemon=True)
        self.device_id = device_id
        self._client: mqtt.Client | None = None

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None = None,
    ) -> None:
        if is_connection_failed(reason_code):
            LOGGER.error("[%s] MQTT 连接失败: %s", self.device_id, reason_code)
            return
        LOGGER.info("[%s] 已连接 MQTT Broker", self.device_id)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: object,
        disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None = None,
    ) -> None:
        if not stop_event.is_set():
            LOGGER.warning("[%s] MQTT 连接断开: %s", self.device_id, reason_code)

    def run(self) -> None:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=make_client_id("sim-temp"),
        )
        self._client = client
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect

        while not stop_event.is_set():
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                break
            except OSError as exc:
                LOGGER.warning(
                    "[%s] 无法连接 Broker %s:%s，3 秒后重试: %s",
                    self.device_id,
                    MQTT_BROKER,
                    MQTT_PORT,
                    exc,
                )
                stop_event.wait(3)

        if stop_event.is_set():
            return

        client.loop_start()
        try:
            while not stop_event.is_set():
                value = round(random.uniform(20.0, 35.0), 1)
                payload = {
                    "device_id": self.device_id,
                    "value": value,
                    "ts": int(time.time() * 1000),
                }
                info = client.publish(
                    TEMP_TOPIC,
                    json.dumps(payload, ensure_ascii=False),
                    qos=0,
                    retain=False,
                )
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    LOGGER.error("[%s] 温度发布失败: rc=%s", self.device_id, info.rc)
                else:
                    LOGGER.info("[%s] 发布温度 %.1f°C", self.device_id, value)

                stop_event.wait(3)
        finally:
            client.loop_stop()
            client.disconnect()

    def stop(self) -> None:
        """主动断开连接，保证 Ctrl+C 时可以优雅退出。"""

        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass


class SmartFan(threading.Thread):
    """模拟一个订阅控制命令、发布状态变化的智能风扇。"""

    def __init__(self, device_id: str = "fan_01") -> None:
        super().__init__(name=device_id, daemon=True)
        self.device_id = device_id
        self._client: mqtt.Client | None = None

    def _publish_status(self, client: mqtt.Client) -> None:
        with fan_state_lock:
            current_state = fan_state

        payload = {
            "device_id": self.device_id,
            "status": current_state,
            "ts": int(time.time() * 1000),
        }
        client.publish(
            FAN_STATUS_TOPIC,
            json.dumps(payload, ensure_ascii=False),
            qos=1,
            retain=True,
        )

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None = None,
    ) -> None:
        if is_connection_failed(reason_code):
            LOGGER.error("[%s] MQTT 连接失败: %s", self.device_id, reason_code)
            return

        client.subscribe(FAN_COMMAND_TOPIC, qos=1)
        self._publish_status(client)
        LOGGER.info(
            "[%s] 已连接并订阅 %s，当前状态=%s",
            self.device_id,
            FAN_COMMAND_TOPIC,
            fan_state,
        )

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: object,
        message: mqtt.MQTTMessage,
    ) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("[%s] 收到非法 JSON: %s", self.device_id, exc)
            return

        command = str(payload.get("command", "")).strip().upper()
        if command not in {"ON", "OFF"}:
            LOGGER.warning("[%s] 不支持的命令: %r", self.device_id, command)
            return

        global fan_state
        with fan_state_lock:
            fan_state = command

        print(f"[fan_01] 状态变为 {command}", flush=True)
        self._publish_status(client)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: object,
        disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None = None,
    ) -> None:
        if not stop_event.is_set():
            LOGGER.warning("[%s] MQTT 连接断开: %s", self.device_id, reason_code)

    def run(self) -> None:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=make_client_id("sim-fan"),
        )
        self._client = client
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect

        while not stop_event.is_set():
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                break
            except OSError as exc:
                LOGGER.warning(
                    "[%s] 无法连接 Broker %s:%s，3 秒后重试: %s",
                    self.device_id,
                    MQTT_BROKER,
                    MQTT_PORT,
                    exc,
                )
                stop_event.wait(3)

        if stop_event.is_set():
            return

        try:
            client.loop_forever()
        finally:
            client.disconnect()

    def stop(self) -> None:
        """从主线程关闭 paho 网络循环。"""

        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass


def main() -> None:
    devices = [TemperatureSensor(), SmartFan()]

    LOGGER.info("启动虚拟设备，Broker=%s:%s", MQTT_BROKER, MQTT_PORT)
    for device in devices:
        device.start()

    try:
        while any(device.is_alive() for device in devices):
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("收到 Ctrl+C，正在停止虚拟设备...")
    finally:
        stop_event.set()
        for device in devices:
            device.stop()
        for device in devices:
            device.join(timeout=5)
        LOGGER.info("虚拟设备已退出")


if __name__ == "__main__":
    main()
