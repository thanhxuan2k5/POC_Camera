import json
import logging
import time
import threading
from typing import Any, Callable, Dict, Optional
import paho.mqtt.client as mqtt
import numpy as np

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MQTTClient:
    def __init__(self, broker: str, port: int, username: str = '', password: str = '', client_id: str = 'wood_inspector'):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id
        self._connected = False
        self._should_reconnect = True
        
        # Determine MQTT callback API version based on paho-mqtt version
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        except AttributeError:
            self.client = mqtt.Client(client_id=self.client_id)
            
        if self.username:
            self.client.username_pw_set(self.username, self.password)
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        self._callbacks: Dict[str, list[Callable]] = {}
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _on_connect(self, client, userdata, flags, reason_code, *args):
        # Handle different signatures of on_connect in paho-mqtt
        if reason_code == 0:
            logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
            with self._lock:
                self._connected = True
                for topic in self._callbacks:
                    self.client.subscribe(topic)
        else:
            logger.error(f"Failed to connect to MQTT broker, reason_code: {reason_code}")
            self._connected = False

    def _on_disconnect(self, client, userdata, *args):
        reason_code = args[-2] if len(args) >= 2 else (args[0] if len(args) > 0 else "Unknown")
        logger.warning(f"Disconnected from MQTT broker. Reason code: {reason_code}")
        with self._lock:
            self._connected = False
        
        if self._should_reconnect:
            self._reconnect()

    def _reconnect(self):
        delay = 1
        max_delay = 60
        while self._should_reconnect and not self._connected:
            try:
                logger.info(f"Attempting to reconnect to MQTT broker in {delay} seconds...")
                time.sleep(delay)
                self.client.reconnect()
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                delay = min(delay * 2, max_delay)

    def _on_message(self, client, userdata, message):
        topic = message.topic
        payload = message.payload.decode('utf-8')
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = payload
            
        with self._lock:
            callbacks = self._callbacks.get(topic, [])
            # Make a copy to iterate to allow modification of callbacks list during iteration
            callbacks_copy = list(callbacks)
            
        for callback in callbacks_copy:
            try:
                callback(topic, data)
            except Exception as e:
                logger.error(f"Error in MQTT callback for topic {topic}: {e}", exc_info=True)

    def connect(self):
        self._should_reconnect = True
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            # Start reconnect loop in a separate thread so it doesn't block connect() caller
            threading.Thread(target=self._reconnect, daemon=True).start()

    def disconnect(self):
        self._should_reconnect = False
        self.client.loop_stop()
        self.client.disconnect()
        with self._lock:
            self._connected = False
        logger.info("Disconnected from MQTT broker.")

    def publish(self, topic: str, payload_dict: Dict[str, Any], qos: int = 1):
        if not self._connected:
            logger.debug(f"Cannot publish to {topic}, MQTT not connected.")
            return False
            
        try:
            payload = json.dumps(payload_dict, cls=NumpyEncoder)
            result = self.client.publish(topic, payload, qos=qos)
            result.wait_for_publish()
            return result.is_published()
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            return False

    def publish_result(self, camera_id: str, track_id: int, result: str, confidence: float, timestamp: float):
        topic = f"conveyor/camera/{camera_id}/result"
        payload = {
            "camera_id": camera_id,
            "track_id": track_id,
            "result": result,
            "confidence": confidence,
            "timestamp": timestamp
        }
        return self.publish(topic, payload)

    def publish_status(self, camera_id: str, status: str):
        topic = f"conveyor/camera/{camera_id}/status"
        payload = {
            "camera_id": camera_id,
            "status": status,
            "timestamp": time.time()
        }
        return self.publish(topic, payload)

    def publish_signal(self, result: str):
        topic = "conveyor/signal/output"
        payload = {
            "signal": result,  # 'OK' or 'NG'
            "timestamp": time.time()
        }
        return self.publish(topic, payload)

    def subscribe(self, topic: str, callback: Callable):
        with self._lock:
            if topic not in self._callbacks:
                self._callbacks[topic] = []
                if self._connected:
                    self.client.subscribe(topic)
            if callback not in self._callbacks[topic]:
                self._callbacks[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Optional[Callable] = None):
        with self._lock:
            if topic in self._callbacks:
                if callback:
                    if callback in self._callbacks[topic]:
                        self._callbacks[topic].remove(callback)
                    if not self._callbacks[topic]:
                        del self._callbacks[topic]
                        if self._connected:
                            self.client.unsubscribe(topic)
                else:
                    del self._callbacks[topic]
                    if self._connected:
                        self.client.unsubscribe(topic)
