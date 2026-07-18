import paho.mqtt.client as mqtt

HOST = "127.0.0.1"
PORT = 1884
TOPIC = "logibridge/#"

def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[UPLINK MONITOR] Connected: {reason_code}")
    client.subscribe(TOPIC, qos=1)
    print(f"[UPLINK MONITOR] Subscribed to {TOPIC}")

def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8", errors="replace")
    print(f"{message.topic} {payload}")

client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="logiedge-uplink-monitor",
)

client.on_connect = on_connect
client.on_message = on_message

client.connect(HOST, PORT, keepalive=60)
client.loop_forever()
