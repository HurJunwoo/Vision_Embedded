import paho.mqtt.client as mqtt
import json
from datetime import datetime

BROKER_HOST = "192.168.0.216"   # ← 여기를 라즈베리파이 IP로 바꿔주세요!
BROKER_PORT = 1883
TOPICS = [("pose/raspi-01/heartbeat", 0),
          ("pose/raspi-01/alert", 0)]

LOG_FILE = "mqtt_server_log.csv"

# CSV 헤더 보장
with open(LOG_FILE, "a", encoding="utf-8") as f:
    if f.tell() == 0:
        f.write("local_time,topic,payload\n")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT] Connected OK")
        for t, q in TOPICS:
            client.subscribe(t, qos=q)
            print(f"[MQTT] SUB {t}")
    else:
        print("[MQTT][ERROR] Connect failed:", rc)

def on_message(client, userdata, msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        payload_str = json.dumps(payload, ensure_ascii=False)
    except Exception:
        payload_str = msg.payload.decode("utf-8", errors="replace")

    print(f"[MSG] {msg.topic} -> {payload_str}")

    # CSV 기록
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        safe = payload_str.replace('"', '""')
        f.write(f'{now},{msg.topic},"{safe}"\n')

def main():
    client = mqtt.Client(client_id="pc-subscriber")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    print(f"[RUN] Broker={BROKER_HOST}:{BROKER_PORT}, Topics={[t for t, _ in TOPICS]}")
    client.loop_forever()

if __name__ == "__main__":
    main()
