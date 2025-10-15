# mqtt_client.py
import json
import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(self, host, port=1883, username=None, password=None, keepalive=30, device_id="raspi-01"):
        self.device_id = device_id
        self.client = mqtt.Client(client_id=f"pose-face-{device_id}") # 클라이언트 ID 설정
        if username:
            self.client.username_pw_set(username, password) # 사용자명/비밀번호 설정(인증 미사용 시 None)
        self.host = host
        self.port = port
        self.keepalive = keepalive

    # MQTT 브로커에 연결
    def connect(self):
        self.client.connect(self.host, self.port, self.keepalive)
        self.client.loop_start() # 네트워크 루프 시작

    # MQTT 브로커와 연결 해제
    def disconnect(self):
        self.client.loop_stop() # 네트워크 루프 중지
        self.client.disconnect() # 브로커와 연결 해제

    def publish(self, topic, obj, qos=0, retain=False): # obj: 전송할 데이터(dict), qos = 0(max 1회), retain = False(브로커 저장 안함)
        payload = json.dumps(obj, ensure_ascii=False) # dict -> JSON 문자열 변환
        self.client.publish(topic, payload=payload, qos=qos, retain=retain) # 메시지 발행
