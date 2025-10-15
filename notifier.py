# notifier.py
from datetime import datetime, timezone
from config import Config

class Notifier:
    def __init__(self, mqtt_client):
        self.mqtt_client = mqtt_client 
        self.topic_alert = f"pose/{Config.DEVICE_ID}/alert" # 알림 토픽

    def check_and_notify(self, action, user_verified):
        if action in Config.ALLOWED_ACTIONS: # 허용된 자세인지 확인
            msg = {
                "iso": datetime.now(timezone.utc).isoformat(),  # 항상 ISO 시간 추가
                "user_verified": (user_verified is not None)  # 얼굴 있으면 True, 없으면 False
            }
            self.mqtt_client.publish(self.topic_alert, msg)
            print("[NOTIFY - FAIL]", msg) # 배포 시 주석 처리 권장

    

