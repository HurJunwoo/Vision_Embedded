# notifier.py
from datetime import datetime, timezone
import time
from config import Config

class Notifier:
    def __init__(self, mqtt_client):
        self.mqtt_client = mqtt_client 
        self.topic_alert = f"pose/{Config.DEVICE_ID}/alert"  # 알림 토픽
        
        # 내부 상태 추적용 변수
        self.current_state = None              # 현재 상태 ("allowed" / "other")
        self.state_start_time = None           # 상태가 유지되기 시작한 시각
        self.last_published_state = None       # 마지막으로 publish된 상태 (True/False)
        self.allowed_duration = 10.0           # 허용 자세 연속 유지 기준 (초)
        self.disallowed_duration = 5.0         # 비허용 자세 연속 유지 기준 (초)
        self.grace_period = 3.0                # 허용 자세 중단 허용 시간 (초)
        self.last_allowed_time = None          # 마지막으로 허용 자세로 인식된 시각

    def check_and_notify(self, action, user_verified):
        now = time.time()
        is_allowed = (action in Config.ALLOWED_ACTIONS)

        # --- 허용 자세로 인식될 때 ---
        if is_allowed:
            # 마지막 허용 자세 시각 갱신
            self.last_allowed_time = now

            # 현재 상태가 바뀌었다면 초기화
            if self.current_state != "allowed":
                self.current_state = "allowed"
                self.state_start_time = now

        # --- 비허용 자세로 인식될 때 ---
        else:
            # 허용 자세에서 벗어났지만, grace_period 이내면 유지로 간주
            if self.last_allowed_time and (now - self.last_allowed_time) <= self.grace_period:
                # 여전히 허용 자세 유지 중으로 간주
                pass
            else:
                # grace_period 초과 시 실제로 비허용 상태로 전환
                if self.current_state != "other":
                    self.current_state = "other"
                    self.state_start_time = now

        # 아직 타이머가 없으면 초기화
        if self.state_start_time is None:
            self.state_start_time = now

        elapsed = now - self.state_start_time

        # 허용된 자세가 10초 연속 유지
        if self.current_state == "allowed" and elapsed >= self.allowed_duration:
            if self.last_published_state != True:
                msg = {
                    "iso": datetime.now(timezone.utc).isoformat(),
                    "user_verified": (user_verified is not None),
                    "pose_ok": True
                }
                self.mqtt_client.publish(self.topic_alert, str(msg))
                print("[NOTIFY - TRUE]", msg)
                self.last_published_state = True

        # 비허용 자세가 5초 연속 유지
        elif self.current_state == "other" and elapsed >= self.disallowed_duration:
            if self.last_published_state != False:
                msg = {
                    "iso": datetime.now(timezone.utc).isoformat(),
                    "user_verified": (user_verified is not None),
                    "pose_ok": False
                }
                self.mqtt_client.publish(self.topic_alert, str(msg))
                print("[NOTIFY - FALSE]", msg)
                self.last_published_state = False
