# run_pose_auto.py
from datetime import datetime, timezone
import time
import cv2
from ultralytics import YOLO
from config import Config
from vision_pose import PoseEstimator
from vision_face import FaceRecognizer
from mqtt_client import MQTTClient
from notifier import Notifier

def main():
    last_heartbeat = 0.0  # 하트비트 주기 관리

    # MQTT 연결
    mqtt_client = MQTTClient(Config.MQTT_HOST, Config.MQTT_PORT,
                             Config.USERNAME, Config.PASSWORD,
                             Config.KEEPALIVE, Config.DEVICE_ID)
    mqtt_client.connect()
    notifier = Notifier(mqtt_client)

    # 모듈 로드
    pose_estimator = PoseEstimator(Config.POSE_MODEL, Config.CONF_THRESHOLD,
                                   Config.IMG_SIZE, Config.YOLO_DEVICE)
    face_recognizer = FaceRecognizer(Config.FACES_DIR, Config.FACES_DB,
                                     Config.REBUILD_FACES_DB, Config.FACE_THR,
                                     Config.BLUR_THR, Config.CTX_ID)

    # 사람 감지용 가벼운 YOLO 모델
    person_detector = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(Config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.HEIGHT)

    active = False       # 현재 활성화 상태 (사람 있을 때 True)
    last_seen = 0.0      # 마지막 사람 감지 시각
    ACTIVE_TIMEOUT = 5.0 # 5초 동안 사람 없으면 비활성화

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # 1️⃣ 사람 존재 감지 (가벼운 YOLO로 빠르게)
            det = person_detector(frame, classes=[0], conf=0.5, verbose=False)
            has_person = len(det[0].boxes) > 0

            # 사람이 보이면 -> 활성화 상태 진입
            if has_person:
                last_seen = time.time()
                if not active:
                    print("[SYSTEM] 사람 인식됨 → 활성화 시작")
                    active = True

                # Pose + Face 인식
                res = pose_estimator.detect(frame)
                has_boxes = (getattr(res, "boxes", None) is not None) and (res.boxes.xyxy is not None)
                has_kps = (getattr(res, "keypoints", None) is not None)

                if has_boxes and has_kps:
                    kps = res.keypoints.xy.cpu().numpy()
                    boxes = res.boxes.xyxy.cpu().numpy()
                    n_pose = int(res.boxes.shape[0])
                else:
                    n_pose = 0

                user_verified = False
                if n_pose > 0:
                    # 얼굴 인식 (여러 얼굴 중 하나라도 등록된 사용자 있으면 True)
                    results = face_recognizer.recognize(frame)
                    matched = sum(1 for uid, sim, bbox in results if uid)
                    user_verified = matched > 0

                    # 각 사람에 대해 자세 분석
                    for i, box in enumerate(boxes):
                        action = pose_estimator.classify_action(kps[i], box)
                        notifier.check_and_notify(action, user_verified)

                # 시각화 (optional)
                frame_vis = res.plot()
                cv2.imshow("Pose+Face", frame_vis)

            # 사람이 일정 시간 안 보이면 → 비활성화
            elif active and (time.time() - last_seen) > ACTIVE_TIMEOUT:
                print("[SYSTEM] 5초 이상 사람 없음 → 비활성화 전환")
                active = False
                notifier.check_and_notify("Unknown", user_verified=False)

            # 기본 대기 화면 (사람 없을 때)
            elif not active:
                cv2.putText(frame, "Idle - No Person Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Pose+Face", frame)

            # 하트비트 주기적으로 전송
            if (time.time() - last_heartbeat) >= Config.HEARTBEAT_INTERVAL:
                hb = {
                    "iso": datetime.now(timezone.utc).isoformat() + "(UTC)",
                    "Power": True
                }
                mqtt_client.publish(Config.HEARTBEAT_TOPIC, hb)
                last_heartbeat = time.time()

            # 종료 조건
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    finally:
        hb = {
            "iso": datetime.now(timezone.utc).isoformat() + "(UTC)",
            "Power": False
        }
        mqtt_client.publish(Config.HEARTBEAT_TOPIC, hb)
        cap.release()
        cv2.destroyAllWindows()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()
