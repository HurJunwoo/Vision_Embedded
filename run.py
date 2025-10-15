from datetime import datetime, timezone
import time
import cv2
from config import Config
from vision_pose import PoseEstimator
from vision_face import FaceRecognizer
from mqtt_client import MQTTClient
from notifier import Notifier

def main():
    last_heartbeat = 0.0 # 전원 확인 interval 초기화
    t0 = time.time()
    frames = 0

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

    cap = cv2.VideoCapture(Config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.HEIGHT)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            res = pose_estimator.detect(frame)
            # ★ 결과 그리기(박스/키포인트)
            frame_vis = res.plot()
            has_boxes = (getattr(res, "boxes", None) is not None) and (res.boxes.xyxy is not None)
            has_kps   = (getattr(res, "keypoints", None) is not None)
            n_pose = int(res.boxes.shape[0]) if has_boxes else 0

            action = "Unknown"

            # 얼굴 인식 (여러 얼굴 중 하나라도 등록된 사용자 있으면 True)
            results = face_recognizer.recognize(frame)
            # 프레임에 얼굴 박스/라벨 표시
            matched = 0
            for uid, sim, bbox in results:
                if bbox:
                    x1, y1, x2, y2 = bbox
                    color = (0, 255, 0) if uid else (0, 128, 255)
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{uid if uid else 'Unknown'} ({sim:.2f})"
                    cv2.putText(frame_vis, label, (x1, max(y1-8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                if uid:
                    matched += 1
            user_verified = matched > 0

            # 포즈 인식
            if has_boxes and has_kps:
                kps = res.keypoints.xy.cpu().numpy()  # (n,17,2)
                boxes = res.boxes.xyxy.cpu().numpy()   # (n,4)
                for i, box in enumerate(boxes):
                    action = pose_estimator.classify_action(kps[i], box, pose_estimator.history[(i,0)])
                    # 박스 상단에 행동 라벨 표기
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.putText(frame_vis, action, (x1, y1-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                    # 허용된 자세면 알림
                    notifier.check_and_notify(action, user_verified)

            cv2.imshow("Pose+Face", frame)

            # 간단 OSD
            frames += 1
            if frames % 10 == 0:
                dt = time.time() - t0
                fps = frames / dt if dt > 0 else 0.0
            else:
                fps = None
            if fps:
                cv2.putText(frame_vis, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame_vis, f"Pose: {n_pose}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame_vis, f"Face: {matched}/{len(results)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Pose+Face", frame_vis)

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            if (time.time() - last_heartbeat) >= Config.HEARTBEAT_INTERVAL:
                hb = {
                    "iso": datetime.now(timezone.utc).isoformat() + "(UTC)", # 항상 ISO 시간 추가
                    "Power": True # 전원 알림
                }
                mqtt_client.publish(Config.HEARTBEAT_TOPIC, hb)
                last_heartbeat = time.time()

    finally:
        hb = {
            "iso": datetime.now(timezone.utc).isoformat() + "(UTC)", # 항상 ISO 시간 추가
            "Power": False # 종료 알림
        }
        mqtt_client.publish(Config.HEARTBEAT_TOPIC, hb)

        cap.release()
        cv2.destroyAllWindows()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()
