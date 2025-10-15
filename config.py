# config.py
class Config:
    DEVICE_ID = "raspi-01" # 디바이스 ID

    # MQTT 설정
    MQTT_HOST = "127.0.0.1" # MQTT 브로커 주소
    MQTT_PORT = 1883 
    USERNAME = None # MQTT 사용자명
    PASSWORD = None # MQTT 비밀번호
    KEEPALIVE = 30 # Client가 Broker에 연결을 유지하는 최대 시간(초)

    # 비전 설정
    POSE_MODEL = "yolov8n-pose.pt" # 포즈 모델 경로 (n = nano (가장 가벼움), s = small, m = medium, l = large)
    YOLO_DEVICE = "cpu" # Topst D3 보드도 CPU로 동작 -> cpu 유지
    IMG_SIZE = 416 # (320 ~ 416 권장) 숫자 클수록 느리지만 정확도 향상
    CONF_THRESHOLD = 0.5 # 객체 탐지 신뢰도 임계값 (0.0 ~ 1.0) ex. 0.5 = 50% 이상일 때만 탐지로 간주
    CAM_INDEX = 0 # 카메라 장치 인덱스 (0 = 기본 카메라)
    WIDTH = 640 # 해상도(640x480 권장) -> 느리면 320으로 변경
    HEIGHT = 480 # 해상도(640x480 권장) -> 느리면 240으로 변경
    PIXEL_FORMAT = "auto" # 색상 포멧 (auto 권장, mpeg, yuyv, gray 등) -> 종류에 따라 cpu 부하 변화

    # 얼굴 인식 설정
    FACES_DIR = "faces" # 등록된 얼굴 이미지 폴더(경로: faces/사용자명/이미지.jpg)
    FACES_DB = "faces_db.npz" # 얼굴 임베딩 데이터베이스 파일(첫 실행 시 faces 폴더에서 생성됨)
    REBUILD_FACES_DB = False # True로 설정 시 매 실행 시 faces 폴더에서 DB 재생성 
                             # (새로운 사용자 추가 시 True로 변경 후 실행 권장, 그 외에는 False 유지)
    FACE_INTERVAL = 2.0 # 얼굴 인식 간격 (초)
    FACE_THR = 0.4 # 얼굴 인식 임계값 (0.0 ~ 1.0) 높을수록 엄격 (0.35 ~ 0.5 권장, 보안 강화 시 0.6 이상 권장)
    BLUR_THR = 20.0 # 얼굴 인식 시 블러 임계값(높을수록 엄격), 낮을수록 흐린 이미지도 허용
    CTX_ID = -1  # CPU = -1 / GPU = 0 (Topst D3 보드는 CPU만 지원)

    # 허용된 자세 (여기에 지정된 자세 + 얼굴 매칭 성공 시 알림 전송)
    ALLOWED_ACTIONS = ["Sitting"] # "Standing", "HandsUp", "Sitting", "Walking?", "Falling/Lying"

    HEARTBEAT_INTERVAL = 5.0   # 전원 확인 주기 (초 단위)
    HEARTBEAT_TOPIC = f"pose/{DEVICE_ID}/heartbeat" # 전원 확인 토픽(pose/raspi-01/heartbeat)
