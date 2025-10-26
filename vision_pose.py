# vision_pose.py
import math
from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", conf=0.5, imgsz=416, device="cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def classify_action(self, kps_xy, bbox):
        def angle_with_vertical(a, b):
            vx, vy = b[0] - a[0], b[1] - a[1]
            mag = math.hypot(vx, vy)
            if mag == 0:
                return 0.0
            # 화면 좌표계(y 아래로 증가) 기준으로, 0=수직, 90=수평
            cosang = max(min(vy / mag, 1.0), -1.0)
            return math.degrees(math.acos(cosang))

        # 키포인트 확보
        try:
            l_sh, r_sh = kps_xy[5], kps_xy[6]
            l_hp, r_hp = kps_xy[11], kps_xy[12]
            l_kn, r_kn = kps_xy[13], kps_xy[14]
        except Exception:
            return "Unknown"

        # 중심점 계산
        mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        mid_hp = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)
        mid_kn = ((l_kn[0] + r_kn[0]) / 2, (l_kn[1] + r_kn[1]) / 2)

        # 박스 치수
        x1, y1, x2, y2 = bbox
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        eps = max(8.0, 0.02 * bh)  # 노이즈 보정용 마진

        # 특징량 계산
        torso_angle = angle_with_vertical(mid_sh, mid_hp)     # 0(수직)~90(수평)
        horizontalish_box = (bw > bh * 1.25)

        # 자세 분류

        # Sitting (무릎-엉덩이, 상체비율, 얼굴비율)
        try:
            if not (mid_hp[1] + eps < mid_kn[1]):
                return "Sitting"
        except Exception:
            pass

        # 무릎/엉덩이 누락 시, 상체 비율 기반 보정
        upper_ratio = (mid_sh[1] - y1) / bh
        if upper_ratio > 0.6:
            return "Sitting"

        # 얼굴 근접 구도 보정
        face_ratio = (l_sh[1] - y1) / bh
        if face_ratio > 0.45:
            return "Sitting"

        # Standing
        if torso_angle <= 75 and not horizontalish_box:
            return "Standing"

        # Lying
        if torso_angle > 75 or horizontalish_box:
            return "Lying"

        # 예외
        return "Standing"

    def detect(self, frame):
        results = self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device
        )
        return results[0]
