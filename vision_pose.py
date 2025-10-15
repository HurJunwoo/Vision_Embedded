# vision_pose.py
import math
from collections import deque, defaultdict
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", conf=0.5, imgsz=416, device="cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.history = defaultdict(lambda: {
            "ankle_dx": deque(maxlen=5),
            "torso_angle": deque(maxlen=5),
            "aspect": deque(maxlen=5)
        })

    def classify_action(self, kps_xy, bbox, hist_deque):
        # --- 자세 분류 로직 (원본 코드 그대로 이식) ---
        def angle_with_vertical(a, b):
            vx, vy = b[0] - a[0], b[1] - a[1]
            mag = math.hypot(vx, vy)
            if mag == 0:
                return 0.0
            # 화면 좌표계(y 아래로 증가)에서 '수직' 기준 기울기 각도(0=수직, 90=수평)
            cosang = max(min(vy / mag, 1.0), -1.0)
            return math.degrees(math.acos(cosang))

            # 필수 keypoints 확보 실패 시
        try:
            l_sh, r_sh = kps_xy[5],  kps_xy[6]
            l_wr, r_wr = kps_xy[9],  kps_xy[10]
            l_hp, r_hp = kps_xy[11], kps_xy[12]
            l_kn, r_kn = kps_xy[13], kps_xy[14]
            l_an, r_an = kps_xy[15], kps_xy[16]
        except Exception:
            return "Unknown"

        # 중심점/치수
        mid_sh = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
        mid_hp = ((l_hp[0]+r_hp[0])/2, (l_hp[1]+r_hp[1])/2)
        mid_kn = ((l_kn[0]+r_kn[0])/2, (l_kn[1]+r_kn[1])/2)

        x1, y1, x2, y2 = bbox
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        eps = max(8.0, 0.02 * bh)  # 키포인트 튐 보정 마진

        # 핵심 특징량
        torso_angle = angle_with_vertical(mid_sh, mid_hp)     # 0(수직) ~ 90(수평)
        horizontalish_box = (bw > bh * 1.25)                  # 가로로 긴 박스
        hands_up = (l_wr[1] + eps < l_sh[1]) and (r_wr[1] + eps < r_sh[1])
        hip_above_knee = (mid_hp[1] + eps < mid_kn[1])        # 엉덩이가 무릎보다 위면 True

        # 발목 간격 변동(걷기 판단용)
        ankle_dx = abs(l_an[0] - r_an[0]) / bw
        hist_deque["ankle_dx"].append(ankle_dx)
        if len(hist_deque["ankle_dx"]) > 5:
            hist_deque["ankle_dx"].popleft()
        ankle_var = max(hist_deque["ankle_dx"]) - min(hist_deque["ankle_dx"])

        # --- 판정 규칙(8월 버전) ---
        if torso_angle > 55 or horizontalish_box:
            return "Falling/Lying"
        if hands_up:
            return "HandsUp"
        if not hip_above_knee:
            return "Sitting"
        if ankle_var > 0.10:
            return "Walking?"
        return "Standing"

    def detect(self, frame):
        results = self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device
        )
        return results[0]
