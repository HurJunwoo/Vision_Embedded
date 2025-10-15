# vision_face.py
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceRecognizer:
    def __init__(self, faces_dir="faces", faces_db="faces_db.npz",
                 rebuild=False, thr=0.4, blur_thr=20.0, ctx_id=-1):
        self.faces_dir = faces_dir
        self.faces_db = faces_db
        self.face_thr = thr
        self.blur_thr = blur_thr
        self.face_last_t = 0.0

        self.app = FaceAnalysis(name="buffalo_l") # buffalo_l: 비교적 정확도가 높은 경량 모델
        self.app.prepare(ctx_id=ctx_id, det_size=(416, 416)) # det_size: 얼굴 탐지 크기 (클수록 정확도 향상, 느려짐), CPU 기반은 (320,320) ~ (416,416) 권장
        self.user_names, self.user_embs = [], None

        if rebuild or not os.path.isfile(self.faces_db):
             print("[Face] Building faces DB from folder:", self.faces_dir)
             self._build_faces_db() # faces 폴더에서 DB 생성(없으면)

        if os.path.isfile(self.faces_db): # 있으면 DB 로드
            db = np.load(self.faces_db, allow_pickle=True)
            self.user_names = db["names"].tolist()
            self.user_embs = db["embs"].astype(np.float32)
            print(f"[Face] Loaded DB: {len(self.user_names)} users")
        else:
            print("[Face][WARN] Faces DB not found. Recognition will be disabled.")

    # 얼굴 DB 생성(등록된 사용자 폴더에서 얼굴 임베딩 추출 후 평균화)
    def _build_faces_db(self):
        names, embs = [], []
        if not os.path.isdir(self.faces_dir): # faces_dir 없으면 종료
            return
        for uid in sorted(os.listdir(self.faces_dir)): # 사용자 폴더 반복(폴더만 대상)
            udir = os.path.join(self.faces_dir, uid)
            if not os.path.isdir(udir):
                continue
            vecs = [] # 임베딩 벡터 리스트
            for fn in os.listdir(udir): # 파일 내 이미지 파일 반복
                if not fn.lower().endswith((".jpg", ".png")): continue # .jpg, .png 만 처리
                img = cv2.imread(os.path.join(udir, fn))
                if img is None: continue
                faces = self.app.get(img)
                if not faces: continue
                emb = faces[0].normed_embedding # 첫 번째 얼굴 임베딩 벡터(가장 큰 얼굴만 사용하여 학습)
                vecs.append(emb)
            if vecs: # 임베딩 벡터가 있으면 평균화 후 DB에 추가
                mean_emb = np.mean(np.stack(vecs), axis=0) # 평균 벡터 계산(노이즈 제거 목적)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb)+1e-9) # 정규화(길이 1로)
                names.append(uid)
                embs.append(mean_emb.astype(np.float32))
        if embs:
            np.savez(self.faces_db, names=np.array(names), embs=np.stack(embs)) # DB 저장(이후 빠르게 로드 가능)

    def recognize(self, frame):
        if self.user_embs is None: # DB 없으면 빈 리스트 반환(에러 방지)
            return []

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백 사진으로 변환
        if cv2.Laplacian(g, cv2.CV_64F).var() < self.blur_thr: # 선명도 검사
            return []  # 흐리면 아예 빈 리스트 반환

        faces = self.app.get(frame) # 프레임에서 탐지된 모든 얼굴 정보
        if not faces:
            return []  # 얼굴이 없으면 빈 리스트 반환

        results = [] # 결과 리스트 초기화(한 프레임에 여러 얼굴 탐지 시 (user_id, confidence) 튜플 계속 추가)
        for f in faces: # 탐지된 얼굴 반복(DB와 비교하여 사용자 인식)
            emb = f.normed_embedding # 탐지된 얼굴 임베딩 벡터
            sims = [np.dot(emb, u) / (np.linalg.norm(emb) * np.linalg.norm(u) + 1e-9) for u in self.user_embs] # 코사인 유사도 계산
            k = int(np.argmax(sims)) # 가장 유사한 사용자 인덱스 추출
            uid = self.user_names[k] if sims[k] >= self.face_thr else None
            # bbox 꺼내기 (float → int)
            if hasattr(f, "bbox") and f.bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                bbox = (x1, y1, x2, y2)
            else:
                bbox = None
            results.append((uid, float(sims[k]), bbox))  # 임계값 미달 → unknown

        return results # (user_id, confidence) 튜플 리스트 반환(user_id는 None 가능)

