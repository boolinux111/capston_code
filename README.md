# 🎥 Object Identification Pipeline

**사망 인식 기반의 영상 분석 자동화 파이프라인**

본 프로젝트는 입력된 영상으로부터 **식 단위로 분석**하여, **객체를 검출하고 사망을 시달**하며, 최종적으로 **시각화된 영상**을 생성하는 End-to-End 파이프라인입니다.

---

## 📌 주요 기능

* ✂️ **식 분해** – 모션 기반 + Adaptive SceneDetect
* 🢍 **객체 검출** – YOLO11 기반 객체 인식 (사망/비사망)
* 🚀 **객체 트랙크키링** – DeepSort 기반 ID 부여
* 🧬 **사망 시달** – 여론(Facenet512) + 전신(OSNet/PCB) + 위치정보
* 🖼️ **결과 시각화** – 프레임별 ID를 영상으로 저장

---

## 💪 설치

### Conda 환경 설정

```bash
conda create -n objectid python=3.10 -y
conda activate objectid
```

### 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### RTX 5090 / CUDA 12.0 이상 사용자용 PyTorch 설치 (Nightly)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 🚀 실행 순서

### ① 식 분해

```bash
cd object_identification-main
python Scene_distribution.py
```

* 입력: `D.P..mp4`
* 출력: `output_scenes/Scene_1.mp4`, `Scene_2.mp4` 등

---

### ② 객체 검출

```bash
python detect_video.py \
  --video output_scenes/Scene_1.mp4 \
  --output_dir detect_vid_output
```

* `detect_vid_output/frames/`: 출출된 프레임 이미지
* `detect_vid_output/content/detections.json`: YOLO 객체 검출 결과

---

### ③ 객체 트랙 + 사망 시달

```bash
python object_identification.py \
  --base_dir "../object_identification-main" \
  --frame_set_name "detect_vid_output" \
  --output_video "final_result.mp4"
```

* `output_pipeline/`: 프레임 별 결과
* `output_faces/`: 여론 크롭 저장
* `final_result.mp4`: 최종 영상

---

## 💡 참고 사항

* **여론 인식**: DeepFace (Facenet512 + MTCNN)
* **전신 임베딩**: OSNet (PCB 기반)
* **위치정보**: 암커 객체(의자/책상 등)과의 거리 기반 보조 판단
* **성능 최적화**: 프레임 스탭, 위치 정규화, 유사도 기반 PID 관리

---

## 🪖 라이센스

MIT License © 2025
자유로운 사용 및 수정 가능합니다.

---

