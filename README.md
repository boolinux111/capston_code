# 🎥 Object Identification Pipeline

이 프로젝트는 다음의 전체 파이프라인을 제공합니다:

1. **씬(장면) 분할** – SceneDetect + 모션 기반
2. **YOLOv8 객체 검출** – 사람 vs 비사람 (의자, 책상 등)
3. **객체 트래킹** – DeepSort 기반 ID 부여
4. **사람 식별** – 얼굴(Facenet/DeepFace) + 전신(PCB/OSNet) + 위치정보 결합
5. **결과 시각화** – 프레임별 ID 표시 및 영상으로 저장

---

## 🧩 구성 요소
conda create -n "env"
conda activate "env"
conda install -r requirements.txt
### 1. 씬 분할
python Scene_distribution.py
### 2. 객체 검출 Scene_{검출할 비디오 입력}
python detect_video.py  —video output_scenes/Scene_1.mp4 —output_dir detect_vid_output
### 3. 객체 추적 및 인식 알고리즘
python object_identification.py   —base_dir "/HOME_DIR/object_identification-main"   —frame_set_name "detect_vid_output"   —output_video "final_result.mp4"


