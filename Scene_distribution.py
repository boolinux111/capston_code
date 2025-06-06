import os
import shutil
import subprocess
import cv2
from scenedetect.detectors import AdaptiveDetector
import argparse

def detect_scenes_dynamic(input_path,
                          motion_threshold: float = 5.0,
                          min_skip: int = 0,
                          max_skip: int = 5,
                          adaptive_threshold: float = 3.0,
                          window_width: int = 2,
                          min_content_val: float = 15.0):
    """
    모션 변화량에 따라 프레임 스킵을 조정하며 AdaptiveDetector로 씬(컷) 구간을 계산합니다.

    Args:
        input_path (str): 입력 비디오 경로
        motion_threshold (float): 모션 값(motion_val)이 이 이상이면 skip=min_skip
        min_skip (int): 모션 클 때 최소 스킵 프레임 수
        max_skip (int): 모션 작을 때 최대 스킵 프레임 수
        adaptive_threshold (float): AdaptiveDetector adaptive ratio 기준 (높을수록 덜 민감)
        window_width (int): adaptive rolling average 윈도우 크기
        min_content_val (float): 컷으로 인식되려면 content score 최소값

    Returns:
        List[Tuple[float, float]]: (start_sec, end_sec) 씬 시간 리스트
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = AdaptiveDetector(
        adaptive_threshold=adaptive_threshold,
        window_width=window_width,
        min_content_val=min_content_val,
        min_scene_len=1
    )

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    cut_frame_nums = []
    frame_num = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_val = cv2.absdiff(prev_gray, gray).mean()
        skip = min_skip if motion_val > motion_threshold else max_skip

        cuts = detector.process_frame(frame_num, frame)
        if cuts:
            cut_frame_nums.extend(cuts)

        for _ in range(skip):
            if not cap.grab():
                break
            frame_num += 1

        prev_gray = gray
        frame_num += 1

    cuts = detector.post_process(frame_num)
    cut_frame_nums.extend(cuts)
    cap.release()

    cut_frame_nums = sorted(set(cut_frame_nums))
    boundaries = [0] + cut_frame_nums + [total_frames]
    scenes = [(boundaries[i]/fps, boundaries[i+1]/fps)
              for i in range(len(boundaries)-1)]
    return scenes


def merge_short_scenes(scenes, min_duration):
    """
    scenes 중 duration이 min_duration 미만인 씬은 이전 씬과 병합합니다.

    Args:
        scenes (List[Tuple[float, float]]): 원본 씬 리스트
        min_duration (float): 최소 씬 길이(초)

    Returns:
        List[Tuple[float, float]]: 병합 후 씬 리스트
    """
    if not scenes:
        return []
    merged = [scenes[0]]
    for start, end in scenes[1:]:
        prev_start, prev_end = merged[-1]
        duration = end - start
        if duration < min_duration:
            # 이전 씬의 끝을 현재 씬의 끝으로 확장
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def split_scenes(input_path, scenes, output_dir="scenes"):
    """
    FFmpeg를 이용해 씬 구간별로 영상을 분할 저장합니다.
    이전 output_dir이 존재하면 삭제 후 새로 생성합니다.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for idx, (start, end) in enumerate(scenes, 1):
        output_file = os.path.join(output_dir, f"Scene_{idx}.mp4")
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', f"{start:.3f}", '-to', f"{end:.3f}",
            '-c', 'copy', output_file
        ]
        print(f"Splitting Scene {idx}: {start:.2f}s → {end:.2f}s")
        subprocess.run(cmd, check=True)

    print("All scenes have been split and saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="씬 분할 스크립트")
    parser.add_argument("video_file", help="입력 비디오 파일 경로")
    parser.add_argument("--output_dir", default="output_scenes", help="씬 저장 폴더 이름")
    parser.add_argument("--motion_threshold", type=float, default=3.0, help="모션 임계값")
    parser.add_argument("--min_skip", type=int, default=0, help="모션 클 때 최소 스킵 프레임 수")
    parser.add_argument("--max_skip", type=int, default=5, help="모션 작을 때 최대 스킵 프레임 수")
    parser.add_argument("--adaptive_threshold", type=float, default=3.2, help="AdaptiveDetector 임계값")
    parser.add_argument("--window_width", type=int, default=4, help="adaptive rolling 윈도우 크기")
    parser.add_argument("--min_content_val", type=float, default=20.0, help="컷으로 인식될 최소 content score")
    parser.add_argument("--min_duration", type=float, default=5.0, help="병합할 최소 씬 길이(초)")
    args = parser.parse_args()

    video_file = args.video_file

    scenes = detect_scenes_dynamic(
        video_file,
        motion_threshold=args.motion_threshold,
        min_skip=args.min_skip,
        max_skip=args.max_skip,
        adaptive_threshold=args.adaptive_threshold,
        window_width=args.window_width,
        min_content_val=args.min_content_val
    )
    # 영상의 길이에 따라 n초 미만 씬은 병합
    scenes = merge_short_scenes(scenes, min_duration=args.min_duration)
    split_scenes(video_file, scenes, output_dir=args.output_dir)
