"""
Preprocessing script for Video Anomaly Detection & Localization model.

Usage:
    python src/preprocessing.py
"""

import os
import cv2
from tqdm import tqdm
from src.config import build_config

config = build_config()


def extract_frames_from_video(video_path, output_dir, sample_rate=16, max_frames=256):
    """
    Extract frames from a video by sampling every `sample_rate` frames.
    Saves up to `max_frames` sampled frames to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    saved_frames = 0
    current_frame = 0

    with tqdm(total=min(frame_count // sample_rate, max_frames), desc=f"Processing {video_name}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % sample_rate == 0:
                frame_filename = f"frame_{saved_frames + 1:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_frames += 1
                pbar.update(1)

                if saved_frames >= max_frames:
                    break

            current_frame += 1

    cap.release()
    cv2.destroyAllWindows()


def preprocess_videos(source_path, output_path):
    """
    Iterate through all .mp4 videos in the source folder and extract frames.
    """
    videos = [f for f in os.listdir(source_path) if f.lower().endswith(".mp4")]
    if not videos:
        print(f"[INFO] No .mp4 videos found in {source_path}")
        return

    print(f"[INFO] Found {len(videos)} videos. Starting preprocessing...\n")

    for video in videos:
        video_path = os.path.join(source_path, video)
        video_name = os.path.splitext(video)[0]
        video_output_dir = os.path.join(output_path, video_name)
        extract_frames_from_video(video_path, video_output_dir)

    print("\n[INFO] Preprocessing completed.")


if __name__ == "__main__":
    source_folder = config.source_video_path
    output_folder = config.frames_output_path

    os.makedirs(output_folder, exist_ok=True)
    preprocess_videos(source_folder, output_folder)
