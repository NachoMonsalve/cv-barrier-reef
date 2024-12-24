import cv2
import os
import os
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict

def get_video_properties(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Unable to open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration
    }

def extract_frames(video_path: str, output_folder: str, frame_skip: int = 1) -> int:
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Unable to open video {video_path}")

    frame_count = 0
    saved_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frames} frames to {output_folder}")
    return saved_frames

def process_voc_folder(annotations_folder: str) -> List[Dict]:
    all_data = []
    for file in os.listdir(annotations_folder):
        if file.endswith(".xml"):
            xml_path = os.path.join(annotations_folder, file)
            all_data.extend(parse_voc_xml(xml_path))
    return all_data


