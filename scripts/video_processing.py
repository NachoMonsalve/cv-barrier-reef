from pathlib import Path
import sys

# Add src to sys.path
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import required functions
from video_utils import get_video_properties, extract_frames

# Define project paths
project_root = Path(__file__).resolve().parent.parent
video_path = project_root / "data/raw/train_video/video/trial.mp4"
output_folder = project_root / "data/raw/train_video/frames"

if __name__ == "__main__":
    # Print resolved paths (for debugging purposes)
    print(f"Video path: {video_path}")
    print(f"Output folder: {output_folder}")

    # Step 1: Get video properties
    video_props = get_video_properties(str(video_path))
    print(f"Video properties: {video_props}")

    # Step 2: Extract frames
    frame_skip = 5
    saved_frames = extract_frames(str(video_path), str(output_folder), frame_skip)
    print(f"Saved frames: {saved_frames}")
