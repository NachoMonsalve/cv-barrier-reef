
# helpers.py
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
from collections import Counter
import json
import matplotlib.pyplot as plt
import pandas as pd

import os
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Dict
import cv2

import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import List, Dict



import pandas as pd
import json
from pathlib import Path

def filter_annotated_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["annotations"] != "[]"].copy()

def parse_annotations_row(row: pd.Series, image_folder: str) -> list:
    annotations = json.loads(row["annotations"].replace("'", '"'))
    parsed_data = []
    for box in annotations:
        parsed_data.append({
            "filename": str(Path(image_folder) / f"video_{row['video_id']}" / f"{row['video_frame']}.jpg"),
            "width": box["width"],
            "height": box["height"],
            "label": "starfish",  # Dataset-specific label
            "xmin": box["x"],
            "ymin": box["y"],
            "xmax": box["x"] + box["width"],
            "ymax": box["y"] + box["height"]
        })
    return parsed_data

def standardize_reef_annotations(df: pd.DataFrame, image_folder: str) -> pd.DataFrame:
    standardized_data = []
    for _, row in df.iterrows():
        standardized_data.extend(parse_annotations_row(row, image_folder))
    return pd.DataFrame(standardized_data)

def preprocess_reef_annotations(csv_path: str, image_folder: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    filtered_df = filter_annotated_rows(df)
    return standardize_reef_annotations(filtered_df, image_folder)




def parse_single_voc_xml(xml_path: str, image_folder: str) -> List[Dict]:
    data = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract image filename
    filename = root.find("filename").text
    image_path = str(Path(image_folder) / filename)

    # Extract image size
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Extract object annotations
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        data.append({
            "filename": image_path,
            "width": width,
            "height": height,
            "label": label,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    return data


def process_voc_folder(annotations_folder: str, image_folder: str) -> List[Dict]:
    all_data = []
    for xml_file in os.listdir(annotations_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(annotations_folder, xml_file)
            all_data.extend(parse_single_voc_xml(xml_path, image_folder))
    return all_data


def voc_annotations_to_dataframe(annotations: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(annotations)


def preprocess_voc_annotations(annotations_folder: str, image_folder: str) -> pd.DataFrame:
    annotations = process_voc_folder(annotations_folder, image_folder)

    return voc_annotations_to_dataframe(annotations)


def visualize_annotations_from_df(df: pd.DataFrame, num_images: int = 5):
    unique_files = df["filename"].unique()[:num_images]

    for filename in unique_files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")

        # Plot bounding boxes
        image_annotations = df[df["filename"] == filename]
        for _, row in image_annotations.iterrows():
            x_min, y_min, x_max, y_max = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            label = row["label"]
            plt.gca().add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="red", facecolor="none")
            )
            plt.text(x_min, y_min - 10, label, color="red", fontsize=12, backgroundcolor="white")

        plt.title(f"Annotations for {filename}")
        plt.show()