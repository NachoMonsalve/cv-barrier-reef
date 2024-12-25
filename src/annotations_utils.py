import os
import json
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict


# -------------------- REEF DATASET FUNCTIONS --------------------

def filter_annotated_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["annotations"] != "[]"].copy()


def parse_reef_annotations(row: pd.Series, image_folder: str) -> List[Dict]:
    annotations = json.loads(row["annotations"].replace("'", '"'))
    parsed_data = []
    for box in annotations:
        parsed_data.append({
            "filename": str(Path(image_folder) / f"video_{row['video_id']}" / f"{row['video_frame']}.jpg"),
            "width": box["width"],
            "height": box["height"],
            "label": "starfish",
            "xmin": box["x"],
            "ymin": box["y"],
            "xmax": box["x"] + box["width"],
            "ymax": box["y"] + box["height"]
        })
    return parsed_data


def preprocess_reef_annotations(csv_path: str, image_folder: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, delimiter=';')
    except Exception as e:
        raise ValueError(f"Error reading Reef annotations CSV: {e}")

    filtered_df = filter_annotated_rows(df)

    standardized_data = []
    for _, row in filtered_df.iterrows():
        standardized_data.extend(parse_reef_annotations(row, image_folder))

    return pd.DataFrame(standardized_data)


# -------------------- VOC DATASET FUNCTIONS --------------------

def parse_single_voc_xml(xml_path: str, image_folder: str) -> List[Dict]:
    data = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    image_path = str(Path(image_folder) / filename)

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

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


def preprocess_voc_annotations(annotations_folder: str, image_folder: str) -> pd.DataFrame:
    all_data = []
    for xml_file in os.listdir(annotations_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(annotations_folder, xml_file)
            all_data.extend(parse_single_voc_xml(xml_path, image_folder))

    return pd.DataFrame(all_data)


# -------------------- GENERAL FUNCTIONS --------------------

def save_annotations_as_csv(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)


def visualize_annotations(df: pd.DataFrame, num_images: int = 5):
    unique_files = df["filename"].unique()[:num_images]
    for filename in unique_files:
        img = cv2.imread(filename)
        if img is None:
            print(f"Image not found: {filename}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")

        annotations = df[df["filename"] == filename]
        for _, row in annotations.iterrows():
            x_min, y_min, x_max, y_max = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            label = row["label"]
            plt.gca().add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="red", facecolor="none")
            )
            plt.text(x_min, y_min - 10, label, color="red", fontsize=12, backgroundcolor="white")

        plt.title(f"Annotations for {filename}")
        plt.show()
