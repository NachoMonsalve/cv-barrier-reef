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
    """
    Filter rows in the Reef dataset that have annotations.
    """
    return df[df["annotations"] != "[]"].copy()


def parse_reef_annotations(row: pd.Series, image_folder: str) -> List[Dict]:
    """
    Parse Reef annotations into a general format.
    """
    annotations = json.loads(row["annotations"].replace("'", '"'))
    parsed_data = []
    for box in annotations:
        parsed_data.append({
            "filename": str(Path(image_folder) / f"{row['video_id']}_{row['video_frame']}.jpg"),
            "width": box["width"],
            "height": box["height"],
            "label": "starfish",
            "xmin": box["x"],
            "ymin": box["y"],
            "xmax": box["x"] + box["width"],
            "ymax": box["y"] + box["height"]
        })
    return parsed_data


def convert_reef_to_general_format(input_csv: str, image_folder: str) -> pd.DataFrame:
    """
    Convert Reef annotations to a general format CSV.
    """
    df = pd.read_csv(input_csv, delimiter=';')

    general_format = []
    for _, row in df.iterrows():
        filename = f"{row['video_id']}_{row['video_frame']}.jpg"
        full_image_path = str(Path(image_folder) / filename)

        if row["annotations"] != "[]":
            annotations = json.loads(row["annotations"].replace("'", '"'))
            for ann in annotations:
                general_format.append({
                    "filename": full_image_path,
                    "width": ann["width"],
                    "height": ann["height"],
                    "label": "starfish",
                    "xmin": ann["x"],
                    "ymin": ann["y"],
                    "xmax": ann["x"] + ann["width"],
                    "ymax": ann["y"] + ann["height"]
                })
        else:
            # If no annotations, still include the image but no bounding boxes
            general_format.append({
                "filename": full_image_path,
                "width": None,
                "height": None,
                "label": None,
                "xmin": None,
                "ymin": None,
                "xmax": None,
                "ymax": None
            })

    return pd.DataFrame(general_format)


# -------------------- VOC DATASET FUNCTIONS --------------------

def parse_single_voc_xml(xml_path: str, image_folder: str) -> List[Dict]:
    """
    Parse a single VOC XML file into a general format.
    """
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
    """
    Parse all VOC XML annotations in a folder into a general format.
    """
    all_data = []
    for xml_file in os.listdir(annotations_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(annotations_folder, xml_file)
            all_data.extend(parse_single_voc_xml(xml_path, image_folder))

    return pd.DataFrame(all_data)


# -------------------- GENERAL FUNCTIONS --------------------

def save_annotations_as_csv(df: pd.DataFrame, output_path: str):
    """
    Save annotations DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)


def visualize_annotations(df: pd.DataFrame, num_images: int = 5):
    """
    Visualize bounding box annotations from a general format DataFrame.
    """
    unique_files = df["filename"].dropna().unique()[:num_images]

    for filename in unique_files:
        img = cv2.imread(filename)
        if img is None:
            print(f"Image not found: {filename}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")

        # Plot bounding boxes
        annotations = df[df["filename"] == filename].dropna()
        for _, row in annotations.iterrows():
            if pd.notna(row["xmin"]):  # Only plot boxes if they exist
                x_min, y_min, x_max, y_max = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                label = row["label"] if pd.notna(row["label"]) else "No Label"
                plt.gca().add_patch(
                    plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=2, edgecolor="red", facecolor="none")
                )
                plt.text(x_min, y_min - 10, label, color="red", fontsize=12, backgroundcolor="white")

        plt.title(f"Annotations for {filename}")
        plt.show()
