def split_images_and_annotations(csv_path: str, image_folder: str, yolo_folder: str, train_ratio: float = 0.8):
    """
    Split images and annotations into YOLO-compatible train and val folders.
    """
    # Create YOLO folder structure
    train_images_folder = os.path.join(yolo_folder, "images/train")
    val_images_folder = os.path.join(yolo_folder, "images/val")
    train_labels_folder = os.path.join(yolo_folder, "labels/train")
    val_labels_folder = os.path.join(yolo_folder, "labels/val")
    
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    # Read annotations CSV
    df = pd.read_csv(csv_path)

    # Split images into train and val
    unique_images = df["filename"].unique()
    train_files, val_files = train_test_split(unique_images, train_size=train_ratio, random_state=42)

    # Helper function to save YOLO annotation files
    def save_yolo_annotations(subset_files, images_folder, labels_folder):
        for image_path in subset_files:
            # Copy image to the correct folder
            image_name = os.path.basename(image_path)
            target_image_path = os.path.join(images_folder, image_name)
            shutil.copy(image_path, target_image_path)

            # Filter annotations for this image
            annotations = df[df["filename"] == image_path]

            # Prepare YOLO-format annotations
            yolo_annotations = []
            for _, row in annotations.iterrows():
                if pd.notna(row["xmin"]):  # Ensure there are bounding boxes
                    class_id = 0  # Assuming a single class "starfish"
                    img_width = row["width"]
                    img_height = row["height"]

                    x_center = ((row["xmin"] + row["xmax"]) / 2) / img_width
                    y_center = ((row["ymin"] + row["ymax"]) / 2) / img_height
                    box_width = (row["xmax"] - row["xmin"]) / img_width
                    box_height = (row["ymax"] - row["ymin"]) / img_height

                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

            # Save YOLO-format annotations to .txt
            label_file_path = os.path.join(labels_folder, f"{Path(image_name).stem}.txt")
            with open(label_file_path, "w") as f:
                f.write("\n".join(yolo_annotations))

    # Save train and val images and annotations
    save_yolo_annotations(train_files, train_images_folder, train_labels_folder)
    save_yolo_annotations(val_files, val_images_folder, val_labels_folder)
