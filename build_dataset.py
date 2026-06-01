import os
import cv2
import glob
import shutil
import numpy as np

ROOT_DIR = r"D:\kinect codes\files\archive (1)\LIDC-IDRI-slices"

OUTPUT_DIR = "dataset"

TARGET_SIZE = 64

TARGET_DEPTH = 64

if os.path.exists(OUTPUT_DIR):

    print("\nDeleting old dataset...\n")

    shutil.rmtree(OUTPUT_DIR)

os.makedirs(
    os.path.join(OUTPUT_DIR, "benign"),
    exist_ok=True
)

os.makedirs(
    os.path.join(OUTPUT_DIR, "malignant"),
    exist_ok=True
)

def fix_depth(volume, target_depth=64):

    current_depth = volume.shape[0]

    if current_depth > target_depth:

        volume = volume[:target_depth]

    elif current_depth < target_depth:

        pad_amount = target_depth - current_depth

        padding = np.zeros(

            (
                pad_amount,
                volume.shape[1],
                volume.shape[2]
            ),

            dtype=volume.dtype
        )

        volume = np.concatenate(
            [volume, padding],
            axis=0
        )

    return volume

def load_volume(image_folder):

    slice_paths = sorted(

        glob.glob(
            os.path.join(image_folder, "*")
        )

    )

    slices = []

    for path in slice_paths:

        img = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        img = cv2.resize(
            img,
            (TARGET_SIZE, TARGET_SIZE)
        )

        img = img.astype(np.float32)

        img = img / 255.0

        slices.append(img)

    if len(slices) == 0:
        return None

    volume = np.stack(
        slices,
        axis=0
    )

    volume = fix_depth(
        volume,
        TARGET_DEPTH
    )

    return volume

patient_dirs = sorted(

    glob.glob(
        os.path.join(ROOT_DIR, "LIDC-IDRI-*")
    )

)

counter = 0

for patient_dir in patient_dirs:

    nodules = sorted(

        glob.glob(
            os.path.join(patient_dir, "nodule-*")
        )

    )

    for nodule_dir in nodules:

        image_dir = os.path.join(
            nodule_dir,
            "images"
        )

        volume = load_volume(image_dir)

        if volume is None:
            continue

        print(
            f"Volume Shape: {volume.shape}"
        )

        if counter % 2 == 0:

            label = "benign"

        else:

            label = "malignant"

        save_path = os.path.join(

            OUTPUT_DIR,

            label,

            f"sample_{counter:04d}.npy"
        )

        np.save(
            save_path,
            volume
        )

        print(
            f"Saved: {save_path}"
        )

        counter += 1

print("DATASET CREATION COMPLETE")


print(f"\nTotal Samples: {counter}")