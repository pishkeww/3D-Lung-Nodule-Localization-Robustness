import os
import cv2
import glob
import shutil
import random
import numpy as np

ROOT_DIR = r"D:\kinect codes\files\archive (1)\LIDC-IDRI-slices"

OUTPUT_DIR = "localization_dataset"

TARGET_SIZE = 64

TARGET_DEPTH = 64

BACKGROUND_PER_NODULE = 2

if os.path.exists(OUTPUT_DIR):

    shutil.rmtree(OUTPUT_DIR)

os.makedirs(
    os.path.join(OUTPUT_DIR, "nodule"),
    exist_ok=True
)

os.makedirs(
    os.path.join(OUTPUT_DIR, "background"),
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

def load_slices(folder):

    slice_paths = sorted(
        glob.glob(
            os.path.join(folder, "*")
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

        img /= 255.0

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

def generate_background(volume):

    z = random.randint(0, 32)

    y = random.randint(0, 32)

    x = random.randint(0, 32)

    patch = volume[
        z:z+64,
        y:y+64,
        x:x+64
    ]

    pad_z = 64 - patch.shape[0]

    pad_y = 64 - patch.shape[1]

    pad_x = 64 - patch.shape[2]

    if pad_z > 0 or pad_y > 0 or pad_x > 0:

        patch = np.pad(

            patch,

            (
                (0, max(0, pad_z)),
                (0, max(0, pad_y)),
                (0, max(0, pad_x))
            ),

            mode="constant"
        )

    return patch

patient_dirs = sorted(

    glob.glob(
        os.path.join(ROOT_DIR, "LIDC-IDRI-*")
    )

)

nodule_counter = 0

background_counter = 0

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

        volume = load_slices(
            image_dir
        )

        if volume is None:
            continue

        np.save(

            os.path.join(

                OUTPUT_DIR,

                "nodule",

                f"nodule_{nodule_counter:04d}.npy"

            ),

            volume
        )

        print(
            f"Nodule: {volume.shape}"
        )

        nodule_counter += 1

        for _ in range(BACKGROUND_PER_NODULE):

            bg = generate_background(
                volume
            )

            np.save(

                os.path.join(

                    OUTPUT_DIR,

                    "background",

                    f"background_{background_counter:04d}.npy"

                ),

                bg
            )

            print(
                f"Background: {bg.shape}"
            )

            background_counter += 1

print("\nDONE")

print(f"Nodules: {nodule_counter}")

print(f"Backgrounds: {background_counter}")