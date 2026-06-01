import os
import glob
import random
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

@dataclass
class Config:

    patch_size: Tuple[int, int, int] = (64, 64, 64)

    output_dir: Path = Path("outputs")

class LungNodule3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(1, 16, 3, padding=1),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),

            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(32, 2),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

    @property
    def gradcam_target_layer(self):

        return self.features[6]

class ManualGradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(
        self,
        module,
        input,
        output,
    ):

        self.activations = output.detach()

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor,
        target_class,
    ):

        self.model.zero_grad()

        logits = self.model(input_tensor)

        score = logits[0, target_class]

        score.backward()

        grads = self.gradients

        acts = self.activations

        weights = grads.mean(
            dim=(2,3,4),
            keepdim=True,
        )

        cam = F.relu(
            (weights * acts).sum(
                dim=1,
                keepdim=True,
            )
        )

        cam = F.interpolate(

            cam,

            size=input_tensor.shape[2:],

            mode="trilinear",

            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return cam

def load_dataset():

    dataset = []

    nodule_files = glob.glob(
        "localization_dataset/nodule/*.npy"
    )

    background_files = glob.glob(
        "localization_dataset/background/*.npy"
    )

    for path in nodule_files:

        volume = np.load(path)

        dataset.append({

            "id":
                os.path.basename(path)
                .replace(".npy", ""),

            "volume":
                volume,

            "label":
                1,

            "ground_truth":
                "Nodule",
        })

    for path in background_files:

        volume = np.load(path)

        dataset.append({

            "id":
                os.path.basename(path)
                .replace(".npy", ""),

            "volume":
                volume,

            "label":
                0,

            "ground_truth":
                "Background",
        })

    random.shuffle(dataset)

    print("DATASETSUMMARY")
    print(f"Nodules: {len(nodule_files)}")

    print(f"Backgrounds: {len(background_files)}")

    print(f"Total: {len(dataset)}")

    return dataset

def add_gaussian_noise(
    volume,
    sigma=0.03,
):

    noise = np.random.normal(
        0,
        sigma,
        volume.shape,
    )

    noisy = volume + noise

    return np.clip(
        noisy,
        volume.min(),
        volume.max(),
    )

def compute_activation_drift(
    cam1,
    cam2,
):

    return np.mean(
        np.abs(cam1 - cam2)
    )

def plot_gradcam(
    volume,
    cam,
    prediction,
    gt,
    sample_id,
    save_path,
):

    D, H, W = volume.shape

    axial = D // 2

    sagittal = W // 2

    coronal = H // 2

    planes = [

        (
            volume[axial],
            cam[axial],
            "Axial",
        ),

        (
            volume[:,:,sagittal],
            cam[:,:,sagittal],
            "Sagittal",
        ),

        (
            volume[:,coronal,:],
            cam[:,coronal,:],
            "Coronal",
        ),
    ]

    fig = plt.figure(
        figsize=(14,5),
        facecolor="#0e1117",
    )

    gs = gridspec.GridSpec(
        1,
        4,
        width_ratios=[1,1,1,0.05],
    )

    norm = Normalize(
        vmin=0,
        vmax=1,
    )

    for i, (img, heat, title) in enumerate(planes):

        ax = fig.add_subplot(gs[0,i])

        ax.imshow(
            img,
            cmap="gray",
        )

        ax.imshow(
            heat,
            cmap="inferno",
            alpha=0.45,
            norm=norm,
        )

        ax.set_title(
            title,
            color="white",
        )

        ax.axis("off")

    cax = fig.add_subplot(gs[0,3])

    cb = fig.colorbar(
        ScalarMappable(
            norm=norm,
            cmap="inferno",
        ),
        cax=cax,
    )

    cb.ax.tick_params(
        colors="white",
    )

    fig.suptitle(

        f"{sample_id}\n"
        f"GT: {gt} | "
        f"Prediction: "
        f"{prediction['predicted_label']} | "
        f"Nodule Prob: "
        f"{prediction['nodule_prob']:.3f}",

        color="white",
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close()

def prepare_tensor(
    volume,
    device,
):

    volume = (
        volume - volume.mean()
    ) / (
        volume.std() + 1e-8
    )

    tensor = (

        torch.tensor(
            volume,
            dtype=torch.float32,
        )

        .unsqueeze(0)

        .unsqueeze(0)

        .to(device)

    )

    return tensor

def get_prediction(
    model,
    tensor,
):

    with torch.no_grad():

        logits = model(tensor)

        probs = F.softmax(
            logits,
            dim=1,
        )[0]

    probs = probs.cpu().numpy()

    return {

        "background_prob":
            float(probs[0]),

        "nodule_prob":
            float(probs[1]),

        "predicted_class":
            int(np.argmax(probs)),

        "predicted_label":
            "Nodule"
            if np.argmax(probs) == 1
            else "Background",
    }

if __name__ == "__main__":

    cfg = Config()

    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )

    logger.info(
        f"Using device: {device}"
    )

    model = LungNodule3DCNN().to(device)

    checkpoint_path = "localizer_model.pth"

    if os.path.exists(checkpoint_path):

        model.load_state_dict(

            torch.load(
                checkpoint_path,
                map_location=device,
            )

        )

        logger.info(
            "Loaded trained weights."
        )

    else:

        logger.warning(

            "No trained weights found.\n"

            "Using random initialization."

        )

    model.eval()

    gradcam = ManualGradCAM(

        model,

        model.gradcam_target_layer,

    )

    dataset = load_dataset()

    all_results = []

    for sample in dataset:

        sid = sample["id"]

        volume = sample["volume"]

        gt = sample["ground_truth"]

        label = sample["label"]

        logger.info(
            f"Processing {sid}"
        )

        tensor = prepare_tensor(
            volume,
            device,
        )

        prediction = get_prediction(
            model,
            tensor,
        )

        cam = gradcam.generate(

            tensor,

            target_class=1,

        )

        plot_gradcam(

            volume=volume,

            cam=cam,

            prediction=prediction,

            gt=gt,

            sample_id=sid,

            save_path=
                cfg.output_dir /
                f"{sid}_gradcam.png",

        )

        noisy_volume = add_gaussian_noise(
            volume
        )

        noisy_tensor = prepare_tensor(
            noisy_volume,
            device,
        )

        noisy_pred = get_prediction(
            model,
            noisy_tensor,
        )

        noisy_cam = gradcam.generate(
            noisy_tensor,
            target_class=1,
        )

        drift = compute_activation_drift(
            cam,
            noisy_cam,
        )

        plot_gradcam(

            volume=noisy_volume,

            cam=noisy_cam,

            prediction=noisy_pred,

            gt=gt,

            sample_id=
                f"{sid}_NOISE",

            save_path=
                cfg.output_dir /
                f"{sid}_noise_gradcam.png",

        )

        all_results.append({

            "sample_id":
                sid,

            "ground_truth":
                label,

            "prediction":
                prediction["predicted_class"],

            "nodule_prob":
                prediction["nodule_prob"],

            "activation_drift":
                drift,
        })

    y_true = [
        r["ground_truth"]
        for r in all_results
    ]

    y_pred = [
        r["prediction"]
        for r in all_results
    ]

    y_prob = [
        r["nodule_prob"]
        for r in all_results
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    try:

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

    except:

        auc = 0.0

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("FINAL METRICS")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")

    print("\nConfusion Matrix:\n")

    print(cm)

    df = pd.DataFrame(all_results)

    df.to_csv(

        cfg.output_dir /
        "results.csv",

        index=False,

    )

    fig, ax = plt.subplots(
        figsize=(5,5)
    )

    im = ax.imshow(
        cm,
        cmap="Blues",
    )

    ax.set_xticks([0,1])

    ax.set_yticks([0,1])

    ax.set_xticklabels([
        "Background",
        "Nodule",
    ])

    ax.set_yticklabels([
        "Background",
        "Nodule",
    ])

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Ground Truth"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    for i in range(2):

        for j in range(2):

            ax.text(

                j,

                i,

                str(cm[i,j]),

                ha="center",

                va="center",

                fontsize=14,

            )

    plt.colorbar(im)

    plt.tight_layout()

    plt.savefig(

        cfg.output_dir /
        "confusion_matrix.png",

        dpi=300,

    )

    plt.close()

