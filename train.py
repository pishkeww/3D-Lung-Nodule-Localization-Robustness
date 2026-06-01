import os
import glob
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

DATASET_DIR = "dataset"
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
##
class LungDataset(Dataset):

    def __init__(self, paths, labels):

        self.paths = paths
        self.labels = labels

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, idx):

        volume = np.load(
            self.paths[idx]
        )

        volume = torch.tensor(
            volume,
            dtype=torch.float32
        )

        volume = volume.unsqueeze(0)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return volume, label

benign = glob.glob(
    os.path.join(
        DATASET_DIR,
        "benign",
        "*.npy"
    )
)

malignant = glob.glob(
    os.path.join(
        DATASET_DIR,
        "malignant",
        "*.npy"
    )
)

paths = benign + malignant

labels = [0] * len(benign) + [1] * len(malignant)

combined = list(zip(paths, labels))

random.shuffle(combined)

paths, labels = zip(*combined)

train_paths, test_paths, train_labels, test_labels = train_test_split(

    paths,
    labels,

    test_size=0.2,

    random_state=42,

    stratify=labels,
)

train_dataset = LungDataset(
    train_paths,
    train_labels,
)

test_dataset = LungDataset(
    test_paths,
    test_labels,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class Simple3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

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

model = Simple3DCNN().to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)



print("\nSTARTING TRAINING\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for volumes, labels in train_loader:

        volumes = volumes.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(volumes)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )


torch.save(
    model.state_dict(),
    "lung_model.pth"
)

print("\nMODELSAVED\n")

model.eval()
y_true = []
y_pred = []
y_prob = []

with torch.no_grad():

    for volumes, labels in test_loader:

        volumes = volumes.to(DEVICE)

        outputs = model(volumes)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            preds.cpu().numpy()
        )

        y_prob.extend(
            probs[:, 1].cpu().numpy()
        )


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

cm = confusion_matrix(
    y_true,
    y_pred
)


print("FINAL RESULTS")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:\n")

print(cm)