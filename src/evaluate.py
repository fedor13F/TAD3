"""Оценка на evaluation split: метрики + confusion matrix."""
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(name: str, n_classes: int, pretrained: bool):
    if name == "resnet18":
        m = models.resnet18(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == "resnet50":
        m = models.resnet50(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    elif name == "simplecnn":
        m = SimpleCNN(n_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def main():
    p = yaml.safe_load(open("params.yaml"))
    base = p["data"]["augmented_dir"] if p["train"]["use_augmented"] else p["data"]["processed_dir"]
    size = p["data"]["img_size"]

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(f"{base}/evaluation", transform=tf)
    dl = DataLoader(ds, batch_size=p["train"]["batch_size"], shuffle=False,
                    num_workers=p["data"]["num_workers"])

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(p["model"]["name"], p["model"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(torch.load("models/model.pkl", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            out = model(x)
            y_pred.extend(out.argmax(1).cpu().tolist())
            y_true.extend(y.tolist())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1": f1_score(y_true, y_pred, average="binary"),
        "model": p["model"]["name"],
        "use_augmented": p["train"]["use_augmented"],
    }
    Path("metrics").mkdir(exist_ok=True)
    json.dump(metrics, open("metrics.json", "w"), indent=2)

    cm = confusion_matrix(y_true, y_pred).tolist()
    Path("plots").mkdir(exist_ok=True)
    json.dump({"confusion_matrix": cm, "labels": ds.classes},
              open("plots/confusion_matrix.json", "w"), indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
