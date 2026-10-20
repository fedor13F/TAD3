"""Обучение. Читает params.yaml, пишет models/model.pkl + metrics/train_metrics.json."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """Своя лёгкая CNN с нуля (~0.5M параметров)."""
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


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


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
    set_seed(p["train"]["seed"])

    base = p["data"]["augmented_dir"] if p["train"]["use_augmented"] else p["data"]["processed_dir"]
    size = p["data"]["img_size"]

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(f"{base}/training", transform=tf)
    val_ds = datasets.ImageFolder(f"{base}/validation", transform=tf)

    train_dl = DataLoader(train_ds, batch_size=p["train"]["batch_size"], shuffle=True,
                          num_workers=p["data"]["num_workers"])
    val_dl = DataLoader(val_ds, batch_size=p["train"]["batch_size"], shuffle=False,
                        num_workers=p["data"]["num_workers"])

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = get_model(p["model"]["name"], p["model"]["num_classes"], p["model"]["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["train"]["lr"], weight_decay=p["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(p["train"]["epochs"]):
        model.train()
        running = 0.0
        for x, y in tqdm(train_dl, desc=f"train {epoch+1}/{p['train']['epochs']}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += loss_fn(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/model.pkl")
    json.dump({
        "model": p["model"]["name"],
        "use_augmented": p["train"]["use_augmented"],
        "final_val_acc": history["val_acc"][-1],
        "history": history,
    }, open("metrics/train_metrics.json", "w"), indent=2)
    print("saved models/model.pkl")


if __name__ == "__main__":
    main()
