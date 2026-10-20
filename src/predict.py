"""Предсказание для одной или нескольких картинок.

Использование:
    python src/predict.py path/to/image.jpg
    python src/predict.py path/to/folder/
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import models, transforms


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
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes),
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


def predict(paths, model, tf, device, class_names=("food", "non_food")):
    results = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[0]
            pred_idx = probs.argmax().item()
            results.append({
                "path": str(p),
                "pred": class_names[pred_idx],
                "prob": probs[pred_idx].item(),
                "probs": probs.cpu().tolist(),
            })
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_or_dir> [more...]")
        sys.exit(1)

    p = yaml.safe_load(open("params.yaml"))
    size = p["data"]["img_size"]

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(p["model"]["name"], p["model"]["num_classes"],
                      pretrained=False).to(device)
    model.load_state_dict(torch.load("models/model.pkl", map_location=device))
    model.eval()

    paths = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jpg")) + sorted(path.glob("*.png")))
        elif path.is_file():
            paths.append(path)
        else:
            print(f"Не найдено: {arg}")

    if not paths:
        print("Картинок не найдено")
        sys.exit(1)

    results = predict(paths, model, tf, device)
    print(f"Модель: {p['model']['name']}, use_augmented={p['train']['use_augmented']}")
    print(f"Картинок: {len(results)}\n")
    for r in results:
        print(f"{r['path']}")
        print(f"  -> {r['pred']}  (уверенность {r['prob']:.2%})")
        print(f"  food={r['probs'][0]:.3f}, non_food={r['probs'][1]:.3f}")


if __name__ == "__main__":
    main()
