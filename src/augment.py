"""Аугментация train-части: оригиналы + n аугментаций на картинку."""
import random
import shutil
import yaml
from pathlib import Path
from PIL import Image
from torchvision import transforms


def make_aug(img: Image.Image, p: dict, rng: random.Random) -> Image.Image:
    ops = []
    if rng.random() < 0.5:
        ops.append(transforms.RandomRotation(p["rotate_deg"]))
    if rng.random() < 0.5:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    if rng.random() < 0.5:
        ops.append(transforms.ColorJitter(
            brightness=p["color_jitter"],
            contrast=p["color_jitter"],
            saturation=p["color_jitter"],
        ))
    if rng.random() < 0.5:
        s = rng.uniform(*p["crop_scale"])
        w, h = img.size
        ops.append(transforms.RandomResizedCrop((h, w), scale=(s, s)))
    t = transforms.Compose(ops) if ops else transforms.Lambda(lambda x: x)
    return t(img)


def main():
    p = yaml.safe_load(open("params.yaml"))
    proc = Path(p["data"]["processed_dir"])
    aug = Path(p["data"]["augmented_dir"])
    n = p["aug"]["n_per_image"]
    rng = random.Random(p["train"]["seed"])

    if aug.exists():
        shutil.rmtree(aug)
    aug.mkdir(parents=True)

    for split in ["validation", "evaluation"]:
        shutil.copytree(proc / split, aug / split)

    for cls in ["food", "non_food"]:
        src_dir = proc / "training" / cls
        dst_dir = aug / "training" / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.glob("*.jpg")):
            img = Image.open(f)
            img.save(dst_dir / f.name)
            for i in range(n):
                aug_img = make_aug(img.copy(), p["aug"], rng)
                aug_img.save(dst_dir / f"{f.stem}_aug{i}.jpg")
    print("Augmented ->", aug)


if __name__ == "__main__":
    main()
