"""Подготовка: resize всех картинок до img_size и сохранение в processed."""
import shutil
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def process_split(src: Path, dst: Path, size: int):
    for split in ["training", "validation", "evaluation"]:
        for cls in ["food", "non_food"]:
            src_dir = src / split / cls
            dst_dir = dst / split / cls
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(src_dir.glob("*.jpg"))
            for f in tqdm(files, desc=f"{split}/{cls}"):
                img = Image.open(f).convert("RGB").resize((size, size))
                img.save(dst_dir / f.name, quality=90)


def main():
    p = yaml.safe_load(open("params.yaml"))
    size = p["data"]["img_size"]
    src = Path(p["data"]["raw_dir"])
    dst = Path(p["data"]["processed_dir"])
    process_split(src, dst, size)
    print(f"Done. Processed -> {dst}")


if __name__ == "__main__":
    main()
