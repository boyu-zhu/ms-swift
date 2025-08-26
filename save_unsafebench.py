from datasets import load_dataset, Image as HFImage
import os, io, re
from PIL import Image, ImageFile

# 允许加载部分损坏的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

def sanitize(s):
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-\.]", "", s)
    return s[:60]

output_dir = "/root/ms-swift/data/unsafebench"
os.makedirs(output_dir, exist_ok=True)

# 1) 加载数据
dataset = load_dataset("yiting/UnsafeBench")["train"]

# 2) 添加 id
dataset = dataset.map(lambda ex, idx: {"id": idx}, with_indices=True)

# 3) 禁用自动 PIL 解码
dataset = dataset.cast_column("image", HFImage(decode=False))

for ex in dataset:
    img_meta = ex["image"]
    img_bytes = img_meta.get("bytes")
    if img_bytes is None and img_meta.get("path"):
        with open(img_meta["path"], "rb") as f:
            img_bytes = f.read()

    if not img_bytes:
        print(f"Skip id {ex['id']}: empty image bytes")
        continue

    try:
        im = Image.open(io.BytesIO(img_bytes))
        im.load()
        if "exif" in im.info:
            im.info.pop("exif")
    except Exception as e:
        print(f"Skip id {ex['id']}: open error -> {e}")
        continue

    # 强制转为 RGB，再存 JPEG
    if im.mode != "RGB":
        im = im.convert("RGB")

    filename = f"{ex['id']}.jpg"
    path = os.path.join(output_dir, filename)

    try:
        im.save(path, format="JPEG", quality=95, optimize=True)
        print("Saved:", path)
    except Exception as e:
        try:
            im.convert("RGB").save(path, format="JPEG", quality=95, optimize=True)
            print("Saved (fallback):", path)
        except Exception as e2:
            print(f"JPEG save fail id {ex['id']}: {e2}")
