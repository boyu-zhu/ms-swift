import zipfile
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import os
from tqdm import tqdm
import requests

HF_TOKEN = os.getenv("HF_TOKEN")

def prepare_vlguard():
    zip_path = hf_hub_download(
        repo_id="ys-zong/VLGuard",
        filename="train.zip",
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir="./data",               # ⭐ 下载到指定目录
        local_dir_use_symlinks=False         # ⭐ 复制而不是软链接（避免仍然指向缓存）
    )
    print("下载路径:", zip_path)

    # 解压
    extract_dir = "./data/vlguard"
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print("解压完成:", extract_dir)

def prepare_vlsbench():
    dataset = load_dataset("Foreshhh/vlsbench", split='train')
    for item in dataset:
        image = item.get("image")
        path = item.get("image_path")
        image_path = os.path.join('/root/ms-swift/data/vlsbench', path)
        if image and path:
            # 确保目录存在
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            try:
                image.save(image_path)
                print(f"✅ Saved image to {path}")
            except Exception as e:
                print(f"❌ Failed to save {path}: {e}")

def prepare_llavaguard():
    hf_token = HF_TOKEN  # 你的 token
    set_name = "train"
    save_dir = "data/llavaguard/train"

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[Info] Folder {save_dir} already exists, skip downloading.")
        return
    os.makedirs(save_dir, exist_ok=True)

    dataset = load_dataset("AIML-TUDA/LlavaGuard", token=hf_token)
    urls = dataset[set_name]["url"]

    headers = {"User-Agent": "Mozilla/5.0"}
    auth_headers = {"Authorization": f"Bearer {hf_token}", "User-Agent": "Mozilla/5.0"}

    for i, url in tqdm(enumerate(urls), total=len(urls)):
        file_path = f"{save_dir}/{i}.jpg"
        if os.path.exists(file_path):
            # 文件已存在 -> 跳过
            continue

        try:
            if "huggingface" in url:
                r = requests.get(url, headers=auth_headers, timeout=(5, 30))
            else:
                r = requests.get(url, headers=headers, timeout=(5, 30))
            r.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(r.content)

        except Exception as e:
            print(f"[Warning] Failed to download {url} -> {e}")


if __name__ == "__main__":
    prepare_llavaguard()