import zipfile
from huggingface_hub import hf_hub_download
from datasets import load_dataset, DownloadConfig
import os
from tqdm import tqdm
import requests
from huggingface_hub import hf_hub_download
import os
import zipfile
import tarfile
import gdown
import boto3
import os
import os
# import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import argparse

def download_files(bucket_name, keys, local_dir):
    """
    从 S3 桶中下载一系列文件到本地目录。

    参数：
    - bucket_name: S3 桶名（string）
    - keys: 要下载的 S3 对象 key 列表（list of strings）
    - local_dir: 本地存放目录（string）
    """
    s3 = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)

    for key in keys:
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)
        try:
            print(f"Downloading s3://{bucket_name}/{key} → {local_path}")
            s3.download_file(bucket_name, key, local_path)
        except NoCredentialsError:
            print("Error: AWS credentials not found.")
            return
        except ClientError as e:
            print(f"Error downloading {key}: {e}")
        else:
            print(f"Successfully downloaded {filename}")



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
    dataset = load_dataset("Foreshhh/vlsbench", split='train', token=HF_TOKEN)
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

def prepare_safesora():
    zip_path = hf_hub_download(
        repo_id="PKU-Alignment/SafeSora-Label",
        filename="videos.tar.gz",
        repo_type="dataset",
        token="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",
        local_dir="./data",               # ⭐ 下载到指定目录
        local_dir_use_symlinks=False      # ⭐ 复制而不是软链接（避免仍然指向缓存）
    )
    print("下载路径:", zip_path)

    # 解压目录
    extract_dir = "./data/safesora"
    os.makedirs(extract_dir, exist_ok=True)

    # ⭐ 如果目录非空，直接跳过解压
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"[Info] 文件夹 {extract_dir} 已存在且非空，跳过解压。")
        return extract_dir

    # 否则执行解压
    with tarfile.open(zip_path, "r:gz") as tar_ref:
        tar_ref.extractall(extract_dir)

    print("解压完成:", extract_dir)



def prepare_text_datasets():
    bucket = 'orby-ucd'
    keys = [
        'sft/Aegis_w_gt_sft.jsonl',
        'sft/beaver_w_gt_sft.jsonl',
        'sft/toxic_w_gt_sft.jsonl',
        'sft/wildguardmix_w_gt_sft.jsonl'
    ]
    target_dir = './data/text_ready_sft_data'  # 修改为你想保存的本地目录，比如 "/ms-swift/downloaded_sft"
    download_files(bucket, keys, target_dir)

def prepare_audio_datasets():
    bucket = 'orby-ucd'
    keys = [
        'sft/Aegis_w_gt_sft.jsonl',
        'sft/beaver_w_gt_sft.jsonl',
        'sft/toxic_w_gt_sft.jsonl',
        'sft/wildguardmix_w_gt_sft.jsonl'
    ]
    target_dir = './data/text_ready_sft_data'  # 修改为你想保存的本地目录，比如 "/ms-swift/downloaded_sft"
    download_files(bucket, keys, target_dir)

def prepare_tocix_chat_audio():
    bucket = 'orby-ucd'
    keys = [
        'data/toxic_chat_audio.tar.gz',
        'sft/toxic_chat_audio.jsonl',

    ]
    target_dir = './data/audio'  # 修改为你想保存的本地目录，比如 "/ms-swift/downloaded_sft"
    # download_files(bucket, keys, target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # 下载文件
    download_files(bucket, keys, target_dir)

    # 找到刚下载的 tar.gz 文件路径
    tar_filename = os.path.basename(keys[0])  # "toxic_chat_audio.tar.gz"
    tar_path = os.path.join(target_dir, tar_filename)

    # 解压 tar.gz 到指定目录
    # 假设你想把它解压到 `./data/audio/extracted`，你可以改目录
    extract_dir = os.path.join(target_dir, "toxic_chat_audio")
    os.makedirs(extract_dir, exist_ok=True)

    # 检查文件是否存在再解压
    if os.path.isfile(tar_path):
        try:
            print(f"Extracting {tar_path} → {extract_dir}")
            with tarfile.open(tar_path, mode="r:gz") as tar_ref:
                tar_ref.extractall(path=extract_dir)
            print("Extraction complete.")
        except tarfile.TarError as e:
            print(f"Error extracting tar file: {e}")
    else:
        print(f"Tar file not found: {tar_path}")


def prepare_fakesv():
    # 要下载的文件 ID 和输出文件名
    drive_files = [
        ("1-W_QHoKczSB-DJ4YzkO35PBK9uSEG1dL", "complete.jsonl"),
        ("1-Wfru9llIW8EloZ5RHoOuTjVKQAOpvkr", "videos.zip")
    ]
    
    # 下载目录
    download_dir = "./downloads"
    os.makedirs(download_dir, exist_ok=True)
    
    for file_id, filename in drive_files:
        output_path = os.path.join(download_dir, filename)
        # 如果文件已存在就跳过下载
        if os.path.exists(output_path):
            print(f"⚡ 已存在，跳过下载：{output_path}")
        else:
            print(f"⬇ 正在下载 {filename} …")
            # fuzzy=True 让 gdown 能解析标准分享链接
            gdown.download(id=file_id, output=output_path, fuzzy=True)
        
        # 如果是 zip 文件，就解压
        if filename.lower().endswith(".zip"):
            extract_to = "./data/fakesv"
            os.makedirs(extract_to, exist_ok=True)
            print(f"📦 解压 {filename} 到 {extract_to}")
            with zipfile.ZipFile(output_path, 'r') as zf:
                zf.extractall(extract_to)
            print(f"✅ 解压完成：{extract_to}")



def main(mode):
    """
    mode: 'text', 'image', 或 'all'
    """
    if mode in ('image', 'all'):
        print("=== Preparing image datasets ===")
        prepare_vlguard()
        prepare_vlsbench()
        prepare_llavaguard()
        pre()
    if mode in ('text', 'all'):
        print("=== Preparing text datasets ===")
        prepare_text_datasets()
    if mode in ('video', 'all'):
        print("=== Preparing image datasets ===")
        # prepare_safesora()
        prepare_fakesv()
    if mode in ('audio', 'all'):
        print("=== Preparing image datasets ===")
        prepare_tocix_chat_audio()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备 dataset 脚本")
    parser.add_argument(
        "--mode",
        choices=["text", "image", "video", "audio", "all"],
        default="all",
        help="选择要准备的资源：text, image, 或 all（默认）"
    )
    args = parser.parse_args()

    print(f"运行模式: {args.mode}")
    main(args.mode)