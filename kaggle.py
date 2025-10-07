import kagglehub

# Download latest version
path = kagglehub.dataset_download("anhoangvo/tikharm-dataset")

print("Path to dataset files:", path)

path = kagglehub.dataset_download("mateohervas/dcsass-dataset")

print("Path to dataset files:", path)

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="PKU-Alignment/SafeSora-Label", filename="videos.tar.gz", repo_type="dataset", local_dir="/ms-swift/data/safesora")