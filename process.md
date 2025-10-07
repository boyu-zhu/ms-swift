python save_unsafebench.py
cd data
aws s3 cp s3://orby-ucd/llava.tar.gz .
mkdir -p llava
tar -xzvf llava.tar.gz -C llava --strip-components=1
aws s3 cp s3://orby-ucd/data/vlguard.zip .
unzip vlguard.zip
mv train vlguard
huggingface-cli download PKU-Alignment/SafeSora-Label videos.tar.gz --repo-type dataset --local-dir .
tar -xzvf videos.tar.gz
mkdir -p ~/.kaggle && printf '%s\n' '{"username":"boyuzhuuuu","key":"188571c84bfc03720ec6b2c92f2f3a7c"}' > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
curl -L -o ./dcsass-dataset.zip  https://www.kaggle.com/api/v1/datasets/download/mateohervas/dcsass-dataset
unzip dcsass-dataset.zip
curl -L -o ./tikharm-dataset.zip  https://www.kaggle.com/api/v1/datasets/download/anhoangvo/tikharm-dataset
unzip tikharm-dataset.zip -d tikharm-dataset

aws s3 cp s3://orby-ucd/data/FakeTT_DATA_OPENSOURCE.zip .
unzip FakeTT_DATA_OPENSOURCE.zip
mv video fakett

uv pip install -e .
uv pip install "deepspeed" -U
uv pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
uv pip install boto3 gdown
uv pip install torchvision
uv pip install flash-attn --no-build-isolation


# vision_distill


dependencies
```bash
uv pip install datasets boto3 openai pillow gdown kagglehub 
```

serve
```bash
nohup vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 128000 \
  --allowed-local-media-path / \
  > vllm_server.log 2>&1 &
```


run image distill
```bash
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 100 --dataset vlguard --run vlguard_qwen3
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 100 --dataset llavaguard --run llavaguard_qwen3
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 100 --dataset unsafebench --run unsafebench_qwen3
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 100 --dataset vlsbench --run vlsbench_qwen3
```


run video distill
```bash
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 10 --dataset lspd --run lspd_qwen3
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 10 --dataset safesora --run safesora_qwen3
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 10 --dataset fakesv --run fakesv_qwen3
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 10 --dataset tikharm --run tikharm_qwen3
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 10 --dataset fvc --run fvc_qwen3


python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 10 --dataset dcsass --run dcsass_qwen3
python main.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --max_concurrent 10 --dataset lspd --run lspd_qwen3
```


submit run
```bash
mcli run -f /vision_distill/infer_script/safesora_1.yaml --follow
mcli run -f /vision_distill/infer_script/safesora_2.yaml --follow
mcli run -f /vision_distill/infer_script/safesora_3.yaml --follow
mcli run -f /vision_distill/infer_script/safesora_4.yaml --follow

mcli run -f infer_script/train/omni2_5_text.yaml --follow
mcli run -f infer_script/train/omni2_5_image.yaml --follow
```

aws usage
```bash
aws s3 ls s3://orby-ucd --recursive
aws s3 cp s3://orby-ucd/distill/safesora_qwen3_0.jsonl .
aws s3 cp s3://orby-ucd/distill/safesora_qwen3_1.jsonl .
aws s3 cp s3://orby-ucd/distill/safesora_qwen3_2.jsonl .
aws s3 cp s3://orby-ucd/distill/safesora_qwen3_3.jsonl .
aws s3 cp s3://orby-ucd/distill/safesora_qwen3_4.jsonl .
```


upload dataset
```
huggingface-cli repo create text_distill --type dataset
# Usage:  huggingface-cli upload [dataset_repo_id] [local_path] [path_in_repo] --repo-type dataset
huggingface-cli upload text_distill data/text_ready_sft_data . --repo-type dataset
hf download boyuzhuGPT/text_distill --repo-type dataset --local-dir ./text_distill_data

huggingface-cli repo create toxic-chat-audio-qwen3 --type dataset
# Usage:  huggingface-cli upload [dataset_repo_id] [local_path] [path_in_repo] --repo-type dataset
huggingface-cli upload vlguard-qwen3 data/vlguard . --repo-type dataset
hf download boyuzhuGPT/text_distill --repo-type dataset --local-dir ./text_distill_data
huggingface-cli upload toxic-chat-audio-qwen3 . . --repo-type dataset
```
download dataset
```
hf download boyuzhuGPT/text_distill  --repo-type dataset --local-dir data/text_ready_sft_data

hf download boyuzhuGPT/vlguard-qwen3  --repo-type dataset --local-dir data/vlguard
cd /ms-swift/data/vlguard
tar -xzf train.tar.gz
cd /ms-swift
```


hf download Foreshhh/vlsbench  --repo-type dataset --local-dir ./vlsbench
tar czf fvc.tar.gz fvc
tar -xf imgs.tar

downlaod image
```bash
uv run hf download boyuzhuGPT/vlsbench-qwen3  --repo-type dataset --local-dir /ms-swift/data/vlsbench
cd /ms-swift/data/vlsbench
tar -xf imgs.tar
uv run hf download boyuzhuGPT/unsafebench-qwen3  --repo-type dataset --local-dir /ms-swift/data/unsafebench
cd /ms-swift/data/unsafebench
tar -xzf train.tar.gz
uv run hf download boyuzhuGPT/vlguard-qwen3  --repo-type dataset --local-dir /ms-swift/data/vlguard
cd /ms-swift/data/vlguard
tar -xzf train.tar.gz
uv run hf download boyuzhuGPT/llavaguard-qwen3  --repo-type dataset --local-dir /ms-swift/data/llavaguard
cd /ms-swift/data/llavaguard
tar -xzf train.tar.gz
cd /ms-swift
```




downlaod audio
```bash
uv run hf download boyuzhuGPT/toxic-chat-audio-qwen3  --repo-type dataset --local-dir /ms-swift/data/toxic_chat_audio
cd /ms-swift/data/toxic_chat_audio
tar -xzf toxic_chat_audio.tar.gz
cd /ms-swift
```

downlaod videos
```bash
uv run hf download boyuzhuGPT/toxic-chat-audio-qwen3  --repo-type dataset --local-dir /ms-swift/data/toxic_chat_audio
cd /ms-swift/data/safesora
tar -xzf videos.tar.gz
cd /ms-swift
```
