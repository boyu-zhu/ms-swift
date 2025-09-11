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

pip install -e .
pip install "deepspeed" -U
pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
pip install torchvision
uv pip install flash-attn --no-build-isolation