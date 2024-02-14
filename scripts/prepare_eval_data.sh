mkdir -p data/downloads
mkdir -p data/eval

# Downloads the BLEURT-base checkpoint.
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip

# TyDiQA-GoldP dataset
mkdir -p data/eval/tydiqa
wget -P data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz
unzip tydiqa-v1.0-dev.jsonl.gz