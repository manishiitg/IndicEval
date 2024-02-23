mkdir -p data/downloads
mkdir -p data/eval

# Downloads the BLEURT-base checkpoint.
FILE="BLEURT-20.zip"
URL="https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"

if [ ! -f "$FILE" ]; then
    wget -c -O "$FILE" "$URL"
    unzip "$FILE"
fi

# TyDiQA-GoldP dataset
# mkdir -p data/eval/tydiqa
# wget -P data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz
# gzip -df data/eval/tydiqa/tydiqa-v1.0-dev.jsonl.gz