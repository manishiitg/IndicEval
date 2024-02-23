import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
import evaluate
from datasets import load_dataset
from eval.utils import (
    generate_completions,
    dynamic_import_function,
)
from transformers import AutoTokenizer
from bleurt import score
import vllm


def main(args):
    random.seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = load_dataset("csebuetnlp/xlsum", args.lang)

    dataset = dataset.map(lambda x: {"summary": x["summary"].strip()})
    dataset = dataset.map(lambda x: {"text": x["text"].strip()})
    dev_data = dataset["validation"].select(
        range(min(len(dataset["validation"]), args.n_instances)))
    test_data = dataset["test"].select(
        range(min(len(dataset["test"]), args.n_instances)))

    # Assuming args.save_dir and args.src_lang, args.tgt_lang are defined
    file_path = os.path.join(
        args.save_dir, f"xlsum_predictions.jsonl")

    # Open the file for reading
    with open(file_path, "r") as fin:
        # Read all lines from the file and parse them into a list of dictionaries
        data_list = [json.loads(line) for line in fin]

    # Now you can work with the data_list as needed
    outputs = []
    for data in data_list:
        example = data
        prediction_text = example["prediction_text"]
        # Do something with the data...
        outputs.append(prediction_text)

    print("Calculating BLEURT ...")
    bleurt = score.BleurtScorer(args.bleurt_model_name_or_path)

    predictions = [output for output in outputs]
    references = [example["summary"] for example in test_data]

    metrics = {
        "bleurt": np.mean(bleurt.score(candidates=predictions, references=references)),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=0,
                        help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", type=str, default="hindi", choices=["hindi", "english"]
    )
    parser.add_argument("--save_dir", type=str,
                        default="/sky-notebook/eval-results/xlsum/llama-7B/")
    parser.add_argument(
        "--bleurt_model_name_or_path",
        type=str,
        default="./BLEURT-20",
        help="bleurt model to load for evaluation.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--max_context_length", type=int, default=3750, help="maximum number of tokens in the context passage."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=1000,
        help="if specified, a maximum of n_instances will be used for the evaluation."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_by_template",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    parser.add_argument(
        "--awq",
        action="store_true",
        help="Load model as awq"
    )
    args = parser.parse_args()
    main(args)
