import argparse
import os
import random
import torch
import numpy as np
import json
import evaluate
from datasets import load_dataset
from eval.utils import (
    dynamic_import_function,
)
from bleurt import score
from transformers import AutoTokenizer
import vllm
import evaluate
exact_match = evaluate.load("exact_match")

lang_map = {
    "as": "Assamese",
    "bn": "Bengali",
    "kn": "Kannada",
    "hi": "Hindi",
    "ml": "Malayalam",
    "or": "Oriya",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}


def main(args):
    random.seed(args.seed)

    dataset = load_dataset("Thanmay/indic-wikibio-hi")

    dataset = dataset.map(lambda x: {"infobox": x["infobox"].strip()})
    dataset = dataset.map(lambda x: {"summary": x["summary"].strip()})
    dev_data = dataset["validation"]
    test_data = dataset["test"]

    # Assuming args.save_dir and args.src_lang, args.tgt_lang are defined
    file_path = os.path.join(
        args.save_dir, f"indicwikibio_predictions.jsonl")

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

    print("Calculating Rouge and BLEURT ...")
    rouge = evaluate.load("rouge")
    bleurt = score.BleurtScorer(args.bleurt_model_name_or_path)

    predictions = [output for output in outputs]
    references = [example["summary"] for example in test_data]

    rouge_metrics = rouge.compute(
        predictions=predictions, references=references)
    metrics = {
        "bleurt": np.mean(bleurt.score(candidates=predictions, references=references)),
        "rouge1": rouge_metrics["rouge1"],
        "rouge2": rouge_metrics["rouge2"],
        "rougeL": rouge_metrics["rougeL"],
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=1,
                        help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", type=str, default="hi", choices=["as", "bn", "kn", "hi", "ml", "or", "pa", "ta", "te"]
    )
    parser.add_argument("--save_dir", type=str,
                        default="/sky-notebook/eval-results/indicwikibio/llama-7B/")
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
    parser.add_argument("--eval_batch_size", type=int,
                        default=1, help="batch size for evaluation.")

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
