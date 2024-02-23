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
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from eval.utils import (
    get_next_word_predictions,
    load_hf_lm_and_tokenizer,
    dynamic_import_function,
)

choices = ["true", "unknown", "false"]


def format_example(premise, hypothesis, label=None):
    user_prompt = "Premise: {premise}\nHypothesis: {hypothesis}".format(
        premise=premise, hypothesis=hypothesis)
    assistant_prompt = "Answer:"
    if label is not None:
        assistant_prompt += " {label}".format(label=label)
    messages = [{"role": "user", "content": user_prompt}, {
        "role": "assistant", "content": assistant_prompt}]
    return messages


def gen_prompt(dev_data, k=-1):
    prompt = f"Answer whether the hypothesis is more likely to be true (entailment), false (contradiction), or unknown (neutral) based on the given premise."
    messages = [{"role": "system", "content": prompt}]
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            label = choices[example["label"]]
            messages += format_example(
                premise=example["premise"], hypothesis=example["hypothesis"], label=label)
    return messages


def main(args):
    random.seed(args.seed)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(
        args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset("Thanmay/indic-xnli")
    for split in dataset.column_names:
        column_names = dataset[split].column_names
        itv2_column_names = []
        for column_name in column_names:
            if "itv2 hi" in column_name.lower():
                itv2_column_names.append(column_name)
        remove_column_names = [x[8:] for x in itv2_column_names]
        dataset[split] = dataset[split].remove_columns(remove_column_names)
        for itv2_column_name in itv2_column_names:
            dataset[split] = dataset[split].rename_column(
                itv2_column_name, itv2_column_name[8:])

    dataset = dataset.map(lambda x: {"premise": x["premise"].strip()})
    dataset = dataset.map(lambda x: {"hypothesis": x["hypothesis"].strip()})
    dev_data = dataset["validation"]
    test_data = dataset["test"]

    prompts = []
    for i, example in enumerate(test_data):
        k = args.ntrain
        prompt_end = format_example(
            premise=example["premise"], hypothesis=example["hypothesis"])
        train_prompt = gen_prompt(dev_data, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            # Remove last 5 characters, which is the EOS token (' </s>').
            prompt = chat_formatting_function(prompt, tokenizer, args)
        else:
            prompt = "\n\n".join([x["content"] for x in prompt])

        tokenized_prompt = tokenizer(
            prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        include_prompt = True
        while len(tokenized_prompt) > 4096:
            k -= 1
            if k < 0:
                include_prompt = False
                break
            train_prompt = gen_prompt(dev_data, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                prompt = chat_formatting_function(prompt, tokenizer, args)
            else:
                prompt = "\n\n".join([x["content"] for x in prompt])

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False).input_ids
        if include_prompt:
            prompts.append(prompt)

    # get the answer for all examples
    # adding a prefix space here, as that's expected from the prompt
    # TODO: should raise a warning if this returns more than one token
    answer_choice_ids = [tokenizer.encode(
        answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model,
        tokenizer,
        prompts,
        candidate_token_ids=answer_choice_ids,
        return_token_predictions=False,
        batch_size=args.eval_batch_size,
    )

    # get the metrics
    ground_truths = [example["label"] for example in test_data]
    predictions = [pred_index for pred_index in pred_indices]
    metrics = {
        "accuracy": accuracy_score(ground_truths, predictions),
        "precision": precision_score(ground_truths, predictions, average="macro"),
        "recall": recall_score(ground_truths, predictions, average="macro"),
        "f1": f1_score(ground_truths, predictions, average="macro"),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", type=str, default="hi", choices=["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "te"]
    )
    parser.add_argument("--save_dir", type=str,
                        default="/sky-notebook/eval-results/indicxnli/llama-7B/")
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
    args = parser.parse_args()
    main(args)
