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
from eval.utils import (
    dynamic_import_function,
)
from transformers import AutoTokenizer
import vllm
import evaluate
exact_match = evaluate.load("exact_match")

choices = ["A", "B", "C", "D", "E"]
num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def format_example(question, answers, label=None):
    prompt = f"{question.strip()}\n"
    for choice, answer in zip(choices, answers):
        prompt += f"{choice}. {answer.strip()}\n"
    prompt += "\nAnswer:"
    if label is not None:
        label = num_to_letter.get(label, label)
        prompt += " {label}\n\n".format(label=label)
    return prompt


def gen_prompt(dev_data, k):
    prompt = f"The following are multiple choice questions (with answers) about science.\n\n"
    # if k > 0:
    #     exemplars = dev_data.select(range(k))
    #     for example in exemplars:
    #         prompt += format_example(
    #             question=example["question"], answers=example["choices"]["text"], label=example["answerKey"]
    #         )
    return prompt


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts, test_data, batch_size=1):
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
    generations = model.generate(prompts, sampling_params)

    prompt_to_output = {
        g.prompt: g.outputs[0].text.strip() for g in generations
    }
    outputs = [prompt_to_output[prompt]
               if prompt in prompt_to_output else "" for prompt in prompts]

    def extract_answer(row):
        choices = row['choices']
        answerKey = row['answerKey']
        answerStr = ""
        for idx, l in enumerate(choices["label"]):
            if l == answerKey:
                answerStr = l + ". " + choices["text"][idx]

        row["answer_text"] = answerStr
        return row

    # Apply the function to each row of the DataFrame
    test_data = test_data.map(extract_answer)

    targets = test_data['answer_text']

    predictions = []
    idx = 0
    for row in test_data:
        row = {
            "question": row["question"],
            "id": row["id"],
            "model_output": outputs[idx],
            "prediction": targets[idx]
        }
        predictions.append(row)

        idx += 1

    # debug
    if len(predictions) > 2:
        print(predictions[:2])
    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    em_score = exact_match.compute(predictions=outputs, references=targets,
                                   ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")

    outputs = [output[0] if len(output) > 0 else "" for output in outputs]
    targets = [target[0] if len(target) > 0 else "" for target in targets]
    # directly measuring A with A instead of of full option match

    em_score_options = exact_match.compute(predictions=outputs, references=targets,
                                           ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match Only Options: {em_score_options}")

    with open(os.path.join(args.save_dir, f"metrics.json"), "w") as fout:
        json.dump({
            "em_score": em_score_options,
            "exact_match": em_score,
        }, fout, indent=4)

    return em_score


def main(args):
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

    if args.awq:
        print("Loading model and tokenizer vllm awq...")
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            # max_num_batched_tokens=4096,
            quantization="AWQ",
            max_model_len=4096,
        )
    else:
        print("Loading model and tokenizer vllm...")
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            # max_num_batched_tokens=4096,
            max_model_len=4096,
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(
        args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset(args.dataset, f"ARC-{args.subset.capitalize()}")
    dev_data = dataset["validation"]
    test_data = dataset["test"]

    prompts = []
    k = args.ntrain
    for i, example in enumerate(test_data):
        prompt_end = format_example(
            question=example["question"], answers=example["choices"]["text"], label=None)
        train_prompt = gen_prompt(dev_data.shuffle(seed=args.seed), k)
        prompt = train_prompt + prompt_end

        messages = [{"role": "user", "content": prompt}]
        if args.use_chat_format:
            prompt = chat_formatting_function(messages, tokenizer, args)
        else:
            prompt = "\n\n".join([x["content"] for x in messages])

        prompts.append(prompt)

    eval_hf_model(args, model, tokenizer, prompts,
                  test_data, args.eval_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="easy",
                        choices=["ai2_arc", "ai4bharat/ai2_arc-hi"])
    parser.add_argument("--subset", type=str, default="easy",
                        choices=["easy", "challenge"])
    parser.add_argument("--save_dir", type=str,
                        default="/sky-notebook/eval-results/mmlu/llama-7B/")
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
        "--n_instances",
        type=int,
        help="if specified, a maximum of n_instances per subject will be used for the evaluation.",
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
