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
from transformers import AutoTokenizer
import vllm
import evaluate
exact_match = evaluate.load("exact_match")

choices = ["A", "B"]
choices_map = {True: "A", False: "B"}


def format_example(passage, question, label=None):
    prompt = f"Passage: {passage}\nQuestion: {question.strip()}\n"
    for choice, answer in zip(choices, ["Yes", "No"]):
        prompt += f"{choice}. {answer.strip()}\n"
    prompt += "\nAnswer:"
    if label is not None:
        label = choices_map[label]
        prompt += " {label}\n\n".format(label=label)
    return prompt


def gen_prompt(dev_data, k=-1):
    prompt = f"The following are binary yes/no choice questions (with answers).\n\n"
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            prompt += format_example(
                passage=example["itv2 hi passage"], question=example["itv2 hi question"], label=example["answer"]
            )
    return prompt

@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts, test_data, batch_size=1):
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=512,
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
        answerStr = ""
        answer = row["answer"]
        if answer == "true":
            answerStr = "A. Yes"
        else:
            answerStr = "B. No"
        row["answer_text"] = answerStr
        return row

    # Apply the function to each row of the DataFrame
    test_data = test_data.map(extract_answer)

    targets = test_data['answer_text']

    em_score = exact_match.compute(predictions=outputs, references=targets,
                                   ignore_case=True, ignore_punctuation=True)["exact_match"]
    
    print(f"Exact match : {em_score}")

    predictions = []
    idx = 0
    for row in test_data:
        row = {
            "question": row["question"],
            "model_output": outputs[idx],
            "prediction": targets[idx]
        }
        predictions.append(row)
        print(row)
        idx += 1

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    with open(os.path.join(args.save_dir, f"metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)

    return em_score

def main(args):
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

    if args.use_vllm:
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
    else:
        # print("Loading model and tokenizer hf...")
        # model, tokenizer = load_hf_lm_and_tokenizer(
        #     model_name_or_path=args.model_name_or_path,
        #     tokenizer_name_or_path=args.tokenizer_name_or_path,
        #     load_in_8bit=args.load_in_8bit,
        #     device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
        #     gptq_model=args.gptq,
        #     use_fast_tokenizer=not args.use_slow_tokenizer,
        # )
        raise Exception("only vllm is supported")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset("Thanmay/boolq-translated")
    dev_data = dataset["train"]
    test_data = dataset["validation"]

    prompts = []
    for i, example in enumerate(test_data):
        k = args.ntrain
        prompt_end = format_example(
            passage=example["itv2 hi passage"], question=example["itv2 hi question"], label=None
        )
        train_prompt = gen_prompt(dev_data.shuffle(seed=args.seed), k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
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
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        if include_prompt:
            prompts.append(prompt)

    em_score = eval_hf_model(args, model, tokenizer, prompts, test_data, args.eval_batch_size)    
    print("Em Score", em_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="/sky-notebook/eval-results/boolq-hi/llama-7B/")
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
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--awq",
        action="store_true",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    args = parser.parse_args()
    main(args)