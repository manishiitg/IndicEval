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

choices = ["A", "B", "C", "D"]


def format_example(ctx, endings, label=None):
    prompt = f"{ctx}\n"
    for choice, ending in zip(choices, endings):
        prompt += f"{choice}. {ending}\n"
    prompt += "\nAnswer:"
    if label is not None:
        prompt += " {label}\n\n".format(label=label)
    return prompt


def gen_prompt(dev_data, k=-1):
    prompt = f"Complete the description with an appropriate ending:\n\n"
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            label = choices[example["label"]]
            prompt += format_example(ctx=example["ctx"], endings=example["endings"], label=label)
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
        label = int(row["label"])
        endings = row["endings"]
        answerStr = endings[label]
        choice = choices[label]
        row["answer_text"] = choice + ". " + answerStr
        return row

    # Apply the function to each row of the DataFrame
    test_data = test_data.map(extract_answer)

    targets = test_data['answer_text']

    predictions = []
    idx = 0
    for row in test_data:
        row = {
            "ctx": row["ctx"],
            "model_output": outputs[idx],
            "prediction": targets[idx]
        }
        predictions.append(row)
        
        idx += 1

    if len(predictions) > 2:
        print(predictions[:2])
    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    em_score = exact_match.compute(predictions=outputs, references=targets,
                                   ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match: {em_score}")

    outputs = [output[0] if len(output) > 0  else "" for output in outputs]
    targets = [target[0] if len(target) > 0  else ""  for target in targets]
    # directly measuring A with A instead of of full option match

    em_score_options = exact_match.compute(predictions=outputs, references=targets,
                                   ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match Only Options: {em_score_options}")

    with open(os.path.join(args.save_dir, f"metrics.json"), "w") as fout:
        json.dump({
            "em_score" : em_score_options,
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

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset(args.dataset)
    dataset = dataset.map(lambda x: {"ctx": x["ctx"].strip()})
    dataset = dataset.map(lambda x: {"endings": [ending.strip() for ending in x["endings"]]})
    test_data = dataset["validation"]
    test_data = test_data.map(lambda x: {"label": int(x["label"])})

    k = args.ntrain
    sample_data = test_data.select(range(k*3))
    prompts = []
    for i, example in enumerate(test_data):
        dev_data = sample_data.filter(lambda x: x["ctx"] != example["ctx"])

        prompt_end = format_example(ctx=example["ctx"], endings=example["endings"], label=None)
        train_prompt = gen_prompt(dev_data.shuffle(seed=args.seed), k)
        prompt = train_prompt + prompt_end

        messages = [{"role": "user", "content": prompt}]
        if args.use_chat_format:
            prompt = chat_formatting_function(messages, tokenizer, args)
        else:
            prompt = "\n\n".join([x["content"] for x in messages])

        
        prompts.append(prompt)



    em_score = eval_hf_model(args, model, tokenizer, prompts, test_data, args.eval_batch_size)    
    print("Em Score", em_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="Rowan/hellaswag")
    parser.add_argument("--save_dir", type=str, default="/sky-notebook/eval-results/hellaswag/llama-7B/")
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
