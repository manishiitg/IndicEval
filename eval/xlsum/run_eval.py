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


def trim_context(context, max_context_length, tokenizer):
    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
    if len(tokenized_context) > max_context_length:
        context = tokenizer.decode(
            tokenized_context[:max_context_length], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return context


def format_example(text, summary=None):
    user_prompt = f"article: {text}"
    assistant_prompt = f"\nsummary:"
    if summary is not None:
        assistant_prompt += f" {summary} \n\n"

    return user_prompt + assistant_prompt


def gen_prompt(dev_data, text, max_context_length, tokenizer, k=-1):
    prompt = f"Summarize the following article(s) as accurately as possible in few sentences."
    messages = [{"role": "system", "content": prompt}]

    user = ""
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            user += format_example(
                text=trim_context(
                    example["text"], max_context_length=max_context_length, tokenizer=tokenizer),
                summary=example["summary"],
            )

    user += format_example(
        text=trim_context(
            text, max_context_length=max_context_length, tokenizer=tokenizer),
    )

    messages.append({"role": "user", "content": user})
    return messages


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
            max_model_len=4096,
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(
        args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset("csebuetnlp/xlsum", args.lang)

    dataset = dataset.map(lambda x: {"summary": x["summary"].strip()})
    dataset = dataset.map(lambda x: {"text": x["text"].strip()})
    dev_data = dataset["validation"].select(
        range(min(len(dataset["validation"]), args.n_instances)))
    test_data = dataset["test"].select(
        range(min(len(dataset["test"]), args.n_instances)))

    prompts = []
    for i, example in enumerate(test_data):
        k = args.ntrain
        prompt = gen_prompt(
            dev_data, example["text"], args.max_context_length, tokenizer, k)

        if args.use_chat_format:
            prompt = chat_formatting_function(prompt, tokenizer, args)
        else:
            prompt = "\n\n".join([x["content"] for x in prompt])

        prompts.append(prompt)

    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=2048,
        stop=["<|im_end|>"],
    )
    # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
    generations = model.generate(prompts, sampling_params)

    prompt_to_output = {
        g.prompt: g.outputs[0].text.strip() for g in generations
    }
    outputs = [prompt_to_output[prompt]
               if prompt in prompt_to_output else "" for prompt in prompts]

    with open(os.path.join(args.save_dir, f"xlsum_predictions.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    # # flush all the GPU memory
    # del model
    # torch.cuda.empty_cache()
    # import gc

    # gc.collect()

    # print("Calculating BLEURT ...")
    # bleurt = score.BleurtScorer(args.bleurt_model_name_or_path)

    # predictions = [output for output in outputs]
    # references = [example["summary"] for example in test_data]

    # metrics = {
    #     "bleurt": np.mean(bleurt.score(candidates=predictions, references=references)),
    # }
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")

    # # save results
    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     json.dump(metrics, fout, indent=4)


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
