from tqdm import tqdm
import argparse
import os
import random
import torch
import json
from datasets import load_dataset
from eval.utils import (
    dynamic_import_function,
)
from huggingface_hub import HfApi
from transformers import AutoTokenizer
import vllm
from datasets import Dataset
from datetime import date
import torch
import re

# in this we simply save prompts outputs to a huggingface repo
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl
prompt = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. 
Begin your evaluation by providing a short explanation. Be as objective as possible. 

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]

After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: 
<explanation_for_rating>

Overall Rating: <overall_rating>
"""

rating_pattern = r'Overall Rating: (\d+(?:\.\d+)?)'


def get_rating(output):
    match = re.search(rating_pattern, output)

    # If a match is found, extract the rating
    if match:
        rating = match.group(1)
        return rating
    else:
        raise ValueError()


def get_lm_judge_rating_prompt(question, answer):
    prompt_1 = prompt.replace("{question}", question)
    prompt_1 = prompt_1.replace("{answer}", answer)
    return prompt_1


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, prompts):
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

    return outputs


def main(args):

    model_name_or_path = "Qwen/Qwen1.5-72B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print("Loading model and tokenizer vllm awq...")
    model = vllm.LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        # max_num_batched_tokens=4096,
        # quantization="AWQ",
        max_model_len=4096,
        dtype="float16",
        gpu_memory_utilization=.8

    )

    ds = load_dataset("manishiitg/llm_judge", split="train")

    final_data = []
    for row in ds:
        final_data.append(row)

    prompts = []
    completed_data = []
    pending_data = []
    for row in tqdm(final_data):
        if row["judgement_pending"] or True:
            instruction = row["simple_prompt"]
            answer = row["response"]
            prompt = get_lm_judge_rating_prompt(
                question=instruction, answer=answer)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)
            pending_data.append(row)
        else:
            completed_data.append(row)

    outputs = eval_hf_model(args, model, tokenizer, prompts)

    for idx, text in enumerate(outputs):
        print("input row")
        print(row)
        print("text")
        print(text)
        try:
            rating = get_rating(text)
            pending_data[idx]["judgement"] = text
            pending_data[idx]["rating"] = float(rating)
            pending_data[idx]["judgement_pending"] = False
        except ValueError:
            pending_data[idx]["judgement"] = text
            pending_data[idx]["rating"] = -1
            pending_data[idx]["judgement_pending"] = False
            print("text failed", text, -1)

    final_data = pending_data + completed_data
    dataset = process_and_update_dataset(final_data)
    dataset.push_to_hub("manishiitg/llm_judge-v2", private=False)


def process_and_update_dataset(new_data):
    new_data_formatted = {key: [item[key]
                                for item in new_data] for key in new_data[0].keys()}
    new_dataset_chunk = Dataset.from_dict(new_data_formatted)
    dataset2 = new_dataset_chunk
    return dataset2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push_output",
        type=str,
        default="manishiitg/llm_judge",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    args = parser.parse_args()
    main(args)
