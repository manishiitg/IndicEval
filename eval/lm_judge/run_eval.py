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

### in this we simply save prompts outputs to a huggingface repo
### i using gemini pro (Free) as LM judge to rate the ouputs
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl


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

    return outputs


def main(args):
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

    if args.use_vllm:
        if args.awq:
            print("Loading model and tokenizer vllm awq...", torch.cuda.device_count())
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="auto",
                tensor_parallel_size=torch.cuda.device_count(),
                # max_num_batched_tokens=4096,
                quantization="AWQ",
                max_model_len=4096,
                dtype="float16",
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

    chat_formatting_function = dynamic_import_function(
        args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset("manishiitg/human_eval")
    test_data = dataset["train"]

    existing_data = []
    api = HfApi()
    if api.repo_exists(repo_id=args.push_output, repo_type="dataset"):
        ds = load_dataset(args.push_output, split="train")
        for row in ds:
            if row["model_name"] == args.model_name_or_path:
                existing_data.append(row)

    prompts = []
    simple_prompts = []
    for i, example in enumerate(test_data):
        messages = json.loads(example["messages"])
        simple_prompts.append("\n\n".join([x["content"] for x in messages]))
        prompt = chat_formatting_function(messages)
        exists = False
        for p in existing_data:
            if p["prompt"] == prompt:
                exists = True
        if not exists:
            prompts.append(prompt)

    outputs = eval_hf_model(args, model, tokenizer,prompts, test_data, args.eval_batch_size)

    final_data = []
    with open(os.path.join(args.save_dir, f"lm_judge_predictions.jsonl"), "w") as fout:
        for example, output, simple_prompt, prompt in zip(test_data, outputs, simple_prompts, prompts):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")
            row = {}
            row["prompt"] = prompt
            row["response"] = output
            row["date"] = date.today().strftime("%m/%d/%Y")
            row["model_name"] = args.model_name_or_path
            row["simple_prompt"] = simple_prompt
            row["judgement_pending"] = True
            row["judgement"] = ""
            row["rating"] = float(-1)
            final_data.append(row)



    api = HfApi()
    if api.repo_exists(repo_id=args.push_output, repo_type="dataset"):
        ds = load_dataset(args.push_output, split="train")
        for row in ds:
            # if row["model_name"] != args.model_name_or_path:
            final_data.append(row)

    dataset = process_and_update_dataset(final_data)
    dataset.push_to_hub(args.push_output, private=False)


def process_and_update_dataset(new_data):
    new_data_formatted = {key: [item[key]
                                for item in new_data] for key in new_data[0].keys()}
    new_dataset_chunk = Dataset.from_dict(new_data_formatted)
    dataset2 = new_dataset_chunk
    return dataset2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=1,
                        help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str,
                        default="/sky-notebook/eval-results/indicwikibio/llama-7B/")
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
        "--max_context_length", type=int, default=768, help="maximum number of tokens in the context passage."
    )
    parser.add_argument("--eval_batch_size", type=int,
                        default=1, help="batch size for evaluation.")
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
    parser.add_argument(
        "--push_output",
        type=str,
        default="manishiitg/llm_judge",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    args = parser.parse_args()
    main(args)
