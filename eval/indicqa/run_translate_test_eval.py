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
    dynamic_import_function,
)
from transformers import AutoTokenizer
import vllm
import evaluate
exact_match = evaluate.load("exact_match")

templates = {
    "with_context": (
        "Answer the following question based on the information in the given passage. Select answer text from passage only. If you cannot answer based on passage reply 'unanswerable'",
        "Passage:",
        "Question:",
        "Answer:",
    ),
    "no_context": (
        "Answer the following question. If you don't know the answer reply 'unanswerable'",
        "Question:",
        "Answer:"
    ),
}


def trim_context(context, max_context_length, tokenizer):
    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
    if len(tokenized_context) > max_context_length:
        context = tokenizer.decode(
            tokenized_context[:max_context_length], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return context


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
        answerStr = row["answers"]["text"][0]
        if answerStr == "":
            answerStr = 'unanswerable'
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

    if len(prompts) > 2:
        print(prompts[:2])

    if len(predictions) > 2:
        print(predictions[:2])

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    em_score = exact_match.compute(predictions=outputs, references=targets,
                                   ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match: {em_score}")
    with open(os.path.join(args.save_dir, f"metrics.json"), "w") as fout:
        json.dump({
            "em_score": em_score,
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

    dataset = load_dataset("ai4bharat/IndicQA", "indicqa.hi")
    dataset = dataset.map(lambda x: {"context": x["context"].strip()})
    dataset = dataset.map(lambda x: {"question": x["question"].strip()})
    test_data = dataset["test"]
    args.no_context = False
    k = args.ntrain
    sample_data = test_data.select(range(k*3))
    prompts = []
    for i, example in enumerate(test_data):
        dev_data = sample_data.filter(
            lambda x: x["question"] != example["question"])  # .shuffle(args.seed)

        if args.no_context:
            prompt, q_template, a_template = templates["no_context"]
            p_template = ""
        else:
            prompt, p_template, q_template, a_template = templates["with_context"]

        train_prompt = [{"role": "system", "content": prompt}]
        # if k > 0:
        #     exemplars = dev_data.select(range(k))
        #     for dev_example in exemplars:
        #         answer = (
        #             "unanswerable" if dev_example["answers"]["text"][0] == "" else dev_example["answers"]["text"][0]
        #         )
        #         if args.no_context:
        #             user_prompt = q_template + " " + dev_example["question"] + "\n"
        #             assistant_prompt = a_template + " " + answer
        #         else:
        #             user_prompt = p_template
        #             + " "
        #             + trim_context(dev_example["context"], args.max_context_length, tokenizer)
        #             + "\n"
        #             + q_template
        #             + " "
        #             + dev_example["question"]
        #             + "\n"
        #             assistant_prompt = a_template + " " + answer
        #         train_prompt.extend(
        #             [{"role":"user", "content":user_prompt + "\n" + assistant_prompt},]
        #         )
        if args.no_context:
            user_prompt = q_template + " " + format(example["question"]) + "\n"
        else:
            user_prompt = (
                p_template
                + " "
                + format(trim_context(example["context"],
                         args.max_context_length, tokenizer))
                + "\n"
                + q_template
                + " "
                + format(example["question"])
                + "\n"
            )
        assistant_prompt = a_template
        prompt_end = [
            {"role": "user", "content": user_prompt + "\n" + assistant_prompt}]

        prompt = train_prompt + prompt_end
        if args.use_chat_format:
            prompt = chat_formatting_function(prompt, tokenizer, args)
        else:
            prompt = "\n\n".join([x["content"] for x in prompt])

        prompts.append(prompt)

    em_score = eval_hf_model(args, model, tokenizer,
                             prompts, test_data, args.eval_batch_size)
    print("Em Score", em_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=1,
                        help="number of examples to use for few-shot evaluation.")
    parser.add_argument(
        "--no_context", action="store_false", help="If given, we're evaluating a model without the gold context passage."
    )
    parser.add_argument(
        "--max_context_length", type=int, default=3750, help="maximum number of tokens in the context passage."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", type=str, default="hi", choices=["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
    )
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
