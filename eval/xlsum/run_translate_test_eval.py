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
    load_hf_lm_and_tokenizer,
    dynamic_import_function,
)
from bleurt import score


def trim_context(context, max_context_length, tokenizer):
    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
    if len(tokenized_context) > max_context_length:
        context = tokenizer.decode(
            tokenized_context[:max_context_length], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return context


def format_example(text, lang, summary=None):
    user_prompt = f"{lang.capitalize()} article: {text}"
    assistant_prompt = f"\n{lang.capitalize()} summary:"
    if summary is not None:
        assistant_prompt += f" {summary}"
    messages = [{"role":"user", "content":user_prompt}, {"role":"assistant", "content":assistant_prompt}]
    return messages


def gen_prompt(dev_data, lang, max_context_length, tokenizer, k=-1):
    prompt = f"Summarize the following {"English".capitalize()} article(s) as accurately as possible in few sentences."
    messages = [{"role": "system", "content": prompt}]
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            messages += format_example(
                text=trim_context(example["text"], max_context_length=max_context_length, tokenizer=tokenizer),
                lang=lang,
                summary=example["summary"],
            )
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

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset("Thanmay/xlsum-hi")
    for split in dataset.column_names:
        column_names = dataset[split].column_names
        itv2_column_names = []
        for column_name in column_names:
            if "itv2 hi" in column_name.lower():
                itv2_column_names.append(column_name)
        remove_column_names = [x[8:] for x in itv2_column_names]
        dataset[split] = dataset[split].remove_columns(remove_column_names)
        for itv2_column_name in itv2_column_names:
            dataset[split] = dataset[split].rename_column(itv2_column_name, itv2_column_name[8:])
            
    dataset = dataset.map(lambda x: {"summary": x["summary"].strip()})
    dataset = dataset.map(lambda x: {"text": x["text"].strip()})
    dev_data = dataset["validation"].select(range(min(len(dataset["validation"]), args.n_instances)))
    test_data = dataset["test"].select(range(min(len(dataset["test"]), args.n_instances)))

    prompts = []
    for i, example in enumerate(test_data):
        k = args.ntrain
        prompt_end = format_example(
            text=trim_context(example["text"], args.max_context_length, tokenizer), lang=args.lang
        )
        train_prompt = gen_prompt(dev_data, args.lang, args.max_context_length, tokenizer, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            prompt = chat_formatting_function(prompt)[:-5] # Remove last 5 characters, which is the EOS token (' </s>').
        else:
            prompt = "\n\n".join([x["content"] for x in prompt])

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=50,
        batch_size=args.eval_batch_size,
        stop_id_sequences=None,
    )
    # remove unnecessary space
    outputs = [output.strip().split("\n", "") for output in outputs]

    with open(os.path.join(args.save_dir, f"xlsum_{args.lang}_predictions.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    # flush all the GPU memory
    del model
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    print("Calculating Rouge and BLEURT ...")
    rouge = evaluate.load("rouge")
    bleurt = score.BleurtScorer(args.bleurt_model_name_or_path)

    predictions = [output for output in outputs]
    references = [example["summary"] for example in test_data]

    rouge_metrics = rouge.compute(predictions=predictions, references=references)
    metrics = {
        "rouge1": rouge_metrics["rouge1"],
        "rouge2": rouge_metrics["rouge2"],
        "rougeL": rouge_metrics["rougeL"],
        "bleurt": np.mean(bleurt.score(candidates=predictions, references=references)),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=1, help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang",
        type=str,
        default="hindi",
        choices=["bengali", "english", "gujarati", "hindi", "marathi", "nepali", "punjabi", "tamil", "telugu", "urdu"],
    )
    parser.add_argument("--save_dir", type=str, default="/sky-notebook/eval-results/xlsum/llama-7B/")
    parser.add_argument(
        "--bleurt_model_name_or_path",
        type=str,
        default="/data/jaygala/bleurt/BLEURT-20",
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
        "--max_context_length", type=int, default=768, help="maximum number of tokens in the context passage."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=1000,
        help="if specified, a maximum of n_instances will be used for the evaluation."
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
    args = parser.parse_args()
    main(args)
