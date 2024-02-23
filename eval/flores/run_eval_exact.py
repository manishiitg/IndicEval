import argparse
import os
import random
from sklearn import metrics
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
from bleurt import score
from transformers import AutoTokenizer
import vllm
import evaluate
exact_match = evaluate.load("exact_match")

lang_map = {
    "asm_Beng": "Assamese",
    "kas_Arab": "Kashmiri",
    "pan_Guru": "Punjabi",
    "ben_Beng": "Bengali",
    "kas_Deva": "Kashmiri",
    "san_Deva": "Sanskrit",
    "brx_Deva": "Bodo",
    "mai_Deva": "Maithili",
    "sat_Olck": "Santali",
    "doi_Deva": "Dogri",
    "mal_Mlym": "Malayalam",
    "snd_Arab": "Sindhi",
    "eng_Latn": "English",
    "mar_Deva": "Marathi",
    "snd_Deva": "Sindhi",
    "gom_Deva": "Konkani",
    "mni_Beng": "Manipuri",
    "tam_Taml": "Tamil",
    "guj_Gujr": "Gujarati",
    "mni_Mtei": "Manipuri",
    "tel_Telu": "Telugu",
    "hin_Deva": "Hindi",
    "npi_Deva": "Nepali",
    "urd_Arab": "Urdu",
    "kan_Knda": "Kannada",
    "ory_Orya": "Odia",
}


def format_example(src_text, src_lang, tgt_lang, tgt_text=None):
    prompt = f"{lang_map[src_lang]}: {src_text}"
    prompt += f"\n{lang_map[tgt_lang]}:"
    if tgt_text is not None:
        prompt += f" {tgt_text}\n\n"
    return prompt


def gen_prompt(dev_data, src_lang, tgt_lang, k=-1):
    prompt = f"Translate the following sentence(s) from {lang_map[src_lang]} into {lang_map[tgt_lang]}.\n\n"
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            prompt += format_example(
                src_text=example[f"sentence_{src_lang}"],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                tgt_text=example[f"sentence_{tgt_lang}"],
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

    return outputs


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

    dataset = load_dataset(
        "facebook/flores", f"{args.src_lang}-{args.tgt_lang}")
    dataset = dataset.map(
        lambda x: {
            f"sentence_{args.src_lang}": x[f"sentence_{args.src_lang}"].strip(),
            f"sentence_{args.tgt_lang}": x[f"sentence_{args.tgt_lang}"].strip(),
        }
    )
    dev_data = dataset["dev"]
    test_data = dataset["devtest"]

    prompts = []
    for i, example in enumerate(test_data):
        k = args.ntrain
        prompt_end = format_example(
            src_text=example[f"sentence_{args.src_lang}"], src_lang=args.src_lang, tgt_lang=args.tgt_lang
        )
        train_prompt = gen_prompt(dev_data, args.src_lang, args.tgt_lang, k)
        prompt = train_prompt + prompt_end

        messages = [{"role": "user", "content": prompt}]
        if args.use_chat_format:
            prompt = chat_formatting_function(messages, tokenizer, args)
        else:
            prompt = "\n\n".join([x["content"] for x in messages])
            
        if prompt[-1] in ["\n", " "]:
            prompt += f"The {lang_map[args.tgt_lang]} translation is: "
        else:
            prompt += f" The {lang_map[args.tgt_lang]} translation is: "

        prompts.append(prompt)

    outputs = eval_hf_model(args, model, tokenizer,
                            prompts, test_data, args.eval_batch_size)

    # debug
    if len(outputs) > 2:
        print(outputs[:2])

    with open(os.path.join(args.save_dir, f"flores_{args.src_lang}_{args.tgt_lang}_predictions.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    # # flush all the GPU memory
    # del model
    # torch.cuda.empty_cache()
    # import gc

    # gc.collect()

    # print("Calculating bleu, chrf, chrf++, bleurt ...")
    # sacrebleu = evaluate.load("sacrebleu")
    # chrf = evaluate.load("chrf")
    # bleurt = score.BleurtScorer(args.bleurt_model_name_or_path)

    # predictions = [output for output in outputs]
    # references = [[example[f"sentence_{args.tgt_lang}"]] for example in test_data]

    # metrics = {
    #     "bleu": sacrebleu.compute(predictions=predictions, references=references)["score"],
    #     "chrf": chrf.compute(predictions=predictions, references=references)["score"],
    #     "chrf2": chrf.compute(predictions=predictions, references=references, word_order=2)["score"],
    #     "bleurt": np.mean(
    #         bleurt.score(candidates=predictions, references=[ref for sublist in references for ref in sublist])
    #     ),
    # }
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")

    # # save results
    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5,
                        help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        choices=list(lang_map.keys()),
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        choices=list(lang_map.keys()),
    )
    parser.add_argument("--save_dir", type=str,
                        default="/sky-notebook/eval-results/flores/llama-7B/")
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
