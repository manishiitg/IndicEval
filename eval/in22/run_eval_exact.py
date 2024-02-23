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
from transformers import AutoTokenizer
import vllm
import evaluate
exact_match = evaluate.load("exact_match")

lang_map = {
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "brx_Deva": "Bodo",
    "doi_Deva": "Dogri",
    "eng_Latn": "English",
    "guj_Gujr": "Gujarati",
    "gom_Deva": "Konkani",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mni_Mtei": "Manipuri",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "snd_Deva": "Sindhi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}


def format_example(src_text, src_lang, tgt_lang, tgt_text=None):
    prompt = f"{lang_map[src_lang]}: {src_text}"
    prompt += f"\n{lang_map[tgt_lang]}:"
    if tgt_text is not None:
        prompt += f" {tgt_text}\n\n"
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

    dataset = load_dataset(args.dataset, f"{args.src_lang}-{args.tgt_lang}")
    dataset = dataset.map(
        lambda x: {
            f"sentence_{args.src_lang}": x[f"sentence_{args.src_lang}"].strip(),
            f"sentence_{args.tgt_lang}": x[f"sentence_{args.tgt_lang}"].strip(),
        }
    )
    test_data = dataset["gen"] if args.dataset == "ai4bharat/IN22-Gen" else dataset["conv"]

    prompts = []
    k = args.ntrain
    for i, example in enumerate(test_data):

        messages = [
            {"role": "system", "content": f"Translate the following sentence(s) from {lang_map[args.src_lang]} into {lang_map[args.tgt_lang]}.\n\n"}]

        prompt = format_example(
            src_text=example[f"sentence_{args.src_lang}"], src_lang=args.src_lang, tgt_lang=args.tgt_lang
        )
        if prompt[-1] in ["\n", " "]:
            prompt += f"The {lang_map[args.tgt_lang]} translation is: "
        else:
            prompt += f" The {lang_map[args.tgt_lang]} translation is: "

        if args.use_chat_format:
            prompt = chat_formatting_function(messages, tokenizer, args)
        else:
            prompt = "\n\n".join([x["content"] for x in messages])

        prompts.append(prompt)

    outputs = eval_hf_model(args, model, tokenizer,
                            prompts, test_data, args.eval_batch_size)

    with open(os.path.join(args.save_dir, f"in22_{args.src_lang}_{args.tgt_lang}_predictions.jsonl"), "w") as fout:
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
    # references = [[example[f"sentence_{args.tgt_lang}"]]
    #               for example in test_data]
    # if len(predictions) > 2:
    #     print(predictions[:2])

    # metrics = {
    #     "bleu": sacrebleu.compute(predictions=predictions, references=references)["score"],
    #     "chrf": chrf.compute(predictions=predictions, references=references)["score"],
    #     "chrf2": chrf.compute(predictions=predictions, references=references, word_order=2)["score"],
    #     "bleurt": np.mean(
    #         bleurt.score(candidates=predictions, references=[
    #                      ref for sublist in references for ref in sublist])
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
        "--dataset", type=str, default="ai4bharat/IN22-Gen", choices=["ai4bharat/IN22-Gen", "ai4bharat/IN22-Conv"]
    )
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
                        default="/sky-notebook/eval-results/in22-gen/llama-7B/")
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
