from transformers import AutoTokenizer
import argparse
import os
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
from eval.mmlu.categories import subcategories, categories
from eval.utils import dynamic_import_function
from datasets import load_dataset
import vllm
import evaluate
exact_match = evaluate.load("exact_match")
choices = ["1", "2", "3", "4"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    ch = df.iloc[idx, 2]
    if isinstance(ch, str):  # in hi data is string but in en its list
        ch = ch.split('\n')

    answer = df.iloc[idx, -1]
    for ix, opt in enumerate(ch):
        if len(opt) == 0:
            continue
        option_str = "{})".format(ix)
        if option_str not in opt:
            prompt += "\n{} {}".format(option_str, opt)
        else:
            prompt += "\n{}".format(opt)

    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(answer)
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i) + "\n"
    return prompt


@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1):
    prompts = []
    chat_formatting_function = dynamic_import_function(
        args.chat_formatting_function) if args.use_chat_format else None
    for i in tqdm(range(0, test_df.shape[0])):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, tokenizer, args)

        tokenized_prompt = tokenizer(
            prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 4096:
            k -= 1
            if k < 0:
                break
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, tokenizer, args)

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False).input_ids

        prompts.append(prompt)

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

        if isinstance(row['choices'], str):  # in hi data is string but in en its list
            choices = row['choices'].split('\n')
        else:
            choices = row["choices"]
        answer_index = int(row['answer'])  # Adjust for zero-based indexing
        if answer_index < len(choices):
            # Remove the number and the bracket
            choice = choices[answer_index].strip()
            option_str = "{})".format(answer_index)
            if option_str not in choice:
                choice = "{} {}".format(option_str, choice)
            return choice
        else:
            return None  # Or handle the case where the answer index is out of range

    # Apply the function to each row of the DataFrame
    test_df['answer_text'] = test_df.apply(extract_answer, axis=1)

    targets = test_df['answer_text'].tolist()
    predictions = []
    idx = 0
    for index, row in test_df.iterrows():
        row = {
            "question": row["question"],
            "subject": subject,
            "model_output": outputs[idx],
            "prediction": targets[idx]
        }
        predictions.append(row)

        idx += 1

    if len(predictions) > 2:
        print(predictions[:2])
    with open(os.path.join(args.save_dir, f"predictions-{subject}.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    em_score = exact_match.compute(predictions=outputs, references=targets,
                                   ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match: {subject} {em_score}")

    outputs = [output[0] if len(output) > 0 else "" for output in outputs]
    targets = [target[0] if len(target) > 0 else "" for target in targets]
    # directly measuring A with A instead of of full option match

    em_score_options = exact_match.compute(predictions=outputs, references=targets,
                                           ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match Only Options: {subject} {em_score_options}")

    with open(os.path.join(args.save_dir, f"metrics-{subject}.json"), "w") as fout:
        json.dump({
            "em_score": em_score_options,
            "exact_match": em_score,
        }, fout, indent=4)

    return em_score


def main(args):

    if args.data_dir == "data/eval/mmlu_hi_translated":
        ds = load_dataset("manishiitg/cais-mmlu", split="test")
        subjects = []
        for row in ds:
            subjects.append(row["subject"])
        subjects = list(set(subjects))
    else:
        ds = load_dataset("cais/mmlu", "all", split="test",
                          trust_remote_code=True)
        subjects = []
        for row in ds:
            subjects.append(row["subject"])
        subjects = list(set(subjects))

    if args.subjects:
        assert all(
            subj in subjects for subj in args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

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
    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):

        if args.data_dir == "data/eval/mmlu_hi_translated":
            dev_df = pd.DataFrame(load_dataset(
                "manishiitg/cais-mmlu", split="dev"))[: args.ntrain]
            test_df = pd.DataFrame(load_dataset(
                "manishiitg/cais-mmlu", split="test").filter(lambda x: x["subject"] == subject))
        else:
            dev_df = pd.DataFrame(load_dataset(
                "cais/mmlu", subject, split="dev", trust_remote_code=True))[: args.ntrain]
            test_df = pd.DataFrame(load_dataset(
                "cais/mmlu", subject, split="test", trust_remote_code=True))

        if args.n_instances and args.n_instances < test_df.shape[0]:
            test_df = test_df.sample(args.n_instances, random_state=42)

        if args.model_name_or_path:
            em_score = eval_hf_model(
                args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)
        else:
            raise Exception("unsupported flow")

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(em_score)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(em_score)
        all_cors.append(em_score)

    # In IndicMMLU, we exclude math specific subjects where the translation outputs are not good.
    idxs = []
    for subcat in subcat_cors:
        if len(subcat_cors[subcat]) > 0:
            try:
                subcat_acc = np.mean(subcat_cors[subcat])
                print(
                    "Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
            except:
                idxs.append(subcat)

    # for idx in idxs:
    #     del subcat_cors[idx]

    for cat in cat_cors:
        if len(cat_cors[cat]) > 0:
            cat_acc = np.mean(cat_cors[cat])
            print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(all_cors)
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subcat_acc": {
                    subcat: np.mean(subcat_cors[subcat])
                    for subcat in subcat_cors
                },
                "cat_acc": {
                    cat: np.mean(cat_cors[cat])
                    for cat in cat_cors
                },
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntrain",
        type=int,
        default=5
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/sky-notebook/eval-results/mmlu/llama-7B/"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )

    parser.add_argument(
        "--subjects",
        nargs="*",
        help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        help="if specified, a maximum of n_instances per subject will be used for the evaluation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_by_template",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )

    parser.add_argument(
        "--awq",
        action="store_true",
        help="Load model as awq"
    )
    args = parser.parse_args()
    main(args)
