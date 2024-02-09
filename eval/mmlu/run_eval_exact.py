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
from eval.utils import get_next_word_predictions, load_hf_lm_and_tokenizer, query_openai_chat_model, dynamic_import_function, generate_completions
from datasets import load_dataset
import vllm
import evaluate
choices = ["1", "2", "3", "4"]
exact_match = evaluate.load("exact_match")
from transformers import AutoTokenizer

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    ch = df.iloc[idx, 2]
    answer = df.iloc[idx, -1]
    prompt += "\n{}".format(ch)
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
def eval_hf_model(args, subject, dev_df, test_df, batch_size=1):
    prompts = []
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            "slow" if args.use_slow_tokenizer else "auto",
        )
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for i in tqdm(range(0, test_df.shape[0])):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 4096:
            k -= 1
            if k < 0:
                break
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                    
            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        
        prompts.append(prompt)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        if args.use_vllm:
            if args.awq:
                model = vllm.LLM(
                    model=args.model_name_or_path + "-awq",
                    tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                    tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                    tensor_parallel_size=torch.cuda.device_count(),
                    max_num_batched_tokens=4096,
                    quantization="AWQ",
                )
            else:
                model = vllm.LLM(
                    model=args.model_name_or_path,
                    tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                    tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                    tensor_parallel_size=torch.cuda.device_count(),
                    max_num_batched_tokens=4096,
                )
            sampling_params = vllm.SamplingParams(
                temperature=0,
                max_tokens=512,
                stop=["\n"],
            )
            # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=[[new_line_token]],
                do_sample=False,
            )
    else:
        instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, f"openai_results.jsonl"),
        )
        outputs = [result["output"] for result in results]
    

    def extract_answer(row):
        choices = row['choices'].split('\n')
        answer_index = int(row['answer'])  # Adjust for zero-based indexing
        if answer_index < len(choices):
            return choices[answer_index].strip()  # Remove the number and the bracket
        else:
            return None  # Or handle the case where the answer index is out of range

    # Apply the function to each row of the DataFrame
    test_df['answer_text'] = test_df.apply(extract_answer, axis=1)

    targets = test_df['answer_text'].tolist()

    em_score = exact_match.compute(predictions=outputs, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["question"],
        "answer": example["answer_text"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_df, outputs, predictions)]

    with open(os.path.join(args.save_dir, f"predictions-{subject}.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 
    
    with open(os.path.join(args.save_dir, f"metrics-{subject}.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)

    return em_score


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):
    
    import tiktoken
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [gpt_tokenizer.encode(" " + x)[0] for x in choices]  # be careful, the tokenizer will tokenize " A" and "A" differently.

    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end        
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )
    
    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))]) # dummy probs, just don't want to dig into the openai probs

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def main(args):
    
    if args.data_dir == "data/eval/mmlu_hi_translated":
        ds = load_dataset("manishiitg/cais-mmlu", split="test")
        subjects = []
        for row in ds:
            subjects.append(row["subject"])
        subjects = list(set(subjects))
    else:
        ds = load_dataset("cais/mmlu", split="test", config="all")
        subjects = []
        for row in ds:
            subjects.append(row["subject"])
        subjects = list(set(subjects))

    ds = ds.select(range(100))

    if args.subjects:
        assert all(subj in subjects for subj in args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        
        # try:
        if args.data_dir == "data/eval/mmlu_hi_translated":
            dev_df = pd.DataFrame(load_dataset("manishiitg/cais-mmlu", split="dev"))[: args.ntrain]
            test_df = pd.DataFrame(load_dataset("manishiitg/cais-mmlu", split="test"))
        else:
            # dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[: args.ntrain]
            # test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
            dev_df = pd.DataFrame(load_dataset("cais/mmlu", split="dev"))[: args.ntrain]
            test_df = pd.DataFrame(load_dataset("cais/mmlu", split="test"))
        # except:
        #     continue
        
        if args.n_instances and args.n_instances < test_df.shape[0]:
            test_df = test_df.sample(args.n_instances, random_state=42)

        if args.model_name_or_path:
            em_score = eval_hf_model(args, subject, dev_df, test_df, args.eval_batch_size)
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
        try:
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
        except:
            idxs.append(subcat)
    
    for idx in idxs:
        del subcat_cors[idx]

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subcat_acc": {
                    subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                    for subcat in subcat_cors
                },
                "cat_acc": {
                    cat: np.mean(np.concatenate(cat_cors[cat]))
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
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions."
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
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--awq",
        action="store_false", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)