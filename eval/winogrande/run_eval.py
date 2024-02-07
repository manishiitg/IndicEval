import argparse
import os
import torch
import json
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.utils import get_next_word_predictions, load_hf_lm_and_tokenizer, dynamic_import_function


def format_example(sentence, option1, option2, answer=None):
    prompt = "{sentence}\nOption1: {option1}\nOption2: {option2}".format(sentence=sentence, option1=option1, option2=option2)
    prompt += "\nAnswer:"
    if answer is not None:
        prompt += " {answer}\n\n".format(answer=answer)
    return prompt


def main(args):
    # Load the model and tokenizer
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

    # Load the Winogrande dataset
    dataset = load_dataset('winogrande', 'winogrande_xl')
    test_data = dataset['validation']

    # Preprocess the dataset
    prompts = []
    for i, example in enumerate(test_data):
        prompt_end = format_example(sentence=example['sentence'], option1=example['option1'], option2=example['option2'])
        prompt = prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is: "
            else:
                prompt += " The answer is: "

        prompts.append(prompt)

    # Get the answer for all examples
    answer_choice_ids = [tokenizer.encode(answer_choice, add_special_tokens=False)[-1] for answer_choice in ['true', 'false']]
    pred_indices, all_probs = get_next_word_predictions(
        model,
        tokenizer,
        prompts,
        candidate_token_ids=answer_choice_ids,
        return_token_predictions=False,
        batch_size=args.eval_batch_size,
    )

    # Get the metrics
    # ground_truths = [int(test_data['answer'][i]) - 1 for i in range(10)]
    ground_truths = [int(example['answer']) - 1 for example in test_data]
    predictions = [pred_index for pred_index in pred_indices]
    metrics = {
        "accuracy": accuracy_score(ground_truths, predictions),
        "precision": precision_score(ground_truths, predictions, average="macro"),
        "recall": recall_score(ground_truths, predictions, average="macro"),
        "f1": f1_score(ground_truths, predictions, average="macro"),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--save_dir", type=str, default="/sky-notebook/eval-results/winogrande/llama-7B/")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Path to the tokenizer.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8bit mode.")
    parser.add_argument("--gptq", action="store_true", help="Evaluate a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="Use the chat format for the prompts.")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format", help="Function to create the chat format.")
    args = parser.parse_args()
    main(args)
