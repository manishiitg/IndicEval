import argparse
import random
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(args):
    random.seed(args.seed)
    AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)
    AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="cpu"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=0,
                        help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
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
