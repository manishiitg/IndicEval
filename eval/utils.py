import torch
import tqdm
from importlib import import_module
from transformers import StoppingCriteria


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(
            stop_id_sequences[0], list
        ), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    add_special_tokens=True,
    disable_tqdm=False,
    **generation_kwargs,
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            if True:
                batch_outputs = model.generate(
                    input_ids=batch_input_ids,
                    # attention_mask=attention_mask,
                    # stopping_criteria=[KeyWordsCriteria(stop_id_sequences)]
                    # if stop_id_sequences
                    # else None,
                    # **generation_kwargs,
                )
                batch_outputs = tokenizer.batch_decode(
                    batch_outputs, skip_special_tokens=True)
                
                print(batch_outputs)
            else:    
                batch_outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    stopping_criteria=[KeyWordsCriteria(stop_id_sequences)]
                    if stop_id_sequences
                    else None,
                    **generation_kwargs,
                )

                # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
                # so some outputs still have the stop sequence, which we need to remove.
                if stop_id_sequences:
                    for output_idx in range(batch_outputs.shape[0]):
                        for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                            if any(
                                batch_outputs[
                                    output_idx, token_idx: token_idx + len(stop_sequence)
                                ].tolist()
                                == stop_sequence
                                for stop_sequence in stop_id_sequences
                            ):
                                batch_outputs[output_idx,
                                            token_idx:] = tokenizer.pad_token_id
                                break

                # remove the prompt from the output
                # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
                # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
                # space is important for some tasks (e.g., code completion).
                batch_outputs = tokenizer.batch_decode(
                    batch_outputs, skip_special_tokens=True)
                batch_prompts = tokenizer.batch_decode(
                    batch_input_ids, skip_special_tokens=True)
                # duplicate the prompts to match the number of return sequences
                batch_prompts = [
                    prompt for prompt in batch_prompts for _ in range(num_return_sequences)
                ]
                batch_generations = [
                    output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
                ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * \
                len(batch_prompts) * num_return_sequences

        generations += batch_generations

        for prompt, generation in zip(batch_prompts, batch_generations):
            print("========")
            print(prompt)
            print("--------")
            print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert (
        len(generations) == len(prompts) * num_return_sequences
    ), "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def get_next_word_predictions(
    model,
    tokenizer,
    prompts,
    candidate_token_ids=None,
    batch_size=1,
    return_token_predictions=False,
    add_special_tokens=True,
    disable_tqdm=False,
):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        # if model.device.type == "cuda":
        batch_input_ids = batch_input_ids.cuda()
        attention_mask = attention_mask.cuda()

        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[
            :, -1, :
        ]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(
                    candidate_token_ids)
                batch_predictions = [candidate_tokens[idx]
                                     for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(
                    batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(
        prompts
    ), "number of predictions should be equal to number of prompts"
    return predictions, probs


def load_hf_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    torch_dtype="auto",
    load_in_8bit=False,
    convert_to_half=False,
    gptq_model=False,
    use_fast_tokenizer=True,
    padding_side="left",
    awq_model=False,
    is_aya_model=False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM, AutoModelForSeq2SeqLM

    if is_aya_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, device_map=device_map, load_in_8bit=load_in_8bit)
    elif awq_model:
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_quantized(
            model_name_or_path, fuse_layers=True)
        # model = model_wrapper.model

    elif gptq_model:
        from auto_gptq import AutoGPTQForCausalLM

        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map, load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map=device_map, torch_dtype=torch_dtype
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype
            )
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()

    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, use_fast=use_fast_tokenizer
        )
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings
    # to avoid wrong embedding index.
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print(
            "Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(
                model.config.max_position_embeddings
            )
        )

    return model, tokenizer


def dynamic_import_function(function_path):
    """
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    """
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
