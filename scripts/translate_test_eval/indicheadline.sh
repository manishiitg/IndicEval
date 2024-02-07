# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=1

model_name_or_path="meta-llama/Llama-2-7b-hf"

# Evaluating Llama-2 7B base model using 0 shot
python -m eval.indicheadline.run_translate_test_eval \
    --ntrain 0 \
    --save_dir /sky-notebook/eval-results/translate_test/llama-2/indicheadline/llama-2-7b-chat-0shot \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path  $model_name_or_path \
    --eval_batch_size 2 \
    --use_chat_format \
     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


model_name_or_path="meta-llama/Llama-2-7b-chat-hf"

# Evaluating Llama-2 7B chat model using 0 shot
python -m eval.indicheadline.run_translate_test_eval \
    --ntrain 0 \
    --save_dir /sky-notebook/eval-results/translate_test/llama-2/indicheadline/llama-2-7b-0shot \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 2
