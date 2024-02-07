export CUDA_VISIBLE_DEVICES=0


model_name_or_path="meta-llama/Llama-2-7b-chat-hf"

echo "evaluating llama-2 7b chat on winogrande ..."

# zero-shot
python3 -m eval.winogrande.run_english_eval \
    --ntrain 0 \
    --save_dir "/sky-notebook/eval-results/winogrande/llama2-7b-chat-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


model_name_or_path="meta-llama/Llama-2-7b-hf"

echo "evaluating llama-2 7b base on winogrande ..."

# zero-shot
python3 -m eval.winogrande.run_english_eval \
    --ntrain 0 \
    --save_dir "/sky-notebook/eval-results/winogrande/llama2-7b-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
