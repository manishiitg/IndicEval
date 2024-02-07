# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# -------------------------------------------------------------
#                       MMLU
# -------------------------------------------------------------

model_name_or_path="meta-llama/Llama-2-7b-chat-hf"

echo "evaluating llama-2 7b chat on mmlu ..."

# zero-shot
python3 -m eval.mmlu.run_english_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir "/sky-notebook/eval-results/mmlu/llama2-7b-chat-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# 5-shot
python3 -m eval.mmlu.run_english_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir "/sky-notebook/eval-results/mmlu/llama2-7b-chat-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format





model_name_or_path="meta-llama/Llama-2-7b-hf"

echo "evaluating llama-2 7b base on mmlu ..."

# zero-shot
python3 -m eval.mmlu.run_english_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir "/sky-notebook/eval-results/mmlu/llama2-7b-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \


# 5-shot
python3 -m eval.mmlu.run_english_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir "/sky-notebook/eval-results/mmlu/llama2-7b-chat-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
