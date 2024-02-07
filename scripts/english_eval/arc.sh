# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# -------------------------------------------------------------
#                       ARC-Easy
# -------------------------------------------------------------
model_name_or_path="meta-llama/Llama-2-7b-chat-hf"

echo "evaluating llama-2 7b chat on arc easy..."

# zero-shot
python3 -m eval.arc.run_english_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/llama2-7b-chat-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# 5-shot
python3 -m eval.arc.run_english_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/llama2-7b-chat-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


model_name_or_path="meta-llama/Llama-2-7b-hf"

echo "evaluating llama-2 7b base on arc easy"

# zero-shot
python3 -m eval.arc.run_english_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/llama2-7b-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \

# 5-shot
python3 -m eval.arc.run_english_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/llama2-7b-base-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \


# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
model_name_or_path="meta-llama/Llama-2-7b-chat-hf"

echo "evaluating llama-2 7b chat on arc challenge..."

# zero-shot
python3 -m eval.arc.run_english_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/llama2-7b-chat-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# 5-shot
python3 -m eval.arc.run_english_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/llama2-7b-chat-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


    model_name_or_path="meta-llama/Llama-2-7b-hf"

echo "evaluating llama-2 7b base on arc challenge..."

# zero-shot
python3 -m eval.arc.run_english_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/llama2-7b-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \

# 5-shot
python3 -m eval.arc.run_english_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/llama2-7b-base-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
