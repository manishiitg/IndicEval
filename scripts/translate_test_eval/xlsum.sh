


# Llama-2 7B Base Checkpoint
model_name_or_path="meta-llama/Llama-2-7b-hf"
echo "evaluating llama 2 7b base on xlsum ..."

# 1-shot
python3 -m eval.xlsum.run_translate_test_eval \
    --ntrain 1 \
    --save_dir "/sky-notebook/eval-results/translate_test/llama-2/xlsum-hin/llama-2-7b-1shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


# Llama-2 7B Chat Checkpoint

model_name_or_path="meta-llama/Llama-2-7b-chat-hf"
echo "evaluating llama 2 7b chat on xlsum ..."

# 1-shot
python3 -m eval.xlsum.run_translate_test_eval \
    --ntrain 1 \
    --save_dir "/sky-notebook/eval-results/translate_test/llama-2/xlsum-hin/llama-2-7b-chat-1shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format
