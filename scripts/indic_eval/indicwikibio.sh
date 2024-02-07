export CUDA_VISIBLE_DEVICES=0


model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on indicwikibio ..."

# 1-shot
python3 -m eval.indicwikibio.run_eval \
    --ntrain 1 \
    --max_context_length 512 \
    --save_dir "/sky-notebook/eval-results/indicwikibio/aditi-v2-1shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on indicwikibio ..."

# 1-shot
python3 -m eval.indicwikibio.run_eval \
    --ntrain 1 \
    --max_context_length 512 \
    --save_dir "/sky-notebook/eval-results/indicwikibio/airavata-1shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
