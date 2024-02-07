export CUDA_VISIBLE_DEVICES=0


model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on implicit hate ..."

# zero-shot
python3 -m eval.implicit_hate.run_eval \
    --ntrain 0 \
    --save_dir "/sky-notebook/eval-results/implicit_hate/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8

# 5-shot
python3 -m eval.implicit_hate.run_eval \
    --ntrain 5 \
    --save_dir "/sky-notebook/eval-results/implicit_hate/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on implicit hate ..."

# zero-shot
python3 -m eval.implicit_hate.run_eval \
    --ntrain 0 \
    --save_dir "/sky-notebook/eval-results/implicit_hate/aditi-v1-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8

# 5-shot
python3 -m eval.implicit_hate.run_eval \
    --ntrain 5 \
    --save_dir "/sky-notebook/eval-results/implicit_hate/aditi-v1-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on implicit hate ..."

# zero-shot
python3 -m eval.implicit_hate.run_eval \
    --ntrain 0 \
    --save_dir "/sky-notebook/eval-results/implicit_hate/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# 5-shot
python3 -m eval.implicit_hate.run_eval \
    --ntrain 5 \
    --save_dir "/sky-notebook/eval-results/implicit_hate/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
