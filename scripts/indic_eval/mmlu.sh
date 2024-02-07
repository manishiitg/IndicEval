# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# -------------------------------------------------------------
#                       MMLU
# -------------------------------------------------------------

model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on mmlu ..."

# zero-shot
python3 -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir "/sky-notebook/eval-results/mmlu/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir "/sky-notebook/eval-results/mmlu/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on mmlu ..."

# zero-shot
python3 -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir "/sky-notebook/eval-results/mmlu/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# 5-shot
python3 -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir "/sky-notebook/eval-results/mmlu/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# -------------------------------------------------------------
#                       Indic MMLU
# -------------------------------------------------------------

model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on indic mmlu ..."

# zero-shot
python3 -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu_hi_translated \
    --save_dir "/sky-notebook/eval-results/mmlu-hi/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu_hi_translated \
    --save_dir "/sky-notebook/eval-results/mmlu-hi/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on indic mmlu ..."

# zero-shot
python3 -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu_hi_translated \
    --save_dir "/sky-notebook/eval-results/mmlu-hi/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# 5-shot
python3 -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu_hi_translated \
    --save_dir "/sky-notebook/eval-results/mmlu-hi/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
