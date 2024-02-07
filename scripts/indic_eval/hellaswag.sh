# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# -------------------------------------------------------------
#                       Hellaswag
# -------------------------------------------------------------

model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on hellaswag ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --save_dir "/sky-notebook/eval-results/hellaswag/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --save_dir "/sky-notebook/eval-results/hellaswag/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on hellaswag ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --save_dir "/sky-notebook/eval-results/hellaswag/aditi-v1-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --save_dir "/sky-notebook/eval-results/hellaswag/aditi-v1-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1

model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on hellaswag ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --save_dir "/sky-notebook/eval-results/hellaswag/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --save_dir "/sky-notebook/eval-results/hellaswag/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# -------------------------------------------------------------
#                       Indic Hellaswag
# -------------------------------------------------------------

model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on hellaswag-hi ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "/sky-notebook/eval-results/hellaswag-hi/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "/sky-notebook/eval-results/hellaswag-hi/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1

model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on hellaswag-hi ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "/sky-notebook/eval-results/hellaswag-hi/aditi-v1-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "/sky-notebook/eval-results/hellaswag-hi/aditi-v1-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on hellaswag ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "/sky-notebook/eval-results/hellaswag-hi/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "/sky-notebook/eval-results/hellaswag-hi/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
