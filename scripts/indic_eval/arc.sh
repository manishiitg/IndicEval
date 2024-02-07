# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# -------------------------------------------------------------
#                       ARC-Easy
# -------------------------------------------------------------
model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on arc easy ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1

model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on arc easy ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/aditi-v1-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/aditi-v1-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1

model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on arc ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on arc challenge ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1

model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on arc challenge ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/aditi-v1-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/aditi-v1-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on arc ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai2_arc" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# -------------------------------------------------------------
#                       Indic ARC-Easy
# -------------------------------------------------------------
model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on indic arc easy ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy-hi/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy-hi/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on indic arc easy ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy-hi/aditi-v1-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy-hi/aditi-v1-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on arc ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy-hi/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "easy" \
    --save_dir "/sky-notebook/eval-results/arc-easy-hi/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# -------------------------------------------------------------
#                       ARC-Challenge
# -------------------------------------------------------------
model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on indic arc challenge ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge-hi/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge-hi/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1

model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on indic arc challenge ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge-hi/aditi-v1-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge-hi/aditi-v1-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on arc ..."

# zero-shot
python3 -m eval.arc.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge-hi/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# 5-shot
python3 -m eval.arc.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/ai2_arc-hi" \
    --subset "challenge" \
    --save_dir "/sky-notebook/eval-results/arc-challenge-hi/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
