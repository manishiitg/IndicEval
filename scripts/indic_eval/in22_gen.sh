# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "/sky-notebook/eval-results/in22-gen/aditi-v2-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8

# 5-shot
python3 -m eval.in22.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "/sky-notebook/eval-results/in22-gen/aditi-v2-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "/sky-notebook/eval-results/in22-gen/aditi-v1-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8

# 5-shot
python3 -m eval.in22.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "/sky-notebook/eval-results/in22-gen/aditi-v1-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "/sky-notebook/eval-results/in22-gen/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# 5-shot
python3 -m eval.in22.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "/sky-notebook/eval-results/in22-gen/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
