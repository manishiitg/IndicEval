# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


model_name_or_path="manishiitg/open-aditi-hi-v2"

echo "evaluating open-aditi-v2 base on indicqa ..."

# no-context
python3 -m eval.indicqa.run_eval \
    --ntrain 1 \
    --max_context_length 768 \
    --no_context \
    --save_dir "/sky-notebook/eval-results/indicqa/aditi-v2-1shot-no-context" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# with context
python3 -m eval.indicqa.run_eval \
    --ntrain 1 \
    --max_context_length 768 \
    --save_dir "/sky-notebook/eval-results/indicqa/aditi-v2-1shot-with-context" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 2

model_name_or_path="manishiitg/open-aditi-hi-v1"

echo "evaluating open-aditi-v1 base on indicqa ..."

# no-context
python3 -m eval.indicqa.run_eval \
    --ntrain 1 \
    --max_context_length 768 \
    --no_context \
    --save_dir "/sky-notebook/eval-results/indicqa/aditi-v1-1shot-no-context" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# with context
python3 -m eval.indicqa.run_eval \
    --ntrain 1 \
    --max_context_length 768 \
    --save_dir "/sky-notebook/eval-results/indicqa/aditi-v1-1shot-with-context" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 2


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on indicqa ..."

# no-context
python3 -m eval.indicqa.run_eval \
    --ntrain 1 \
    --max_context_length 768 \
    --no_context \
    --save_dir "/sky-notebook/eval-results/indicqa/airavata-1shot-no-context" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# with context
python3 -m eval.indicqa.run_eval \
    --ntrain 1 \
    --max_context_length 768 \
    --save_dir "/sky-notebook/eval-results/indicqa/airavata-1shot-with-context" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 2 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
