model_names=(
    # "LoneStriker/Smaug-72B-v0.1-AWQ"
    "Qwen/Qwen1.5-72B-Chat-AWQ"
    "manishiitg/open-aditi-hi-v2-awq"
    "manishiitg/open-aditi-hi-v1-awq"
    "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"
    "manishiitg/open-aditi-hi-v2-dpo-awq-1.1"
    "manishiitg/open-aditi-hi-v2"
)
FOLDER_BASE=/sky-notebook/eval-results/lmjudge

TASK_NAME=lmjudge
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi

    python3 -m eval.lm_judge.run_eval \
        --save_dir $FOLDER \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name_or_path $model_name_or_path \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
        --use_vllm \
        $awq_param
done

model_name_or_path=ai4bharat/Airavata
model_name=Airavata
FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}"
python3 -m eval.lm_judge.run_eval \
    --save_dir $FOLDER \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm


model_name_or_path=sarvamai/OpenHathi-7B-Hi-v0.1-Base
model_name=OpenHathi-7B-Hi-v0.1-Base
FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}"
python3 -m eval.lm_judge.run_eval \
    --save_dir $FOLDER \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --use_vllm