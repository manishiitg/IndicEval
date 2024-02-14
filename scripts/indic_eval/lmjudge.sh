#!/bin/bash

source ./common_vars.sh
FOLDER_BASE=/sky-notebook/eval-results/lmjudge

TASK_NAME=lmjudge
for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}"
    FILE=$FOLDER/lm_judge_predictions.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi
    if [ ! -f "$FILE" ]; then
        python3 -m eval.lm_judge.run_eval \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 1 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param
    fi
done

FILE=$FOLDER/lm_judge_predictions.json
model_name_or_path=ai4bharat/Airavata
model_name=Airavata
FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}"
if [ ! -f "$FILE" ]; then
    python3 -m eval.lm_judge.run_eval \
        --save_dir $FOLDER \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name_or_path $model_name_or_path \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
fi


model_name_or_path=sarvamai/OpenHathi-7B-Hi-v0.1-Base
model_name=OpenHathi-7B-Hi-v0.1-Base
FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}"
if [ ! -f "$FILE" ]; then
    python3 -m eval.lm_judge.run_eval \
        --save_dir $FOLDER \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name_or_path $model_name_or_path \
        --eval_batch_size 1 \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format
fi